import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import DeformConv2d
from pytorch_lightning import seed_everything
from einops import rearrange
import numbers

import cv2 
import os 
import math 
from torchvision.models import mobilenet_v2

import sys
sys.path.append('core')
import argparse
from raft import RAFT
from utils_raft import flow_viz
from utils_raft.utils import InputPadder

from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=40)

seed_everything(13)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(pl.LightningModule):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(pl.LightningModule):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(pl.LightningModule):
    def __init__(self, dim, num_heads, stride, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.stride = stride
        self.qk = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=self.stride, padding=1, groups=dim*2, bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qk = self.qk_dwconv(self.qk(x))
        q,k = qk.chunk(2, dim=1)
        
        v = self.v_dwconv(self.v(x))
        
        b, f, h1, w1 = q.size()

        q = rearrange(q, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class BFA(pl.LightningModule):
    def __init__(self, dim, num_heads, stride, ffn_expansion_factor, bias, LayerNorm_type):
        super(BFA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, stride, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(pl.LightningModule):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class alignment(pl.LightningModule):
    def __init__(self, dim=48, memory=False, stride=1, type='group_conv'):
        
        super(alignment, self).__init__()
        
        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)

        self.offset_conv = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        self.back_projection = ref_back_projection(dim, stride=1)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        
        if memory==True:
            self.bottleneck_o = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
            
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def forward(self, x, prev_offset_feat=None):
        
        B, f, H, W = x.size()
        ref = x[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, B, dim=0)

        offset_feat = self.bottleneck(torch.cat([ref, x], dim=1))

        if not prev_offset_feat==None:
            offset_feat = self.bottleneck_o(torch.cat([prev_offset_feat, offset_feat], dim=1))

        offset, mask = self.offset_gen(self.offset_conv(offset_feat)) 

        aligned_feat = self.deform(x, offset, mask)
        aligned_feat[0] = x[0].unsqueeze(0)

        aligned_feat = self.back_projection(aligned_feat)
        
        return aligned_feat, offset_feat


class EDA(pl.LightningModule):
    def __init__(self, in_channels=48):
        super(EDA, self).__init__()
        
        num_blocks = [4,6,6,8] 
        num_refinement_blocks = 4
        heads = [1,2,4,8]
        bias = False
        LayerNorm_type = 'WithBias'

        self.encoder_level1 = nn.Sequential(*[BFA(dim=in_channels, num_heads=heads[0], stride=1, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
        self.encoder_level2 = nn.Sequential(*[BFA(dim=in_channels, num_heads=heads[1], stride=1, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
                
        self.down1 = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)        
        self.down2 = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

        self.alignment0 = alignment(in_channels, memory=True)
        self.alignment1 = alignment(in_channels, memory=True)
        self.alignment2 = alignment(in_channels)
        self.cascade_alignment = alignment(in_channels, memory=True)

        self.offset_up1 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        self.offset_up2 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)

        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)        
        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = self.encoder_level1(x)
        enc1 = self.down1(x)

        enc1 = self.encoder_level2(enc1)
        enc2 = self.down2(enc1)
        enc2, offset_feat_enc2 = self.alignment2(enc2)
        
        dec1 = self.up2(enc2)
        offset_feat_dec1 = self.offset_up2(offset_feat_enc2) * 2
        enc1, offset_feat_enc1 = self.alignment1(enc1, offset_feat_dec1)
        dec1 = dec1 + enc1

        dec0 = self.up1(dec1)
        offset_feat_dec0 = self.offset_up1(offset_feat_enc1) * 2
        x, offset_feat_x = self.alignment0(x, offset_feat_dec0)
        x = x + dec0

        alinged_feat, offset_feat_x = self.cascade_alignment(x, offset_feat_x)    
        
        return alinged_feat

class ref_back_projection(pl.LightningModule):
    def __init__(self, in_channels, stride):

        super(ref_back_projection, self).__init__()

        bias = False
        
        self.feat_fusion = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1), nn.GELU())        
        self.feat_expand = nn.Sequential(nn.Conv2d(in_channels, in_channels*2, 3, stride=1, padding=1), nn.GELU())
        self.diff_fusion = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1), nn.GELU())

        self.encoder1 = nn.Sequential(*[BFA(dim=in_channels*2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(2)])
        
    def forward(self, x):
        
        B, f, H, W = x.size()

        ref = x[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, B, dim=0)
        feat = self.encoder1(torch.cat([ref, x], dim=1))

        fused_feat = self.feat_fusion(feat)
        exp_feat = self.feat_expand(fused_feat)

        residual = exp_feat - feat
        residual = self.diff_fusion(residual)

        fused_feat = fused_feat + residual

        return fused_feat

class no_ref_back_projection(pl.LightningModule):
    def __init__(self, in_channels, stride):

        super(no_ref_back_projection, self).__init__()

        bias = False
        
        self.feat_fusion = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1), nn.GELU())        
        self.feat_expand = nn.Sequential(nn.Conv2d(in_channels, in_channels*2, 3, stride=1, padding=1), nn.GELU())
        self.diff_fusion = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1), nn.GELU())
        
        self.encoder1 = nn.Sequential(*[BFA(dim=in_channels*2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(2)])

    def burst_fusion(self, x):
        b, f, H, W = x.size()
        x = x.view(-1, f*2, H, W)
        return x

    def forward(self, x):
        B, f, H, W = x.size()
        shifted_x = torch.roll(x, 1, 0)
        feat = x.view(-1, f*2, H, W)
        shifted_feat = shifted_x.view(-1, f*2, H, W)
        feat = self.encoder1(torch.cat([feat, shifted_feat], dim=0))

        fused_feat = self.feat_fusion(feat)
        rec_feat = self.feat_expand(fused_feat)

        residual = self.diff_fusion(feat - rec_feat)
        fused_feat = fused_feat + residual
        
        return fused_feat


class adapt_burst_pooling(pl.LightningModule):
    def __init__(self, in_channels, out_burst_num):

        super(adapt_burst_pooling, self).__init__()

        cur_burst_num = out_burst_num - 1
        self.adapt_burst_pool = nn.AdaptiveAvgPool1d(in_channels*cur_burst_num) 

    def forward(self, x):

        B, f, H, W = x.size()
        x_ref = x[0].unsqueeze(0)        
        x = x.view(-1, H, W)
        x = x.permute(1, 2, 0).contiguous()
        x = self.adapt_burst_pool(x)
        x = x.permute(2, 0, 1).contiguous()
        x = x.view(-1, f, H, W)
        x = torch.cat([x_ref, x], dim=0)

        return x


class BAENet(pl.LightningModule):
    def __init__(self):
        super(BAENet, self).__init__()
        
        self.mobilenet = mobilenet_v2(pretrained=False)

        original_conv1 = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=5,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        self.mobilenet.classifier[1] = nn.Linear(
            self.mobilenet.last_channel, 5
        )

        # Xavier Initialization
        self.apply(self.init_weights)

        # define optical flow network
        parser_raft = argparse.ArgumentParser()
        parser_raft.add_argument('--small', action='store_false', help='use small model')
        parser_raft.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser_raft.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args_raft = parser_raft.parse_args()
        self.raft = RAFT(args_raft)
        ckpt_name = "core/model_weight/raft-small.pth"
        checkpoint = torch.load(ckpt_name)
        print(ckpt_name)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.raft.load_state_dict(checkpoint)
        for param in self.raft.parameters():
            param.requires_grad = False

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2, x, meta_info, total_exp_time):
        
        B, C, H, W = x.size()

        padder = InputPadder(x1.shape)
        x1, x2 = padder.pad(x1, x2)
        flow_low, flow_up = self.raft(x1*255., x2*255., iters=20, test_mode=True)
        flo = flow_up[0].permute(1,2,0)
        optflow_val = torch.mean(torch.norm(flo, dim=2)).item()
        optflow_val = min(optflow_val, 10.0) / 10.0

        motion_channel = torch.full((B, 1, H, W), optflow_val, device=x.device)
        x = torch.cat((x, motion_channel), dim=1)  

        iso_channel = torch.full((B, 1, H, W), meta_info['pred_iso'].item(), device=x.device)
        x = torch.cat((x, iso_channel), dim=1)

        x = self.mobilenet(x)
        # Output transformation to the desired range
        x = self.bounded_softmax(x, 1 / total_exp_time)
        return x
    
    def bounded_softmax(self, logits, min_prob):
        n_classes = logits.size(-1)
        scale = 1 - (n_classes - 1) * min_prob
        softmax_output = F.softmax(logits, dim=-1)
        
        mask = torch.ones_like(softmax_output)
        mask[..., -1] = 0
        
        return scale * softmax_output + min_prob * mask


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class CeilSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class BurstSimulator(pl.LightningModule):
    '''
    To synthesize an exposure of x/1920 seconds, we average (x-7) interpolated frames.
    The base exposure is 8/1920 seconds (the original frame), and each additional interpolated frame is interpreted as a 1/1920 second increment,
    enabling finer temporal integration through accumulation.
    '''

    def __init__(self):
        super(BurstSimulator, self).__init__()

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(2)
        self.round_ste = RoundSTE.apply
        self.floor_ste = FloorSTE.apply
        self.ceil_ste = CeilSTE.apply
        
        self.min_exp = 1/240.

    def forward(self, pred_exps, irr_seq, meta_info, batch_idx):
        
        self.exposures = (irr_seq.shape[1]-1-7*3) * pred_exps.squeeze()
        self.raws = torch.zeros(1, 4, 4, irr_seq.shape[3]//2, irr_seq.shape[4]//2).to(irr_seq.device)

        dp = self.exposures[0] - self.floor_ste(self.exposures[0])
        tensor_ceil_int = int(math.ceil(self.exposures[0].item()))
        if tensor_ceil_int % 2 == 0: # even
            if dp != 0:
                center = (self.exposures[0]-dp+1.) / 2.
            else:
                center = (self.exposures[0]-dp) / 2.
        else: # odd
            if dp != 0:
                center = (self.exposures[0]-dp) / 2.
            else:
                center = (self.exposures[0]-dp+1.) / 2.
        
        for i in range(len(self.exposures)-1):
            try:
                curr_exposure = self.exposures[i]
                next_exposure = self.exposures[i+1]
                
                f_onehot = torch.zeros(irr_seq.shape[1], dtype=irr_seq.dtype, device=irr_seq.device)
                c_onehot = torch.zeros(irr_seq.shape[1], dtype=irr_seq.dtype, device=irr_seq.device)
                
                f_odd, c_odd = self.floor_ceil_odd(curr_exposure)
                
                center_dp = center - self.floor_ste(center)
                fs_idx = int(math.floor(center))-(int(f_odd.item())-1)//2
                fe_idx = int(math.floor(center))+(int(f_odd.item())-1)//2+1
                f_onehot[fs_idx:fe_idx] = 1.
                f_onehot[fs_idx] -= center_dp
                f_onehot[fe_idx] += center_dp

                cs_idx = int(math.floor(center))-(int(c_odd.item())-1)//2
                ce_idx = int(math.floor(center))+(int(c_odd.item())-1)//2
                if not torch.allclose(f_odd, curr_exposure, atol=1e-6):
                    ce_idx += 1
                c_onehot[cs_idx:ce_idx] = 1.
                c_onehot[cs_idx] -= center_dp
                c_onehot[ce_idx] += center_dp

                r = (curr_exposure - f_odd) / (c_odd - curr_exposure) 
                onehot = (f_onehot + r * c_onehot) / (1 + r)
                
                if onehot[fs_idx] < 1.:
                    tmp = onehot[fs_idx] + onehot[cs_idx]
                    if torch.allclose(tmp, torch.ones_like(tmp), atol=1e-6):
                        onehot[fs_idx] = tmp
                        onehot[cs_idx] = 0.
                    elif tmp > 1.0:
                        onehot[cs_idx] = tmp - 1.0
                        onehot[fs_idx] = self.floor_ste(tmp)
                    else: # tmp < 1.0 
                        onehot[fs_idx] = tmp
                        onehot[cs_idx] = self.floor_ste(tmp)
                if onehot[fe_idx] < 1.:
                    tmp = onehot[fe_idx] + onehot[ce_idx]
                    if torch.allclose(tmp, torch.ones_like(tmp), atol=1e-6):
                        onehot[fe_idx] = tmp
                        onehot[ce_idx] = 0.
                    elif tmp > 1.0:
                        onehot[ce_idx] = tmp - 1.0
                        onehot[fe_idx] = self.floor_ste(tmp)
                    else: # tmp < 1.0
                        onehot[fe_idx] = tmp
                        onehot[ce_idx] = self.floor_ste(tmp)
                
                # shifting to fully use first frame
                if i == 0:
                    shift_val = 1. - onehot[0]

                    nonzero_indices = torch.nonzero(onehot > 0, as_tuple=False).squeeze()
                    max_idx = torch.max(nonzero_indices).item()
                    
                    onehot[0] = onehot[0] + shift_val
                    if shift_val > 0.5:
                        onehot[max_idx-1] = onehot[max_idx-1] + onehot[max_idx] - shift_val 
                        onehot[max_idx] = 0.
                        center = center - 1.
                    elif shift_val > 0:
                        onehot[max_idx] = onehot[max_idx] - shift_val
                        center = center - 0.5
                    center += self.floor_ste(curr_exposure) / 2. + dp + (next_exposure + 14.) / 2
                else: 
                    center += (curr_exposure + next_exposure + 14.) / 2
            except IndexError as e:
                import pdb;pdb.set_trace()
            
            onehot_IDX = torch.nonzero(onehot).squeeze(1)
            curr_irrs = irr_seq[:,onehot_IDX] # irr_seq >> (b, 1281, c, h, w)
            
            curr_irrs = self.gamma_reverse(curr_irrs) # gamma expansion
            curr_irrs *= onehot[onehot_IDX].view(1,curr_irrs.shape[1],1,1,1)
            curr_irrs = torch.sum(curr_irrs, axis=1) / curr_exposure # blur synthesis
            curr_irrs = curr_irrs.squeeze(0)
            curr_irrs = self.apply_ccm(curr_irrs, meta_info['rgb2cam'].squeeze()) # lin to cam color space
            curr_irrs = self.apply_gains(curr_irrs, 1/meta_info['rgb_gain'], 1/meta_info['red_gain'], 1/meta_info['blue_gain'], clamp=False) # inverse white-balance
            curr_irrs = self.mosaic(curr_irrs.unsqueeze(0)) # mosaic (3ch to 1ch)
            # add noise (heteroscedastic Gaussian random noise)
            curr_exp_time = 1/1920. * (curr_exposure + 7.)
            curr_iso = meta_info['max_iso'] * self.min_exp / curr_exp_time
            if meta_info["split"][0] == "test":
                shot_noise, read_noise = self.random_noise_levels_test(curr_iso)
            else: # "train"
                shot_noise, read_noise = self.random_noise_levels(curr_iso)
            scale_var = read_noise + shot_noise * curr_irrs
            # reparameterization trick
            epsilon = torch.randn_like(scale_var)
            total_noise = torch.sqrt(scale_var) * epsilon
            curr_irrs = (curr_irrs + total_noise).clip(1e-8,1)
            
            self.raws[:,i] = self.pixel_unshuffle(curr_irrs)
        
        return self.raws, self.exposures[:-1]
    
    def floor_ceil_odd(self, tensor):
        tensor_ro = self.round_ste(tensor)
        tensor_ro_int = int(tensor_ro.item())

        if tensor_ro_int % 2 == 0:
            return tensor_ro - 1., tensor_ro + 1.
        elif tensor_ro_int % 2 != 0 and tensor >= tensor_ro:
            return tensor_ro, tensor_ro + 2.
        elif tensor_ro_int % 2 != 0 and tensor < tensor_ro:
            return tensor_ro - 2., tensor_ro
        else:
            import pdb;pdb.set_trace()

    def gather(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """ Gathers pixels from `x` according to `index` which follows bayer pattern.
        NOTE: Avoid using `torch.gather` since it does not supports deterministic mode.

        Args:
            x: Tensor of shape :math:`(N, C_{in}, H, W)`.
            index: Gather index with bayer pattern of shape :math:`(1|N, C_{out}, 2, 2)`.

        Returns:
            Gathered values of shape :math:`(N, C_{out}, H, W)`.
        """
        assert x.dim()     == 4, f"`x` must be 4-dimensional, got {x.dim()}"
        assert index.dim() == 4, f"`index` must be 4-dimensional, got {index.dim()}"

        xN, xC, xH, xW = x.shape
        iN, iC, iH, iW = index.shape
        C_in, C_out = xC, iC

        assert iN == xN or iN == 1, \
            f"Batch dimension of `index` must be `1` or equal to `x`, got {iN}"
        assert xH % 2 == 0 and xW % 2 == 0, \
            f"`x` must have even height and width, got {xH}x{xW}"
        assert iH == 2 and iW == 2, \
            f"`index` must be 2x2 size, got {iH}x{iW}"

        # x_down: (N, 4*C_in, H/2, W/2)
        x_down = F.pixel_unshuffle(x, 2)
        # x_quat: (N, 4, 1, C_in, H/2, W/2)
        x_quat = x_down.reshape((xN, -1, 4, xH//2, xW//2)).swapaxes(1, 2).unsqueeze(2)

        # i_down: (1|N, 4*C_out)
        i_down = F.pixel_unshuffle(index, 2).squeeze(-1).squeeze(-1)
        # i_quat: (1|N, 4, C_out, 1)
        i_quat = i_down.reshape((iN, -1, 4)).swapaxes(1, 2).unsqueeze(3)

        # mask_index: (N, 4, C_out, C_in)
        mask_index = torch.arange(C_in, dtype=index.dtype, device=index.device).repeat(xN, 4, C_out, 1)
        # mask: (N, 4, C_out, C_in)
        mask = (mask_index == i_quat)

        # x_gath: (N, 4, C_out, H/2, W/2)
        x_gath = (x_quat * mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=3)

        # y_down: (N, 4*C_out, H/2, W/2)
        y_down = x_gath.swapaxes(1, 2).flatten(start_dim=1, end_dim=2)
        # y: (N, C_out, H, W)
        y = F.pixel_shuffle(y_down, 2)

        return y

    def mosaic(self, x: torch.Tensor) -> torch.Tensor:
        """ Mosaicing RGB images to Bayer pattern. 

        Args:
            x: RGB image of shape :math:`(N, 3, H, W)`.
            bayer_pattern: Bayer pattern of `x` of shape :math:`(1|N, 1, 2, 2)`.

        Returns:
            Mosaicked image of shape :math:`(N, 1, H, W)`.
        """
        
        bayer_pattern = torch.tensor([[[[0,1],[1,2]]]]).to(x.device)
        
        assert x.dim()             == 4, f"`x` must be 4-dimensional, got {x.dim()}"
        assert bayer_pattern.dim() == 4, f"`bayer_pattern` must be 4-dimensional, got {bayer_pattern.dim()}"

        xN, xC, xH, xW = x.shape
        bN, bC, bH, bW = bayer_pattern.shape

        assert xC == 3, f"Channel dimension of `x` must be 3, got {xC}"
        assert xH % 2 == 0 and xW % 2 == 0, f"`x` must have even height and width, got {xH}x{xW}"
        assert bN == xN or bN == 1, f"Batch dimension of `bayer_mask` must be `1` or equal to `x`, got {bN}"
        assert bC == 1, f"Channel dimension of `bayer_mask` must be 1, got {bC}"
        assert bH == 2 and bW == 2, f"Height and width dimension of `bayer_mask` must be 2, got {bH}x{bW}"

        y = self.gather(x, bayer_pattern)
        return y
    
    def random_noise_levels(self, iso):
        """Generates random noise levels from a log-log linear distribution."""
        iso2shot = lambda x: 9.2857e-07 * x + 8.1006e-05
        shot_value = iso2shot(iso)
        shot_noise = shot_value + torch.normal(mean=0.0, std=5e-05, size=shot_value.size()).to(shot_value.device)

        while shot_noise <= 0:
            shot_value = iso2shot(iso)
            shot_noise = shot_value + torch.normal(mean=0.0, std=5e-05, size=shot_value.size()).to(shot_value.device)

        log_shot_noise = torch.log(shot_noise)
        logshot2logread = lambda x: 2.2282 * x + 0.45982
        logread_value = logshot2logread(log_shot_noise)
        log_read_noise = logread_value + torch.normal(mean=0.0, std=0.25, size=logread_value.size()).to(logread_value.device)
        read_noise = torch.exp(log_read_noise)

        while read_noise <= 0:
            logread_value = logshot2logread(log_shot_noise)
            log_read_noise = logread_value + torch.normal(mean=0.0, std=0.25, size=logread_value.size()).to(logread_value.device)
            read_noise = torch.exp(log_read_noise)

        return shot_noise, read_noise

    def random_noise_levels_test(self, iso):
        """Generates random noise levels from a log-log linear distribution."""
        iso2shot = lambda x: 9.2857e-07 * x + 8.1006e-05
        shot_value = iso2shot(iso)
        shot_noise = shot_value

        while shot_noise <= 0:
            shot_value = iso2shot(iso)
            shot_noise = shot_value

        log_shot_noise = torch.log(shot_noise)
        logshot2logread = lambda x: 2.2282 * x + 0.45982
        logread_value = logshot2logread(log_shot_noise)
        log_read_noise = logread_value
        read_noise = torch.exp(log_read_noise)

        while read_noise <= 0:
            logread_value = logshot2logread(log_shot_noise)
            log_read_noise = logread_value
            read_noise = torch.exp(log_read_noise)

        return shot_noise, read_noise

    def gamma(self, pre_img):
        Mask = lambda x: (x>0.0031308).float()
        sRGBDeLinearize = lambda x,m: m * (1.055 * (m * x) ** (1/2.4) - 0.055) + (1-m) * (12.92 * x)
        return  sRGBDeLinearize(pre_img, Mask(pre_img))

    def gamma_reverse(self, pre_img): 
        Mask = lambda x: (x>0.04045).float()
        sRGBLinearize = lambda x,m: m * ((m * x + 0.055) / 1.055) ** 2.4 + (1-m) * (x / 12.92)
        return  sRGBLinearize(pre_img, Mask(pre_img))

    def apply_ccm(self, image, ccm):
        """Applies a color correction matrix."""
        assert image.dim() == 3 and image.shape[0] == 3
        shape = image.shape
        image = image.reshape(3, -1)
        ccm = ccm.to(image.device).type_as(image)

        image = torch.mm(ccm, image)

        return image.view(shape)

    def apply_gains(self, image, rgb_gain, red_gain, blue_gain, clamp=True):
        """Inverts gains while safely handling saturated pixels."""
        assert image.dim() == 3 and image.shape[0] in [3, 4]
        
        if image.shape[0] == 3:
            gains = torch.tensor([red_gain, 1.0, blue_gain]).to(image.device) * rgb_gain
        else:
            gains = torch.tensor([red_gain, 1.0, 1.0, blue_gain]).to(image.device) * rgb_gain
        gains = gains.view(-1, 1, 1)
        gains = gains.to(image.device).type_as(image)

        if clamp:
            return (image * gains).clamp(0.0, 1.0)
        else:
            return (image * gains)
    

class Burstormer(pl.LightningModule):
    def __init__(self, args=None, mode='color', num_features=48, burst_size=8, reduction=8, bias=False):
        super(Burstormer, self).__init__()

        self.train_loss1 = nn.L1Loss()
        self.valid_psnr = PSNR(boundary_ignore=10)

        if args is not None:
            self.args = args

        self.baenet = BAENet()
        self.burstsimulator = BurstSimulator()
    
    def training_step(self, train_batch, batch_idx):
        
        pred1, pred2, pred, irr_seq, y, meta_info = train_batch
        pred_exps = self.baenet(pred1, pred2, pred, meta_info, irr_seq.shape[1]-1-7*3)
        x, e = self.burstsimulator(pred_exps, irr_seq, meta_info, batch_idx) 
        
        if isinstance(meta_info['e_gt'], list) and meta_info['e_gt'][0]=="P8":
            e_gt = torch.tensor([8, 16, 24, 32], dtype=e.dtype, device=e.device)
        else:
            e_gt = meta_info['e_gt'].repeat(e.shape[0])
            
        loss = self.train_loss1(e, e_gt)
        self.log('train_loss', loss, batch_size=x.size(0), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        pred1, pred2, pred, irr_seq, y, meta_info = val_batch
        pred_exps = self.baenet(pred1, pred2, pred, meta_info, irr_seq.shape[1]-1-7*3)
        _, e = self.burstsimulator(pred_exps, irr_seq, meta_info, batch_idx)
        if isinstance(meta_info['e_gt'], list) and meta_info['e_gt'][0]=="P8":
            e_gt = torch.tensor([8, 16, 24, 32], dtype=e.dtype, device=e.device)
        else:
            e_gt = meta_info['e_gt'].repeat(e.shape[0])
        loss = self.train_loss1(e, e_gt)

        return loss

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        loss = torch.stack(outs).mean()
        self.log('val_psnr', loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, predict_batch, batch_idx):

        pred1, pred2, pred, irr_seq, y, meta_info = predict_batch
        pred_exps = self.baenet(pred1, pred2, pred, meta_info, irr_seq.shape[1]-1-7*3) 
        x = self.burstsimulator(pred_exps, irr_seq, meta_info, batch_idx) 
        pred = self.forward(x).clamp(0, 1.) 
        PSNR = self.valid_psnr(pred, y)

        psnr = (os.path.join(meta_info['clip_name'][0], "%04d" %(batch_idx % 10000)), PSNR.item())
        return psnr

    def configure_optimizers(self):        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.num_epochs, eta_min=self.args.eta_min)         
        return [optimizer], [lr_scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
