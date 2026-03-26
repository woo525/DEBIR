import os
import torch
from pytorch_lightning.plugins import DDPPlugin
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")
import sys

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
seed_everything(13)


from Network_DEBIR import Burstormer
from datasets.dataset_stage2m_and_test import GoProRAW2RAW
from torch.utils.data.dataloader import DataLoader


log_dir = './logs/Track_1/'

class Args:
    def __init__(self):
        self.image_dir = "../resources/GOPRO_x8_DEBIR"
        self.exp_name = os.path.join(log_dir, "debir_ob")
        self.pretrained_baenet = "debir_o"
        self.learning_rate = 1e-7
        self.eta_min = 1e-10
        self.num_epochs = 300
        self.loss_type = "Burstormer Loss"
        self.burstormer_type = "NUNI_US64"
        self.patch_size = 256
        self.pred_exp = 64
        self.NUM_WORKERS = 6
        self.batch_size = 4
        self.gradient_clip = 0.01

args = Args()
if not os.path.exists(args.exp_name):
    os.makedirs(args.exp_name, exist_ok=True)

def write_class_args_to_file(args, filename=os.path.join(args.exp_name, "args.txt")):
    with open(filename, "w") as file:
        for name, value in vars(args).items():
            file.write(f"{name}: {value}\n")       
write_class_args_to_file(args)

train_dataset = GoProRAW2RAW(root=args.image_dir, split='train', patch_size=args.patch_size, args=args)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=args.NUM_WORKERS, pin_memory=True)
test_dataset = GoProRAW2RAW(root=args.image_dir,  split='test', patch_size=512, args=args)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.NUM_WORKERS, pin_memory=True)

model = Burstormer(args=args)
model.cuda()

# prediction network weights
ckpts = os.listdir("logs/Track_1/%s" %(args.pretrained_baenet))
ckpt_path = os.path.join("logs/Track_1/%s" %(args.pretrained_baenet),[file for file in ckpts if "epoch" in file][0])
baenet_ckpt = torch.load(ckpt_path)
# permute burstormer weights
bt_ckpt_path = "logs/checkpoints/%s.ckpt" %(args.burstormer_type)
bt_ckpt = torch.load(bt_ckpt_path)

model_state_dict = model.state_dict()
for name, param in baenet_ckpt['state_dict'].items():
    if "prednet" in name or "baenet" in name:
        name = name.replace("prednet","baenet") 
        model_state_dict[name].copy_(param)
for name, param in bt_ckpt['state_dict'].items():
    if "prednet" not in name and "baenet" not in name:
        model_state_dict[name].copy_(param)
for name, param in model.named_parameters():
    if "prednet" not in name and "baenet" not in name:
        param.requires_grad = False   

model.summarize()

checkpoint_callback = ModelCheckpoint(
    monitor='val_psnr',
    dirpath=args.exp_name,
    filename='{epoch:02d}-{val_psnr:.2f}',
    save_top_k=1,
    save_last=True,
    mode='max',
)

torch.use_deterministic_algorithms(True, warn_only=True)
logger = TensorBoardLogger(save_dir="lightning_logs", name="baseline", version=args.exp_name.split('/')[-1])

trainer = Trainer(gpus=-1,
                logger=logger, 
                auto_select_gpus=True,
                accelerator='ddp',
                max_epochs= 100, # early-stopping
                precision=16,
                gradient_clip_val=0.01,
                callbacks=[checkpoint_callback],
                benchmark=True,
                check_val_every_n_epoch=5,
                progress_bar_refresh_rate=1,
                plugins=DDPPlugin(find_unused_parameters=False))

trainer.fit(model, train_loader, test_loader)

