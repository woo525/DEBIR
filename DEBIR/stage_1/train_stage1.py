import os
import torch
from pytorch_lightning.plugins import DDPPlugin
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

seed_everything(13)

from Network_Burstormer_stage1 import Burstormer
from datasets.gopro_raw2raw_dataset import GoProRAW2RAW
from torch.utils.data.dataloader import DataLoader

log_dir = './logs/Track_1/'

class Args:
    def __init__(self):
        self.image_dir = "../resources/STAGE1_dataset"
        self.exp_name = os.path.join(log_dir, "NUNI_US64")
        self.learning_rate = 3e-4
        self.eta_min = 1e-8
        self.num_epochs = 500
        self.loss_type = "L1"
        self.patch_size = 256
        self.pred_exp = 32
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

def load_data(image_dir, burst_size):

    train_dataset = GoProRAW2RAW(root=image_dir,  split='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=args.NUM_WORKERS, pin_memory=True)

    test_dataset = GoProRAW2RAW(root=image_dir,  split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader

model = Burstormer(args=args)
model.cuda()
model.summarize()

train_loader, test_loader = load_data(args.image_dir, 4)

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
                max_epochs=args.num_epochs,
                precision=16,
                gradient_clip_val=0.01,
                callbacks=[checkpoint_callback],
                benchmark=True,
                check_val_every_n_epoch=10,
                progress_bar_refresh_rate=100,
                plugins=DDPPlugin(find_unused_parameters=False))

trainer.fit(model, train_loader, test_loader)
