import torch
import torch.nn as nn
from torchvision.models import resnet50
# from dataos.imgdataset import ViewDataset
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import argparse
from torchvision.io import read_image           # loads [C,H,W] uint8
from torchvision.transforms.functional import convert_image_dtype
from pathlib import Path
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def validate(model, dataloader, loss_fn, device,tbwriter,numepoch):
    model.eval()
    val_loss = 0.0
    count = 0
    for images, gt, gtpath in dataloader:
        images, gt = images.to(device), gt.to(device)
        images= images.squeeze(1)
        gt = gt.squeeze(1).permute(0, 3, 1, 2)
        pred = model(images)
        pred = torch.clamp(pred, 0.0, 1.0) 
        loss = loss_fn(pred,gt)
        val_loss += loss.item()
        gtimg = read_image(gtpath[0])           # [C,H,W] uint8
        gtimg = convert_image_dtype(gtimg, torch.float32).to(device)
        gtimg = torch.nn.functional.pad(gtimg, (3,3,3,3), mode="constant", value=0)
        stripuq = torch.cat([preds[0:1], gt[0:1]], dim=3)
        tbwriter.add_images(f"Loss/uqpred-uqgt-epoch{numepoch}",stripuq,count)
        count+=1
    tbwriter.add_scalar(f"Loss/train-epoch", val_loss / len(dataloader), numepoch)
    return val_loss / len(dataloader)

class ResNet50Regressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # 1. Encoder: ResNet-50 without global-pool & fc
        weight_path = "pretrained/resnet50-19c8e357.pth"
        state_dict = torch.load(weight_path, map_location="cpu")
        # 1. Encoder: ResNet-50 without global-pool & fc
        backbone = resnet50(pretrained=False)
        backbone.load_state_dict(state_dict)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # B×2048×17×17

        # 2. Decoder: 1×1 bottleneck + upsampling
        self.reduce = nn.Conv2d(2048, 512, 1, bias=False)

        # Upsample 17×17 → 518×518 via transposed convolutions (exact arithmetic)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 34×34
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 68×68
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 136×136
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),  # 272×272
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  1,   kernel_size=4, stride=2, padding=1),  # 544×544
        )

        # Exact crop to 518×518
        self.crop = nn.AdaptiveAvgPool2d((518, 518))

    def forward(self, x):
        x = self.encoder(x)      # B×2048×17×17
        x = self.reduce(x)
        x = self.up(x)           # B×1×544×544
        x = self.crop(x)         # B×1×518×518
        return x                 # values in (-∞, +∞)

# ------------------------------------------------------------------
# Loss & training loop
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    model = ResNet50Regressor(pretrained=True).to(device)
    loss_fn = nn.L1Loss()          # regression, continuous 0-1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 150
    # Data
    train_set = ViewDataset(root='',mode='train')   # <-- replace
    val_set   = ViewDataset(root='',mode='test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
    tbwriter = SummaryWriter(log_dir="logs")
    out_dir = Path(args.save_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    best_val = 999999
    for epoch in range(num_epochs):
        for step, (images, gt, gtpath) in enumerate(train_loader):
            images = images.to(device, non_blocking=True).squeeze(1)    # B×3×518×518
            gt     = gt.to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2) # B×1×518×518  range [0,1]
            preds = model(images)                     # B×1×518×518
            preds = torch.clamp(preds, 0.0, 1.0)  
            loss  = loss_fn(preds, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f'Epoch {epoch} | step {step:04d} | L1 {loss.item():.4f}')
                if tbwriter is not None:
                    gtimg = read_image(gtpath[0])           # [C,H,W] uint8
                    gtimg = convert_image_dtype(gtimg, torch.float32).to(device)
                    gtimg = torch.nn.functional.pad(gtimg, (3,3,3,3), mode="constant", value=0)
                    tbwriter.add_scalar(f"Loss/loss-epoch{epoch}", loss.item(), step)
                    strip = torch.cat([images[0:1], gtimg[None]], dim=3)
                    stripuq = torch.cat([preds[0:1], gt[0:1]], dim=3)
                    tbwriter.add_images(f"Loss/img-gt-epoch{epoch}",strip,step)
                    tbwriter.add_images(f"Loss/uqpred-uqgt-epoch{epoch}",stripuq,step)
        if epoch%1==0:
            val_loss = validate(model,val_loader,loss_fn,device,tbwriter,epoch)
            torch.save(model, out_dir / "last.pth")
            if val_loss < best_val:
                best_val = val_loss
                (out_dir / "last.pth").replace(out_dir / "best.pth")
    tbwriter.close()

if __name__ =='__main__':
    main()