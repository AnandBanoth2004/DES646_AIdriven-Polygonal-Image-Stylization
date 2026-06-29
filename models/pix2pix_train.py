import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import models
from PIL import Image
from tqdm import tqdm
import yaml

# ----------------------------
# Data
# ----------------------------
class PairedFolderDataset(Dataset):
    """
    Expects two folders with matching filenames:
    - photos_dir: input photos
    - targets_dir: polygon targets (same filenames)
    """
    def __init__(self, photos_dir: str, targets_dir: str, image_size: int = 256, augment_cfg: dict | None = None):
        import glob, os
        self.photos = sorted([
            os.path.join(photos_dir, f)
            for f in os.listdir(photos_dir)
            if f.lower().endswith((".jpg",".jpeg",".png"))
        ])
        # Build target list by matching base names and handling synthesized suffixes like _triangle_256.png
        self.targets = []
        for p in self.photos:
            base = os.path.splitext(os.path.basename(p))[0]
            exact = os.path.join(targets_dir, base + ".png")
            if os.path.isfile(exact):
                self.targets.append(exact)
                continue
            # Try synthesized naming patterns
            cand = glob.glob(os.path.join(targets_dir, base + "_triangle_*.png"))
            if not cand:
                cand = glob.glob(os.path.join(targets_dir, base + "_rectangle_*.png"))
            if cand:
                self.targets.append(sorted(cand)[0])
            else:
                raise FileNotFoundError(f"No target found for photo '{p}'. Looked for '{exact}' or synthesized patterns in '{targets_dir}'.")

        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        self.augment_cfg = augment_cfg or {}

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        x = Image.open(self.photos[idx]).convert("RGB")
        y = Image.open(self.targets[idx]).convert("RGB")
        # Resize first
        x = self.resize(x); y = self.resize(y)
        # Paired geometric augmentation
        if self.augment_cfg.get("hflip", False) and random.random() < 0.5:
            x = TF.hflip(x); y = TF.hflip(y)
        rot_deg = self.augment_cfg.get("rotate_deg", 0)
        if rot_deg and rot_deg > 0:
            angle = random.uniform(-rot_deg, rot_deg)
            x = TF.rotate(x, angle); y = TF.rotate(y, angle)
        # Color jitter only on input (not on target)
        if any(self.augment_cfg.get(k, 0) for k in ("brightness","contrast","saturation","hue")):
            cj = transforms.ColorJitter(
                brightness=self.augment_cfg.get("brightness", 0),
                contrast=self.augment_cfg.get("contrast", 0),
                saturation=self.augment_cfg.get("saturation", 0),
                hue=self.augment_cfg.get("hue", 0),
            )
            x = cj(x)
        return self.to_tensor(x), self.to_tensor(y)

# ----------------------------
# Models (U-Net Generator & PatchGAN Discriminator)
# Based on pix2pix (Isola et al. 2017)
# ----------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=64):
        super().__init__()
        # Encoder
        self.d1 = nn.Sequential(nn.Conv2d(in_ch, features, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)) # no BN first
        self.d2 = UNetBlock(features, features*2, down=True)
        self.d3 = UNetBlock(features*2, features*4, down=True)
        self.d4 = UNetBlock(features*4, features*8, down=True)
        self.d5 = UNetBlock(features*8, features*8, down=True)
        self.d6 = UNetBlock(features*8, features*8, down=True)
        self.d7 = UNetBlock(features*8, features*8, down=True)
        self.bottleneck = nn.Sequential(nn.Conv2d(features*8, features*8, 4, 2, 1), nn.ReLU(inplace=True))
        # Decoder
        self.u1 = UNetBlock(features*8, features*8, down=False, use_dropout=True)
        self.u2 = UNetBlock(features*16, features*8, down=False, use_dropout=True)
        self.u3 = UNetBlock(features*16, features*8, down=False, use_dropout=True)
        self.u4 = UNetBlock(features*16, features*8, down=False)
        self.u5 = UNetBlock(features*16, features*4, down=False)
        self.u6 = UNetBlock(features*8, features*2, down=False)
        self.u7 = UNetBlock(features*4, features, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        bn = self.bottleneck(d7)
        u1 = self.u1(bn)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))
        out = self.final(torch.cat([u7, d1], dim=1))
        return out

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=3, features=64):
        super().__init__()
        def C(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            C(in_ch*2, features, bn=False),
            C(features, features*2),
            C(features*2, features*4),
            nn.Conv2d(features*4, 1, 4, 1, 1)  # Patch score map
        )

    def forward(self, x, y):
        # Input is concatenation of x and y
        return self.net(torch.cat([x, y], dim=1))

# ----------------------------
# VGG perceptual features
# ----------------------------
class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]

    def forward(self, x: torch.Tensor):
        # Expect [-1,1] range, normalize to ImageNet
        x = (x + 1)/2
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        x = (x - mean)/std
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return [f1, f2, f3]

# ----------------------------
# Training
# ----------------------------
def train(config_path="configs/default.yaml", photos_dir="data/images", targets_dir="data/targets/triangles_256", out_dir="checkpoints"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    beta1, beta2 = cfg["beta1"], cfg["beta2"]
    lambda_L1 = cfg["lambda_L1"]
    lambda_perc = cfg.get("lambda_perc", 0.0)
    os.makedirs(out_dir, exist_ok=True)

    ds = PairedFolderDataset(photos_dir, targets_dir, image_size=image_size, augment_cfg=cfg.get("augment", {}))
    n = len(ds)
    n_train = int(n * cfg["data"]["train_split"]) if "data" in cfg else int(n*0.9)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n - n_train])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    G = UNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    g_opt = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    d_opt = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    vgg = VGGFeatures().to(device) if lambda_perc > 0 else None

    for epoch in range(epochs):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            # Train D
            with torch.cuda.amp.autocast(enabled=False):
                fake = G(x).detach()
                d_real = D(x, y)
                d_fake = D(x, fake)
                real_lbl = torch.ones_like(d_real)
                fake_lbl = torch.zeros_like(d_fake)
                d_loss = bce(d_real, real_lbl) + bce(d_fake, fake_lbl)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # Train G
            fake = G(x)
            d_fake = D(x, fake)
            adv_loss = bce(d_fake, torch.ones_like(d_fake))
            l1_loss = l1(fake, y) * lambda_L1
            if vgg is not None:
                # Perceptual loss on features
                feats_hat = vgg(fake)
                feats = vgg(y)
                perc_loss = sum(nn.functional.l1_loss(a, b) for a,b in zip(feats_hat, feats)) * lambda_perc
            else:
                perc_loss = 0.0
            g_loss = adv_loss + l1_loss + (perc_loss if isinstance(perc_loss, torch.Tensor) else 0.0)
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()
            pbar.set_postfix(d=float(d_loss.item()), g=float(g_loss.item()))
        # Save checkpoint each epoch
        ckpt = os.path.join(out_dir, cfg["checkpoints"]["pix2pix_name"])
        torch.save({"G": G.state_dict()}, ckpt)
    print(f"Saved checkpoints to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--photos_dir", type=str, default="data/images")
    ap.add_argument("--targets_dir", type=str, default="data/targets/triangles_256")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    args = ap.parse_args()
    train(args.config, args.photos_dir, args.targets_dir, args.out_dir)
