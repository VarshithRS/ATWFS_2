import argparse
import logging
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

from utils.config import load_config
from utils.logging_utils import setup_logging
from utils.seed import get_device
from training.train import load_model_from_checkpoint
from models.unet_mobilenet import evidential_uncertainty

logger = logging.getLogger(__name__)

class ExternalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, cfg):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.cfg = cfg
        
        self.images = sorted(list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg")))
        
        w, h = cfg["data"]["image_size"]
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=cfg["data"]["imagenet_mean"], std=cfg["data"]["imagenet_std"])
        ])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png") # Assume mask has same stem and .png
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Resize mask using nearest neighbor
        w, h = self.cfg["data"]["image_size"]
        mask = mask.resize((w, h), Image.NEAREST)
        
        img_t = self.transform(img)
        mask_t = torch.from_numpy(np.array(mask)).long()
        # binary mask 1 = drivable
        
        return {"image": img_t, "mask": mask_t, "path": str(img_path)}


def run_external_eval():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-csv", default="outputs/reports/external_eval.csv")
    a = ap.parse_args()

    setup_logging()
    cfg = load_config(a.config)
    device = get_device()
    
    model, _ = load_model_from_checkpoint(a.checkpoint, device)
    model.eval()

    ds = ExternalDataset(a.images, a.masks, cfg)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    total_inter = 0
    total_union = 0
    total_inter_bg = 0
    total_union_bg = 0
    
    ece_bins = cfg["train"]["ece_bins"]
    conf_sum = np.zeros(ece_bins)
    acc_sum = np.zeros(ece_bins)
    bin_counts = np.zeros(ece_bins)

    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            
            out = model(images)
            logits = out["seg_logits"]
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            max_conf, _ = probs.max(dim=1)
            
            # Compute per-image stats
            for i in range(images.size(0)):
                p = preds[i]
                t = targets[i]
                
                inter = ((p == 1) & (t == 1)).sum().item()
                union = ((p == 1) | (t == 1)).sum().item()
                iou = inter / max(union, 1e-6)
                
                results.append({"path": batch["path"][i], "iou": iou})
                
                # Global stats
                total_inter += inter
                total_union += union
                
                inter_bg = ((p == 0) & (t == 0)).sum().item()
                union_bg = ((p == 0) | (t == 0)).sum().item()
                total_inter_bg += inter_bg
                total_union_bg += union_bg
                
                # ECE
                conf = max_conf[i].cpu().numpy()
                acc = (p == t).float().cpu().numpy()
                
                bin_idx = np.clip(np.floor(conf * ece_bins).astype(int), 0, ece_bins - 1)
                for b, c, a_ in zip(bin_idx.flatten(), conf.flatten(), acc.flatten()):
                    bin_counts[b] += 1
                    conf_sum[b] += c
                    acc_sum[b] += a_

    drivable_iou = total_inter / max(total_union, 1e-6)
    bg_iou = total_inter_bg / max(total_union_bg, 1e-6)
    miou = (drivable_iou + bg_iou) / 2.0
    
    valid_bins = bin_counts > 0
    bin_accs = acc_sum[valid_bins] / bin_counts[valid_bins]
    bin_confs = conf_sum[valid_bins] / bin_counts[valid_bins]
    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts[valid_bins] / bin_counts.sum()))

    logger.info(f"Drivable IoU: {drivable_iou:.4f}")
    logger.info(f"mIoU: {miou:.4f}")
    logger.info(f"ECE: {ece:.4f}")

    # Sort results by iou ascending (worst first)
    results.sort(key=lambda x: x["iou"])
    
    out_path = Path(a.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "iou"])
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Per-image scores written to {out_path}")

if __name__ == "__main__":
    run_external_eval()
