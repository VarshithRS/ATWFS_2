import argparse
import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from utils.config import load_config
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed
from training.train import load_model_from_checkpoint
from models.unet_mobilenet import build_model

logger = logging.getLogger(__name__)

def train_terrain():
    ap = argparse.ArgumentParser(description="Train ONLY the terrain classifier head on a folder dataset.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", required=True, help="Path to folder-per-class dataset")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
    ap.add_argument("--output", default="outputs/checkpoints/student_terrain.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    setup_logging()
    cfg = load_config(a.config)
    set_global_seed(cfg["seed"])
    device = get_device()

    # Load original model
    model, _ = load_model_from_checkpoint(a.checkpoint, device)
    model.eval()

    # Create a copy for training to compare seg outputs later
    train_model = copy.deepcopy(model).to(device)

    # Freeze everything except the terrain head
    for name, param in train_model.named_parameters():
        if not name.startswith("terrain_head."):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Data Loading
    w, h = cfg["data"]["image_size"]
    
    transform_train = T.Compose([
        T.Resize((h, w)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.ToTensor(),
        T.Normalize(mean=cfg["data"]["imagenet_mean"], std=cfg["data"]["imagenet_std"])
    ])

    transform_val = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=cfg["data"]["imagenet_mean"], std=cfg["data"]["imagenet_std"])
    ])

    dataset = ImageFolder(a.data)
    
    # 80/20 split
    num_val = int(0.2 * len(dataset))
    num_train = len(dataset) - num_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [num_train, num_val])
    
    # Apply different transforms (hacky but works for subset)
    train_ds.dataset.transform = transform_train
    val_ds.dataset.transform = transform_val

    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=2)

    optimizer = optim.Adam([p for p in train_model.parameters() if p.requires_grad], lr=a.lr)
    criterion = nn.CrossEntropyLoss()

    num_classes = len(dataset.classes)
    
    logger.info("Starting training of terrain head only...")
    for epoch in range(a.epochs):
        train_model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            out = train_model(images)
            loss = criterion(out["terrain_logits"], targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{a.epochs} - Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    train_model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    correct = 0
    total = 0
    
    val_images_for_assert = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            if len(val_images_for_assert) < 5:
                val_images_for_assert.append(images)
            out = train_model(images)
            preds = out["terrain_logits"].argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            for t, p in zip(targets, preds):
                confusion[t.item(), p.item()] += 1

    logger.info(f"Val Accuracy: {correct/total:.4f}")
    logger.info(f"Confusion Matrix:\n{confusion.numpy()}")
    
    # Per-class accuracy
    per_class = confusion.diag() / confusion.sum(dim=1)
    for i, cls in enumerate(dataset.classes):
        logger.info(f"Class {cls}: {per_class[i]:.4f}")

    # Save
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": train_model.state_dict(), "cfg": cfg}, out_path)
    logger.info(f"Saved terrain model to {out_path}")

    # Assert that segmentation outputs are identical before and after
    logger.info("Asserting segmentation outputs are identical to original...")
    if len(val_images_for_assert) > 0:
        test_imgs = torch.cat(val_images_for_assert, dim=0)[:5]
        with torch.no_grad():
            orig_out = model(test_imgs)
            new_out = train_model(test_imgs)
            assert torch.allclose(orig_out["seg_logits"], new_out["seg_logits"], atol=1e-5), "Segmentation outputs changed!"
        logger.info("Assertion passed: Segmentation logits are identical.")

if __name__ == "__main__":
    train_terrain()
