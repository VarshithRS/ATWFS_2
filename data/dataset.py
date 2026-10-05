"""Datasets: real IDD (official splits) and synthetic, sharing one processing pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.augment import StandardAugment, SyntheticConditionAugment
from data.labels import collapse_to_binary, ids_to_mask
from data.synthetic import SyntheticSource
from utils.config import resolve_path

logger = logging.getLogger(__name__)


class SampleSource(Protocol):
    """Anything that yields ``(RGB uint8 image, raw level-3 mask, terrain class or -1)``."""

    def __len__(self) -> int: ...
    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray, int]: ...


def get_split_files(data_root: str, split: str, image_suffix: str, mask_suffix: str,
                    require_masks: bool = True) -> List[Tuple[Path, Optional[Path]]]:
    """Official IDD split listing, by the official directory layout.

    Layout (AutoNUE README): ``leftImg8bit/{split}/{drive_no}/{id}_leftImg8bit.png`` and
    ``gtFine/{split}/{drive_no}/{id}_gtFine_labellevel3Ids.png``.

    Args:
        data_root: IDD root.
        split: ``train`` | ``val`` | ``test``.
        image_suffix: Image suffix.
        mask_suffix: Mask suffix.
        require_masks: If True, raise when a mask is missing (the public ``test`` split has no GT).

    Returns:
        Sorted list of ``(image_path, mask_path_or_None)``.
    """
    root = Path(data_root)
    imgs = sorted((root / "leftImg8bit" / split).glob(f"*/*{image_suffix}"))
    if not imgs:
        raise FileNotFoundError(
            f"No images under {root / 'leftImg8bit' / split}. Set paths.data_root in config.yaml "
            f"to your IDD Segmentation folder.")
    out: List[Tuple[Path, Optional[Path]]] = []
    for ip in imgs:
        mp = root / "gtFine" / split / ip.parent.name / (ip.name[: -len(image_suffix)] + mask_suffix)
        if not mp.exists():
            if require_masks:
                raise FileNotFoundError(f"Missing mask {mp}. Generate level-3 masks with AutoNUE "
                                        f"preperation/createLabels.py --id-type level3Id.")
            mp = None
        out.append((ip, mp))
    return out


class IDDSource:
    """Real IDD sample source for one official split."""

    def __init__(self, cfg: Dict[str, Any], split: str, require_masks: bool = True) -> None:
        """Index the split and sanity-check its size against the official numbers.

        Args:
            cfg: Full config.
            split: ``train`` | ``val`` | ``test``.
            require_masks: Require GT masks.
        """
        d = cfg["data"]
        self.files = get_split_files(str(resolve_path(cfg["paths"]["data_root"])), split,
                                     d["image_suffix"], d["mask_suffix"], require_masks)
        exp = d["expected_split_sizes"].get(split)
        if exp is not None and len(self.files) != exp:
            logger.warning("Split '%s' has %d samples but the official size is %d.", split, len(self.files), exp)
        else:
            logger.info("Split '%s': %d samples (matches official size).", split, len(self.files))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.files)

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Read image + raw mask.

        Args:
            i: Index.

        Returns:
            ``(RGB image, raw level-3 mask (255 if unavailable), terrain=-1)``.
            Terrain is -1 (ignored): IDD has no terrain labels (TODO supply your own).
        """
        ip, mp = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(ip), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if mp is not None:
            mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = mask.astype(np.uint8)
        else:
            mask = np.full(img.shape[:2], 255, np.uint8)
        return img, mask, -1


class DrivableDataset(Dataset):
    """Binary drivable-region dataset with optional augmentation.

    Item dict: ``image (3,H,W) float``, ``mask (H,W) long {0,1}``, ``obstacle (H,W) long {0,1}``, ``valid (H,W) long {0,1}`` (0 = ignore label),
    ``terrain () long`` (-1 = unlabeled), ``index () long``.
    """

    def __init__(self, source: SampleSource, cfg: Dict[str, Any], train: bool = False, seed: int = 0) -> None:
        """Create the dataset.

        Args:
            source: Sample source.
            cfg: Full config.
            train: Whether to apply augmentation.
            seed: Seed for the augmenters.
        """
        self.source, self.train = source, train
        d = cfg["data"]
        self.size = tuple(d["image_size"])
        self.drivable_ids, self.obstacle_ids = d["drivable_level3_ids"], d["obstacle_level3_ids"]
        self.ignore_label = int(d["ignore_label"])
        self.mean = np.asarray(d["imagenet_mean"], np.float32)
        self.std = np.asarray(d["imagenet_std"], np.float32)
        self.std_aug = StandardAugment(cfg["augmentation"]["standard"], seed)
        self.syn_aug = SyntheticConditionAugment(cfg["augmentation"]["synthetic"], seed + 1)

    def __len__(self) -> int:
        """Return the dataset length."""
        return len(self.source)

    def reseed(self, seed: int) -> None:
        """Re-seed augmenters (called per worker).

        Args:
            seed: New seed.
        """
        self.std_aug.reseed(seed)
        self.syn_aug.reseed(seed + 1)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """Load, collapse labels, augment and normalise sample ``i``.

        Args:
            i: Index.

        Returns:
            Sample dict (see class docstring).
        """
        img, raw, terrain = self.source.load(i)
        h, w = self.size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = collapse_to_binary(raw, self.drivable_ids)
        obst = ids_to_mask(raw, self.obstacle_ids)
        valid = (raw != self.ignore_label).astype(np.uint8)  # 0 where annotators left the pixel unlabeled
        if self.train:
            img, (mask, obst, valid) = self.std_aug(img, [mask, obst, valid])
            img = self.syn_aug(img)  # no-op unless augmentation.synthetic.enabled
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return {
            "image": torch.from_numpy(x.transpose(2, 0, 1).copy()),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "obstacle": torch.from_numpy(obst.astype(np.int64)),
            "valid": torch.from_numpy(valid.astype(np.int64)),
            "terrain": torch.tensor(int(terrain), dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }


def build_dataset(cfg: Dict[str, Any], split: str, synthetic: bool = False, train: Optional[bool] = None,
                  num_samples: Optional[int] = None) -> DrivableDataset:
    """Build a dataset for a split.

    Args:
        cfg: Full config.
        split: ``train`` | ``val`` | ``test``.
        synthetic: Use the procedural generator instead of real IDD.
        train: Override augmentation flag (default: True only for ``train``).
        num_samples: Synthetic length (defaults to ``smoke`` config numbers).

    Returns:
        :class:`DrivableDataset`.
    """
    train = (split == "train") if train is None else train
    seed = cfg["seed"]
    if synthetic:
        n = num_samples or (cfg["smoke"]["num_train"] if split == "train" else cfg["smoke"]["num_val"])
        src: SampleSource = SyntheticSource(n, tuple(cfg["data"]["image_size"]), seed={"train": 1, "val": 2, "test": 3}[split] + seed)
    else:
        src = IDDSource(cfg, split, require_masks=(split != "test"))
    return DrivableDataset(src, cfg, train=train, seed=seed)


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker init: give each worker's augmenters a distinct seed."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(int(torch.initial_seed() % 2**31) + worker_id)


def build_dataloader(ds: Dataset, cfg: Dict[str, Any], shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:
    """Create a reproducible DataLoader.

    Args:
        ds: Dataset.
        cfg: Full config.
        shuffle: Shuffle flag.
        batch_size: Optional override.

    Returns:
        DataLoader (``pin_memory`` only when CUDA is available).
    """
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    nw = int(cfg["data"]["num_workers"])
    return DataLoader(ds, batch_size=batch_size or cfg["train"]["batch_size"], shuffle=shuffle,
                      num_workers=nw, worker_init_fn=_seed_worker, generator=g,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=nw > 0)
