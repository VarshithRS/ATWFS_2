"""IDD label hierarchy handling and binary drivable/non-drivable collapsing.

Source of truth (CONFIRMED from the official AutoNUE repo, not guessed):
``github.com/AutoNUE/public-code`` -> ``helpers/anue_labels.py`` and README.

Level-3 ids (the ids stored in ``*_gtFine_labellevel3Ids.png``):
    0 road | 1 parking AND drivable fallback | 2 sidewalk | 3 rail track AND
    non-drivable fallback | 4 person(+animal) | 5 rider | 6 motorcycle | 7 bicycle |
    8 autorickshaw | 9 car | 10 truck | 11 bus | 12 caravan/trailer/train/vehicle
    fallback | 13 curb | ... | 25 sky/fallback background | 255 unlabeled/ignore.
In the official table, road, parking and drivable fallback all have level1Id 0
("drivable"); everything else has level1Id >= 1. Hence drivable == level3 {0, 1}.

TODO(user, one-time): the ids above come from the official helper script. Confirm
they match YOUR downloaded files by running
``python -m data.labels --data-root <your idd folder>``; it scans mask files and
reports any unexpected values.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)

# Official evaluation class names for level-3 ids 0..25 (evaluate_mIoU.py).
IDD_LEVEL3_NAMES = [
    "road", "drivable fallback", "sidewalk", "non-drivable fallback", "person", "rider",
    "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "vehicle fallback",
    "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light",
    "pole", "obs-str-bar-fallback", "building", "bridge", "vegetation", "sky",
]
NUM_LEVEL3_CLASSES = len(IDD_LEVEL3_NAMES)  # 26


def collapse_to_binary(mask: np.ndarray, drivable_ids: Sequence[int]) -> np.ndarray:
    """Collapse a level-3 id mask to binary (1 = drivable, 0 = everything else).

    Unlabeled/ignore pixels (255) fall into 0 (conservative: never claim drivable
    where the annotators did not).

    Args:
        mask: ``(H, W)`` integer array of level-3 ids.
        drivable_ids: Level-3 ids considered drivable.

    Returns:
        ``(H, W)`` uint8 array with values in {0, 1}.
    """
    return np.isin(mask, np.asarray(list(drivable_ids))).astype(np.uint8)


def ids_to_mask(mask: np.ndarray, ids: Iterable[int]) -> np.ndarray:
    """Binary mask of pixels whose id is in ``ids`` (used for obstacle GT).

    Args:
        mask: ``(H, W)`` id mask.
        ids: Ids to select.

    Returns:
        ``(H, W)`` uint8 mask.
    """
    return np.isin(mask, np.asarray(list(ids))).astype(np.uint8)


def verify_label_files(data_root: str, split: str = "train", mask_suffix: str = "_gtFine_labellevel3Ids.png",
                       max_files: int = 100) -> Set[int]:
    """Scan mask files and report the unique ids found.

    Args:
        data_root: IDD root folder.
        split: Split to scan.
        mask_suffix: Mask file suffix.
        max_files: Max number of files to scan.

    Returns:
        Set of unique values found. Logs an error if any value is outside 0..25 or 255.
    """
    from PIL import Image

    files = sorted(Path(data_root, "gtFine", split).glob(f"*/*{mask_suffix}"))[:max_files]
    if not files:
        logger.error("No mask files matching '*%s' under %s/gtFine/%s. Did you run the AutoNUE "
                     "createLabels.py --id-type level3Id step?", mask_suffix, data_root, split)
        return set()
    found: Set[int] = set()
    for f in files:
        found |= set(np.unique(np.array(Image.open(f))).tolist())
    valid = set(range(NUM_LEVEL3_CLASSES)) | {255}
    bad = found - valid
    if bad:
        logger.error("UNEXPECTED mask values %s in %d files. Check data.drivable_level3_ids!", sorted(bad), len(files))
    else:
        logger.info("Label check OK over %d files. Unique ids: %s", len(files), sorted(found))
    return found


if __name__ == "__main__":
    from utils.config import load_config, resolve_path
    from utils.logging_utils import setup_logging

    setup_logging()
    ap = argparse.ArgumentParser(description="Verify IDD level-3 mask ids.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    c = load_config()
    verify_label_files(str(resolve_path(a.data_root or c["paths"]["data_root"])), a.split, c["data"]["mask_suffix"])
