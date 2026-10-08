# Vision-Based Road Drivable-Region Estimation for Lane-Less Roads

Multi-task PyTorch pipeline (IDD 20k Part I): MobileNetV3-Large U-Net with binary segmentation, a single-pass
**evidential (Dirichlet)** uncertainty head, a 4-class terrain head and deep supervision; GroupNorm throughout;
Kendall-style learnable loss weights; distillation to MobileNetV3-Small; INT8 / structured-pruning comparison;
classical fallback + geometric/temporal checks + trust fusion; threaded live inference with YOLOv8n obstacle subtraction.

## 1. First things to do
1. `pip install -r requirements.txt`
2. If using the Kaggle "archive" set (`image_archive/` and `mask_archive/`), it must first be converted into the standard IDD format. Run:
   `python convert_idd.py --images image_archive --masks mask_archive --out ./data/idd_converted`
3. **Update `paths.data_root` in `config.yaml`** to your converted folder (default `./data/idd_converted`).
4. Verify the install: `python run_all_smoke_tests.py` (no data/GPU needed).

## 2. Folder structure
```
config.yaml                every path / hyperparameter
run_all_smoke_tests.py     runs all stage smoke tests in order
data/        labels.py (hierarchy + collapse) | dataset.py (IDD + synthetic, official splits) |
             augment.py (standard + independent rare-condition module) | synthetic.py (fake scenes/videos)
models/      unet_mobilenet.py (multi-task net) | compression.py (INT8 PTQ, structured pruning, size/FPS)
training/    losses.py | metrics.py | train.py | distill.py | fusion_train.py (STUB)
inference/   classical.py | consistency.py | fusion.py (ATWFS) | detector.py | decision.py | live.py
evaluation/  evaluate.py (AutoNUE-style mIoU) | benchmark.py | failure_taxonomy.py | gradcam.py
utils/       config, seed/device, logging
tests/       one smoke test per stage
notebooks/   colab_train.ipynb
```

## 3. How to run
| Task | Command | Where |
|---|---|---|
| All smoke tests | `python run_all_smoke_tests.py` | Windows CPU OK |
| 2-epoch pipeline sanity run | `python -m training.train --cpu-fallback` | Windows CPU OK (<1 min) |
| Train teacher on IDD | `python -m training.train` | **GPU/Colab** |
| Distil student | `python -m training.distill` | **GPU/Colab** (CPU only for tiny tests) |
| Compression benchmark | `python -m evaluation.benchmark` (add `--synthetic` for a dry run) | CPU OK (INT8 FPS is CPU by design); real val set is faster on GPU for accuracy |
| Evaluate (IDD protocol) | `python -m evaluation.evaluate --checkpoint outputs/checkpoints/teacher_best.pt` | CPU OK, GPU faster |
| Failure taxonomy | `python -m evaluation.failure_taxonomy --checkpoint ... --save-failures` | CPU OK |
| Grad-CAM (offline) | `python -m evaluation.gradcam --checkpoint ... --images outputs/reports/failures` | CPU OK |
| Live / video inference | `python -m inference.live --source video.mp4 --checkpoint ...` (`--source 0` webcam, rtsp/http URL IP cam) | CPU OK |
| FusionNet training | `python -m training.fusion_train` | STUB: needs data that does not exist yet |

Device is chosen automatically with `torch.cuda.is_available()`. Seeds (torch/numpy/random) are set in every entry point.
Add `--synthetic` to train/distill/evaluate to use procedurally generated data. Colab: `notebooks/colab_train.ipynb`
(or `training.train.colab_setup()`); IDD needs registration so place it in Drive and set `IDD_DRIVE_PATH`.

## 4. IDD details (confirmed vs. to-verify)
* **Dataset Note**: The reported mIoU in our results is on a custom `val` split of the Kaggle data, **not** the official IDD val split.
* masks: `gtFine/{split}/{drive}/{id}_gtFine_labellevel3Ids.png`; images `leftImg8bit/{split}/{drive}/{id}_leftImg8bit.png`/{split}/{drive}/{id}_gtFine_labellevel3Ids.png`; images `leftImg8bit/{split}/{drive}/{id}_leftImg8bit.png`
* level-3 ids: road = 0; parking **and** drivable fallback = 1 (both level1 "drivable"); sidewalk = 2;
  non-drivable fallback = 3; 255 = unlabeled. => drivable = `{0, 1}` (`data.drivable_level3_ids`). Unlabeled -> non-drivable.
* mIoU: 27x27 confusion matrix with an ignore bucket, FP includes ignored-GT pixels (`evaluation/evaluate.py`).

**TODO (you, once):** these were confirmed from the official *code*, not from your downloaded files. Run
`python -m data.labels --data-root <idd>`; it scans masks and errors on any unexpected id. If your files differ, edit
`data.drivable_level3_ids` / `data.mask_suffix` in `config.yaml`.

Known data caveats:
* The **public IDD test split has no ground-truth masks**; `data.eval_split` (default `val`) is used for evaluation/benchmarks.
  Split sizes are checked against 6,993 / 981 / 2,029 and a warning is logged on mismatch.
* **IDD has no terrain labels**: on real data the terrain label is -1 (ignored by the loss, so the terrain head is untrained
  there). Supply your own terrain labels (TODO) to train/evaluate it. Synthetic data carries terrain labels.

## 5. Placeholders / stubs (all commented in code)
* `inference/fusion.py` - FusionNet is initialised to a **hand-coded heuristic** (reproduced exactly); `training/fusion_train.py` is a stub awaiting real data.
* `inference/detector.py` - pretrained COCO YOLOv8n; `CattleDetectorPlaceholder` / `CompositeDetector` mark where a fine-tuned cattle detector plugs in.
* `inference/live.py` - the `[angle, speed, trust_score, obstacle_flag]` console line is a **stub for a future Arduino link**. No serial/hardware code exists.
* `speed_mps_input` (temporal check) is a constant config value until real odometry is available.

## 6. Notes
* Speed formula, steering formula and obstacle logic are documented at the top of `inference/decision.py`.
* ImageNet weights download on first use (torchvision). If the download fails the encoder falls back to random init **with a warning**.
* INT8: FX-graph static PTQ (`torch.ao`), calibrated on val batches; torch marks `torch.ao.quantization` as deprecated (migration path: torchao). Falls back to dynamic INT8 / flagged fp32 copy if unavailable.
* Structured pruning removes decoder-internal channels only (real size reduction; encoder untouched). INT8/pruning are applied to the teacher by default (`--base student` to change).
* Untrained/random-weight models give meaningless metrics - the logs say so when a checkpoint is missing.
