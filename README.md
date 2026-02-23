Mask-aware CNN for TN/TP Prediction (PartialConv + Masked Pooling)
=================================================================

This repository contains PyTorch implementations of two mask-aware CNN models
for predicting Total Nitrogen (TN) and Total Phosphorus (TP) from multi-source
remote-sensing image patches. The models are designed to be robust to missing
pixels via PartialConv-like masked convolution and mask-aware average pooling.

Key features
------------
1) Multi-source patch input: 33×36 pixels with 41 channels
   - 1 DNBR band (NTL proxy)
   - 7 MODIS reflectance bands (B1–B7)
   - 33 ERA5 hydrometeorological variables

2) Validity mask (binary)
   - A pixel is valid if and only if all 41 channels are available (non-missing).
   - Missing values are represented by NaN in the input arrays.

3) Mask-aware building blocks
   - PartialConv-like convolution: convolution over valid pixels only + count-based renormalization.
   - Mask-aware average pooling: averages only valid entries and propagates the mask.

4) Separate architectures for TN and TP
   - TN model: 5 masked conv layers, GroupNorm + ReLU
   - TP model: 6 masked conv layers, BatchNorm2d + ReLU + stage-wise dropout (0.2/0.3/0.4)


Data requirements
-----------------
The scripts assume the following variables are already available in the runtime:
- X_final : numpy array of shape (N, H, W, C) = (N, 33, 36, 41)
           Missing values must be NaN.
- y_final : numpy array of shape (N, >=2)
           TN and TP columns are stored in y_final (see scripts for column indices).
- meta_final : list of dicts with at least:
           meta_final[i]["date"] as a string in "YYYYMMDD" format
           (optional) meta_final[i]["site_id"] (or index/id) for saving outputs

Example:
X_final[i] is a patch for one in situ record.
y_final[i] is the corresponding target (TN/TP).
meta_final[i]["date"] is the sampling date.

Pre-processing and splitting
----------------------------
1) Patch completeness filtering:
   - Samples are kept if valid-pixel ratio >= 0.5
   - Additional outlier filtering uses z-score of NaN ratio (threshold = 2.5)

2) Time-based split:
   - Samples with date < 20240101 are used for training/validation
   - Samples with date >= 20240101 are used as an independent test set
   - If the time split fails (empty train or test), a random 8:2 split is used

3) Standardization:
   - Per-channel z-score standardization using training-set statistics only

4) Targets:
   - Training is performed in log space: log(y)
   - Predictions are converted back using exp(pred_log) - 1e-6

How to run
----------
Option A (recommended): prepare a small runner that loads data then imports the script.
Example pseudo-steps:
1) Load X_final, y_final, meta_final into memory (numpy arrays / python list)
2) Run:
   python tp_train_min.py
   python tn_train_min.py

Option B: modify each script to load your data from disk at the beginning
(e.g., np.load / np.loadz / pickle) before calling main().

Outputs
-------
The scripts save:
1) Model checkpoints
   - tp_maskaware_checkpoint.pt
   - tn_maskaware_checkpoint.pt
   Each checkpoint includes:
   - state_dict (model weights)
   - band_mean, band_std (training-set normalization statistics)
   - input_channels
   - arch identifier

2) Prediction summaries (NPZ)
   - tp_predictions_train_val_test.npz
   - tn_predictions_train_val_test.npz
   These files store predictions and metadata (dates/site ids) for later analysis.
   NOTE: Loading requires allow_pickle=True for meta arrays.

Environment
-----------
Core dependencies:
- Python 3.9+ (recommended)
- PyTorch (tested with 2.x)
- NumPy
- SciPy
- scikit-learn

Optional performance settings in the original code include:
- mixed precision (autocast + GradScaler)
- channels_last memory format
- cudnn.benchmark and TF32 (if CUDA is available)

Notes / common pitfalls
-----------------------
1) Do NOT upload large data files or model checkpoints to GitHub by default.
   Add them to .gitignore (e.g., *.npz, *.pt, *.pth).

2) Ensure NaNs represent missing pixels consistently across all 41 channels.
   The validity mask assumes a pixel is valid only if ALL channels are non-NaN.

3) If using BatchNorm2d (TP model), batch size can affect training stability.
   The scripts use batch_size=128 on GPU and 32 on CPU.

Citation
--------
If you use this code in academic work, please cite the associated paper/project.

Contact
-------
Maintainer: (your name)
Email: (your email)
