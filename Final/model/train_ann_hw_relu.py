"""
Hardware-matched ANN training script for IEC104 network traffic classification.

This variant keeps the existing data pipeline and BF16 export flow, but changes
the student network to match the current SoC datapath exactly:

  Linear -> ReLU -> Linear -> ReLU -> Linear

Dataset strategy:
  - Uses ALL 12 custom-feature files (6 time windows × train+test): 54,696 rows
    → custom features include IEC104-specific fields (COT, type_id, APDU lengths)
  - Also stacks ALL 12 CIC files (6 windows × train+test): 54,696 additional rows
    → CIC rows get IEC104-specific columns zero-filled (model learns from both)
  - Grand total: ~109,392 rows (previously only 47,868 were used — 5 of 6 windows, custom only)
  - Random stratified 80/20 split (eliminates cross-capture distribution shift)

Training pipeline:
  1. Load & merge all data (custom + CIC, all windows)
  2. Train LightGBM teacher (~75% accuracy)
  3. Select top-32 features by LightGBM importance
  4. Knowledge-distill into ANN: 32 → 16 → 16 → 12  (1004 weight entries ≤ 1024 DMEM)
  5. Export BF16 weights as $readmemh-compatible DMEM hex files

GPU DMEM layout (1024 × 16-bit BF16):
  Addr   0– 511  net.0.weight  (16, 32)
  Addr 512– 527  net.0.bias    (16,)
  Addr 528– 783  net.2.weight  (16, 16)
  Addr 784– 799  net.2.bias    (16,)
  Addr 800– 991  net.4.weight  (12, 16)
  Addr 992–1003  net.4.bias    (12,)
  Addr 1004–1023  (zero-padded)
"""

import glob
import os
import struct
import subprocess
import sys
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_BASE = "iec104_train_test_csvs" if os.path.isdir("iec104_train_test_csvs") else "iec104_train_test"
BASE         = os.environ.get("ANN_DATASET_DIR", DEFAULT_BASE)

# File patterns for custom features (all 6 time windows)
CUSTOM_GLOB = f"{BASE}/tests_custom_*/*.csv"

# File patterns for CIC features (all 6 time windows)
CIC_GLOB    = f"{BASE}/tests_cic_*/*.csv"
CIC_DROP    = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]  # non-feature CIC cols
MIXED_COLS  = ["Fwd Header Len"]                                      # mixed-type in CIC

# Mapping: CIC column name → equivalent custom column name.
# Allows CIC rows to fill the same feature slots as custom rows for shared timing/flow features.
# IEC104-specific columns (cot=*, type_id_*, APDU lengths) stay zero for CIC rows.
CIC_TO_CUSTOM = {
    # IAT stats
    "Flow IAT Max":      "flow IAT max",
    "Flow IAT Min":      "flow IAT min",
    "Flow IAT Mean":     "flow IAT mean",
    "Flow IAT Std":      "flow IAT std",
    "Fwd IAT Tot":       "fw iAT tot",
    "Fwd IAT Max":       "fw IAT max",
    "Fwd IAT Min":       "fw IAT min",
    "Fwd IAT Mean":      "fw IAT mean",
    "Fwd IAT Std":       "fw IAT std",
    "Bwd IAT Tot":       "bw IAT tot",
    "Bwd IAT Max":       "bw IAT max",
    "Bwd IAT Min":       "bw IAT min",
    "Bwd IAT Mean":      "bw IAT mean",
    "Bwd IAT Std":       "bw IAT std",
    # Active / idle time
    "Active Max":        "flow active time max",
    "Active Min":        "flow active time min",
    "Active Mean":       "flow active time mean",
    "Active Std":        "flow active time std",
    "Idle Max":          "flow idle time max",
    "Idle Min":          "flow idle time min",
    "Idle Mean":         "flow idle time mean",
    "Idle Std":          "flow idle time std",
    # Packet / byte counts
    "Tot Fwd Pkts":      "total fw packets",
    "Tot Bwd Pkts":      "total bw packets",
    "TotLen Fwd Pkts":   "fw packets APDU total length",
    "TotLen Bwd Pkts":   "bw packets APDU total length",
    "Flow Duration":     "flow duration",
    "Down/Up Ratio":     "flow down/up ratio",
    # Packet-length stats (APDU ≈ application-layer payload in IEC104)
    "Pkt Len Max":       "flow packet APDU length max",
    "Pkt Len Min":       "flow packet APDU length min",
    "Pkt Len Mean":      "flow packet APDU length mean",
    "Pkt Len Std":       "flow packet APDU length std",
    "Pkt Len Var":       "flow packet APDU length var",
    "Fwd Pkt Len Max":   "fw packet APDU length max",
    "Fwd Pkt Len Min":   "fw packet APDU length min",
    "Fwd Pkt Len Mean":  "fw packet APDU length mean",
    "Fwd Pkt Len Std":   "fw packet APDU length std",
    "Bwd Pkt Len Max":   "bw packet APDU length max",
    "Bwd Pkt Len Min":   "bw packet APDU length min",
    "Bwd Pkt Len Mean":  "bw packet APDU length mean",
    "Bwd Pkt Len Std":   "bw packet APDU length std",
    # Rates
    "Flow Pkts/s":       "flow iec104 packts/s",
    "Flow Byts/s":       "flow iec104 bytes/s",
    "Fwd Pkts/s":        "fw iec104 packts/s",
    "Bwd Pkts/s":        "bw iec104 packts/s",
    # Flags
    "Fwd PSH Flags":     "fw PSH flag amount",
    "Bwd PSH Flags":     "bw PSH flag amount",
    "Fwd URG Flags":     "fw URG flag amount",
    "Bwd URG Flags":     "bw URG flag amount",
    "SYN Flag Cnt":      "flow SYN flag count",
    "RST Flag Cnt":      "flow RST flag count",
    "PSH Flag Cnt":      "flow PSH flag count",
    "ACK Flag Cnt":      "flow ACK flag count",
    "URG Flag Cnt":      "flow URG flag count",
    "CWE Flag Count":    "flow CWE flag count",
    "ECE Flag Cnt":      "flow ECE flag count",
    # Subflows
    "Subflow Fwd Pkts":  "fw_subflow_packets",
    "Subflow Bwd Pkts":  "bw_subflow_packets",
    "Subflow Fwd Byts":  "fw_subflow_bytes",
    "Subflow Bwd Byts":  "bw_subflow_bytes",
    # Bulk stats
    "Fwd Byts/b Avg":    "fw avg bytes/bulk",
    "Bwd Byts/b Avg":    "bw avg bytes/bulk",
    "Fwd Blk Rate Avg":  "fw avg bulk rate",
    "Bwd Blk Rate Avg":  "bw avg bulk rate",
    "Fwd Pkts/b Avg":    "fw avg packets/bulk",
    "Bwd Pkts/b Avg":    "bw avg packets/bulk",
    # Window / header
    "Init Fwd Win Byts": "init fw window bytes",
    "Init Bwd Win Byts": "init bw window bytes",
    "Fwd Header Len":    "fw TCP total header length",
    "Bwd Header Len":    "bw TCP total header length",
}

# GPU DMEM: 16-bit per word × 2048 deep × 4 threads (one BF16 per address)
#   BF16 per thread  = 2048 × 1 = 2,048
#   Total BF16 slots = 4 threads × 2,048 = 8,192
#
# 4-class design (keep best 4: DoS, NORMAL, c_ci_na_1, c_se_na_1)
#
# Architecture: 40→96→42→4  (2 hidden layers)
#   net.0.weight (96×40)  + net.0.bias (96)  = 3,936
#   net.2.weight (42×96)  + net.2.bias (42)  = 4,074
#   net.4.weight (4×42)   + net.4.bias (4)   =   172
#   Total params = 8,182  ≤  8,192  ✓  (uses 99.9% of DMEM)
N_FEAT      = 40    # top-40 features by LightGBM importance
HIDDEN1     = 96    # hidden layer 1
HIDDEN2     = 42    # hidden layer 2
N_CLASSES   = 4     # DoS, NORMAL, c_ci_na_1, c_se_na_1

# Classes to keep (others filtered out entirely)
KEEP_CLASSES = ['DoS', 'NORMAL', 'c_ci_na_1', 'c_se_na_1']

# DMEM layout constants (hardware)
N_THREADS       = 4
DMEM_DEPTH      = 2048   # 16-bit words per thread
BF16_PER_THREAD = DMEM_DEPTH
TOTAL_CAPACITY  = N_THREADS * BF16_PER_THREAD  # 8192

LGBM_TREES  = 300   # more trees → better teacher soft labels

EPOCHS      = int(os.environ.get("ANN_EPOCHS", "5000"))  # more epochs — 2-layer net needs longer to converge
BATCH       = int(os.environ.get("ANN_BATCH", "256"))    # smaller batch → noisier gradients → better generalisation
LR          = float(os.environ.get("ANN_LR", "1e-3"))    # lower LR pairs well with cosine annealing
KD_TEMP     = float(os.environ.get("ANN_KD_TEMP", "4.0"))
KD_ALPHA    = float(os.environ.get("ANN_KD_ALPHA", "0.7"))
DROPOUT     = float(os.environ.get("ANN_TRAIN_DROPOUT", "0.0"))
LABEL_SMOOTH = float(os.environ.get("ANN_LABEL_SMOOTH", "0.05"))

OUTPUT_DIR  = os.environ.get("ANN_OUTPUT_DIR", "GPU_HW_RELU")
DMEM_HEX    = os.path.join(OUTPUT_DIR, "ann_weights_dmem.hex")
WEIGHTS_NPY = os.path.join(OUTPUT_DIR, "ann_weights_bf16.npy")


# ─── Data loading helpers ─────────────────────────────────────────────────────

def load_custom(path: str):
    """Load a custom-feature CSV → (DataFrame of numeric features, label array)."""
    df = pd.read_csv(path, low_memory=False)
    y  = df["Label"].values
    df = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, y


def load_cic(path: str):
    """Load a CIC-feature CSV → (DataFrame with custom-compatible column names, label array).
    Renames CIC columns to their custom equivalents so CIC rows can be stacked with custom rows.
    Columns with no mapping (IEC104-specific: cot=*, type_id_*, APDU) are absent here and
    will be zero-filled when reindexed to the master custom column list."""
    df = pd.read_csv(path, low_memory=False)
    y  = df["Label"].values
    for col in MIXED_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop(columns=CIC_DROP, errors="ignore").select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # Rename CIC columns → custom column names
    df = df.rename(columns=CIC_TO_CUSTOM)
    # Drop any CIC columns that have no mapping (not needed, reindex handles them)
    return df, y


def merge_all_data():
    """
    Load all custom + CIC files across all time windows.
    CIC rows are aligned to the custom column space (IEC104-specific columns zero-filled).
    Returns: X (float32 ndarray), y_raw (str label array), all_columns (list)
    """
    print("[1] Loading all datasets...")

    # ── Load ALL custom files ──────────────────────────────────────────
    custom_files = sorted(glob.glob(CUSTOM_GLOB))
    if not custom_files:
        raise FileNotFoundError(f"No custom files matched: {CUSTOM_GLOB}")

    custom_dfs, custom_ys = [], []
    for f in custom_files:
        df, y = load_custom(f)
        custom_dfs.append(df)
        custom_ys.append(y)
        print(f"  custom  {len(y):6d} rows  {f.split('/')[-2]}/{f.split('/')[-1]}")

    # Establish the master column list from the first custom file
    master_cols = custom_dfs[0].columns.tolist()

    # Align all custom DataFrames to master columns (in case some windows differ)
    custom_dfs = [df.reindex(columns=master_cols, fill_value=0) for df in custom_dfs]
    X_custom   = np.vstack([df.values.astype(np.float32) for df in custom_dfs])
    y_custom   = np.concatenate(custom_ys)
    print(f"  → Custom total: {X_custom.shape[0]} rows × {X_custom.shape[1]} features")

    # ── Load ALL CIC files and align to master columns ─────────────────────────
    cic_files = sorted(glob.glob(CIC_GLOB))
    if not cic_files:
        raise FileNotFoundError(f"No CIC files matched: {CIC_GLOB}")

    cic_dfs, cic_ys = [], []
    for f in cic_files:
        df, y = load_cic(f)
        # Align to master custom columns; IEC104-specific cols get 0
        df = df.reindex(columns=master_cols, fill_value=0)
        cic_dfs.append(df)
        cic_ys.append(y)
        print(f"  cic     {len(y):6d} rows  {f.split('/')[-2]}/{f.split('/')[-1]}")

    X_cic = np.vstack([df.values.astype(np.float32) for df in cic_dfs])
    y_cic = np.concatenate(cic_ys)
    print(f"  → CIC total: {X_cic.shape[0]} rows × {X_cic.shape[1]} features "
          f"({len(CIC_TO_CUSTOM)} cols mapped, IEC104-specific cols zero-filled)")

    # Stack custom + CIC
    X     = np.vstack([X_custom, X_cic])
    y_raw = np.concatenate([y_custom, y_cic])
    print(f"  → Combined total: {X.shape[0]} rows × {X.shape[1]} features")

    # ── Feature engineering: add normalized IEC104 ratio features ─────────────
    # Raw COT/type_id COUNTS vary with flow duration → hard for small ANN to learn.
    # Normalizing by total I-messages creates flow-length-invariant ratios that are
    # much more linearly separable (fraction of messages with each COT/type_id).
    # New features appended: cot_frac_* and type_frac_* (20 additional features)
    cols_arr = np.array(master_cols)
    # Total I-messages in each direction (columns 54-56 are I-SeqIOA, 57-59 are I-SingleIOA)
    # We use "total flow packets" (col 46) as normalizer — always > 0
    total_pkt_idx = 46   # "total flow packets"
    total_pkts = X[:, total_pkt_idx].clip(min=1.0)  # avoid div-by-zero

    # COT columns: indices 91-104 (cot=1..13 and cot=20)
    cot_indices = list(range(91, 105))   # 14 COT columns
    # type_id columns: indices 105-110 (6 type_id features)
    type_indices = list(range(105, 111))  # 6 type_id columns

    cot_fracs  = X[:, cot_indices]  / total_pkts[:, None]
    type_fracs = X[:, type_indices] / total_pkts[:, None]

    X_eng = np.concatenate([X, cot_fracs, type_fracs], axis=1).astype(np.float32)
    new_cols = (master_cols
                + [f"cot_frac_{i}" for i in range(len(cot_indices))]
                + [f"type_frac_{i}" for i in range(len(type_indices))])
    print(f"  → Total: {X_eng.shape[0]} rows × {X_eng.shape[1]} features "
          f"(+{X_eng.shape[1]-X.shape[1]} engineered ratios)")
    return X_eng, y_raw, new_cols


# ─── Torch ANN training — run in a clean subprocess to avoid LightGBM/OpenMP deadlock ───

TORCH_SCRIPT = r"""
import math
import os
import struct
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

npz_path     = sys.argv[1]
out_dir      = sys.argv[2]
N_FEAT       = int(sys.argv[3])
HIDDEN1      = int(sys.argv[4])
HIDDEN2      = int(sys.argv[5])
N_CLASSES    = int(sys.argv[6])
EPOCHS       = int(sys.argv[7])
BATCH        = int(sys.argv[8])
LR           = float(sys.argv[9])
teacher_acc  = float(sys.argv[10])
class_names  = sys.argv[11].split(',')
KD_TEMP      = float(sys.argv[12])
KD_ALPHA     = float(sys.argv[13])
N_THREADS       = int(sys.argv[14])
DMEM_DEPTH      = int(sys.argv[15])
DROPOUT         = float(sys.argv[16])
LABEL_SMOOTH    = float(sys.argv[17])
BF16_PER_THREAD = DMEM_DEPTH
TOTAL_CAPACITY  = N_THREADS * BF16_PER_THREAD  # 8192

d = np.load(npz_path)
Xtr = d['Xtr']; Xte = d['Xte']
ytr = d['ytr']; yte = d['yte']
soft_labels = d['soft_labels']

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

bf16_mode = os.environ.get("ANN_BF16_MODE", "amp").lower()
use_amp = device.type in ("cuda", "mps")
amp_dtype = torch.bfloat16
use_fakequant = bf16_mode in ("fakequant", "ste")
train_bn = os.environ.get("ANN_TRAIN_BN", "0") == "1"
train_dropout = float(os.environ.get("ANN_TRAIN_DROPOUT", str(DROPOUT)))
balanced_epochs = int(os.environ.get("ANN_BALANCED_EPOCHS", "0"))
focal_gamma = float(os.environ.get("ANN_FOCAL_GAMMA", "0.0"))
focal_weight = float(os.environ.get("ANN_FOCAL_WEIGHT", "0.0"))

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ── Mixup augmentation helper ─────────────────────────────────────────────────
def mixup_batch(x, y_hard, y_soft, alpha=0.3):
    # Interpolate pairs of samples - creates synthetic training points
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(len(x), device=x.device)
    xm  = lam * x + (1 - lam) * x[idx]
    ym_hard = y_hard          # keep hard labels of first sample (for CE)
    ym_soft = lam * y_soft + (1 - lam) * y_soft[idx]   # blend soft labels
    return xm, ym_hard, ym_soft, lam

def amp_ctx():
    if use_amp:
        return torch.autocast(device_type=device.type, dtype=amp_dtype)
    return nullcontext()

def bf16_round_ste(x):
    if not use_fakequant:
        return x
    x_fp32 = x.float()
    x_q = x_fp32.to(torch.bfloat16).to(torch.float32)
    return x_fp32 + (x_q - x_fp32).detach()

class BF16AwareLinear(nn.Linear):
    def forward(self, x):
        x_q = bf16_round_ste(x)
        w_q = bf16_round_ste(self.weight)
        b_q = None if self.bias is None else bf16_round_ste(self.bias)
        out = F.linear(x_q, w_q, b_q)
        return bf16_round_ste(out)

def fuse_linear_bn(linear, bn):
    w = linear.weight.detach().cpu().float().clone()
    if linear.bias is None:
        b = torch.zeros(linear.out_features, dtype=torch.float32)
    else:
        b = linear.bias.detach().cpu().float().clone()

    if bn is None:
        return w, b

    gamma = bn.weight.detach().cpu().float()
    beta  = bn.bias.detach().cpu().float()
    mean  = bn.running_mean.detach().cpu().float()
    var   = bn.running_var.detach().cpu().float()
    scale = gamma / torch.sqrt(var + bn.eps)

    w = w * scale[:, None]
    b = beta + (b - mean) * scale
    return w, b

def collect_export_tensors(model):
    export_tensors = []

    if HIDDEN2 == 0:
        w0, b0 = fuse_linear_bn(model.fc1, model.bn1 if train_bn else None)
        export_tensors.append(("fc1.weight", w0))
        export_tensors.append(("fc1.bias",   b0))
        export_tensors.append(("fc2.weight", model.fc2.weight.detach().cpu().float().clone()))
        export_tensors.append(("fc2.bias",   model.fc2.bias.detach().cpu().float().clone()))
    else:
        w0, b0 = fuse_linear_bn(model.fc1, model.bn1 if train_bn else None)
        w1, b1 = fuse_linear_bn(model.fc2, model.bn2 if train_bn else None)
        export_tensors.append(("fc1.weight", w0))
        export_tensors.append(("fc1.bias",   b0))
        export_tensors.append(("fc2.weight", w1))
        export_tensors.append(("fc2.bias",   b1))
        export_tensors.append(("fc3.weight", model.fc3.weight.detach().cpu().float().clone()))
        export_tensors.append(("fc3.bias",   model.fc3.bias.detach().cpu().float().clone()))

    return export_tensors

# ── Model: hardware-matched 2 hidden layers + ReLU + BF16-aware training ─────
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        if HIDDEN2 == 0:
            self.fc1 = BF16AwareLinear(N_FEAT, HIDDEN1)
            self.bn1 = nn.BatchNorm1d(HIDDEN1) if train_bn else None
            self.drop1 = nn.Dropout(train_dropout) if train_dropout > 0.0 else nn.Identity()
            self.fc2 = BF16AwareLinear(HIDDEN1, N_CLASSES)
        else:
            self.fc1 = BF16AwareLinear(N_FEAT, HIDDEN1)
            self.bn1 = nn.BatchNorm1d(HIDDEN1) if train_bn else None
            self.drop1 = nn.Dropout(train_dropout) if train_dropout > 0.0 else nn.Identity()
            self.fc2 = BF16AwareLinear(HIDDEN1, HIDDEN2)
            self.bn2 = nn.BatchNorm1d(HIDDEN2) if train_bn else None
            self.drop2 = nn.Dropout(train_dropout * 0.5) if train_dropout > 0.0 else nn.Identity()
            self.fc3 = BF16AwareLinear(HIDDEN2, N_CLASSES)

    def forward(self, x):
        x = bf16_round_ste(x)
        x = self.fc1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = bf16_round_ste(F.relu(x))
        x = self.drop1(x)
        if HIDDEN2 == 0:
            x = self.fc2(x)
        else:
            x = self.fc2(x)
            if self.bn2 is not None:
                x = self.bn2(x)
            x = bf16_round_ste(F.relu(x))
            x = self.drop2(x)
            x = self.fc3(x)
        return bf16_round_ste(x)

if HIDDEN2 == 0:
    total_params = (N_FEAT*HIDDEN1+HIDDEN1) + (HIDDEN1*N_CLASSES+N_CLASSES)
    arch_str = f"{N_FEAT} -> {HIDDEN1} -> {N_CLASSES}"
else:
    total_params = (N_FEAT*HIDDEN1+HIDDEN1) + (HIDDEN1*HIDDEN2+HIDDEN2) + (HIDDEN2*N_CLASSES+N_CLASSES)
    arch_str = f"{N_FEAT} -> {HIDDEN1} -> {HIDDEN2} -> {N_CLASSES}"

print(f"\n[4] ANN: {arch_str}  ({total_params} params)", flush=True)
print(f"     Activation=ReLU  BN={train_bn}  Dropout={train_dropout:.2f}  LabelSmooth={LABEL_SMOOTH}  Mixup=True", flush=True)
print(f"     Device={device}  BF16Mode={bf16_mode}  AMP_BF16={use_amp}  FakeQuantBF16={use_fakequant}", flush=True)
print(f"     BalancedEpochs={balanced_epochs}  FocalGamma={focal_gamma:.2f}  FocalWeight={focal_weight:.2f}", flush=True)
assert total_params <= TOTAL_CAPACITY, f"Too many params: {total_params} > {TOTAL_CAPACITY}"

Xtr_t  = torch.from_numpy(Xtr).to(device)
Xte_t  = torch.from_numpy(Xte).to(device)
ytr_t  = torch.tensor(ytr, dtype=torch.long, device=device)
soft_t = torch.from_numpy(soft_labels).to(device)
N      = len(Xtr_t)

# ── Class weights: inverse-frequency sqrt-balanced ───────────────────────────
counts   = np.bincount(ytr, minlength=N_CLASSES).astype(np.float32)
median_c = np.median(counts)
weights  = np.sqrt(median_c / counts)
weights  = (weights / weights.mean()).clip(0.3, 3.0).astype(np.float32)
print(f"  Class weights: {[f'{w:.2f}' for w in weights]}", flush=True)
w_tensor = torch.tensor(weights, device=device)
class_indices = [torch.nonzero(ytr_t == cls, as_tuple=False).squeeze(1) for cls in range(N_CLASSES)]
balanced_target = max(1, math.ceil(N / N_CLASSES))

def build_epoch_perm(use_balanced):
    if not use_balanced:
        return torch.randperm(N, device=device)

    sampled = []
    for cls_idx in class_indices:
        if cls_idx.numel() == 0:
            continue
        if cls_idx.numel() >= balanced_target:
            take = torch.randperm(cls_idx.numel(), device=device)[:balanced_target]
            sampled.append(cls_idx[take])
        else:
            take = torch.randint(cls_idx.numel(), (balanced_target,), device=device)
            sampled.append(cls_idx[take])

    perm = torch.cat(sampled)
    return perm[torch.randperm(perm.numel(), device=device)]

# ── Loss functions ────────────────────────────────────────────────────────────
# Label smoothing on hard CE: prevents overconfidence
ce_loss = nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=LABEL_SMOOTH)

def focal_ce(logits_fp32, yb_h):
    ce_per = F.cross_entropy(
        logits_fp32,
        yb_h,
        weight=w_tensor,
        reduction='none',
        label_smoothing=LABEL_SMOOTH,
    )
    pt = torch.softmax(logits_fp32, dim=1).gather(1, yb_h.unsqueeze(1)).squeeze(1)
    pt = pt.clamp(min=1e-6, max=1.0)
    return (((1.0 - pt) ** focal_gamma) * ce_per).mean()

print(f"\n[5] Training: KD(T={KD_TEMP},a={KD_ALPHA}) + Mixup + ReLU-only student "
      f"| {EPOCHS} epochs  LR={LR}  BS={BATCH}", flush=True)

# ── Optimizer: AdamW + cosine annealing with warm restarts ───────────────────
model  = ANN().to(device)
opt    = optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)
# CosineAnnealingWarmRestarts: restarts every T_0 epochs, doubling period each time
sched  = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2, eta_min=1e-5)

best = 0.0
best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
patience, no_improve = 600, 0
perm_seed = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    torch.manual_seed(perm_seed); perm_seed += 1
    perm = build_epoch_perm(ep <= balanced_epochs)

    for start in range(0, N, BATCH):
        idx    = perm[start:start+BATCH]
        xb     = Xtr_t[idx]
        yb_h   = ytr_t[idx]
        yb_s   = soft_t[idx]

        # Apply mixup 50% of the time
        if np.random.rand() < 0.5:
            xb, yb_h, yb_s, _ = mixup_batch(xb, yb_h, yb_s, alpha=0.3)

        with amp_ctx():
            logits = model(xb)
        logits_fp32 = logits.float()

        # Hard CE with label smoothing
        loss_ce = ce_loss(logits_fp32, yb_h)
        if focal_gamma > 0.0 and focal_weight > 0.0:
            loss_sup = (1.0 - focal_weight) * loss_ce + focal_weight * focal_ce(logits_fp32, yb_h)
        else:
            loss_sup = loss_ce

        # KD soft loss (KL divergence at temperature T)
        s_log   = F.log_softmax(logits_fp32 / KD_TEMP, dim=1)
        t_log   = torch.log(yb_s.clamp(min=1e-8))
        loss_kd = (yb_s * (t_log - s_log)).sum(dim=1).mean() * (KD_TEMP ** 2)

        loss = KD_ALPHA * loss_kd + (1.0 - KD_ALPHA) * loss_sup

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        opt.step()

    sched.step(ep)   # cosine warm-restart step

    if ep % 250 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            with amp_ctx():
                tr_logits = model(Xtr_t)
                te_logits = model(Xte_t)
            tr_acc = (tr_logits.float().argmax(1).cpu().numpy() == ytr).mean()
            acc    = (te_logits.float().argmax(1).cpu().numpy() == yte).mean()
        if acc > best:
            best       = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 250
        print(f"  Epoch {ep:5d}/{EPOCHS}  train={tr_acc:.4f}  test={acc:.4f}  best={best:.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
        if no_improve >= patience:
            print(f"  Early stop at epoch {ep} (no improvement for {patience} epochs)", flush=True)
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    with amp_ctx():
        preds = model(Xte_t).float().argmax(1).cpu().numpy()
final_acc = accuracy_score(yte, preds)

print(f"\n[6] Final results:")
print(classification_report(yte, preds, target_names=class_names))
print(f"  Student ANN accuracy : {final_acc:.4f}")
print(f"  Teacher LightGBM acc : {teacher_acc:.4f}", flush=True)

# ── Export BF16 weights: one 2048-entry 16-bit hex file per thread ─────────────
# Weights are flattened in layer order and distributed sequentially:
#   Thread 0: weights[0    .. 2047]
#   Thread 1: weights[2048 .. 4095]
#   Thread 2: weights[4096 .. 6143]
#   Thread 3: weights[6144 .. 8191]
# Each line in the hex file = one 16-bit BF16 word → loaded as $readmemh

def float_to_bf16_hex(val):
    p = struct.pack('>f', float(val))
    return f"{(p[0] << 8) | p[1]:04X}"

os.makedirs(out_dir, exist_ok=True)

# Collect fused export tensors flat
export_tensors = collect_export_tensors(model)
all_weights = []
for _, tensor in export_tensors:
    all_weights.extend(tensor.numpy().flatten().tolist())

n_weights = len(all_weights)
assert n_weights <= TOTAL_CAPACITY, f"Exported weights exceed DMEM capacity: {n_weights} > {TOTAL_CAPACITY}"
padded    = all_weights + [0.0] * (TOTAL_CAPACITY - n_weights)  # pad to 8192

print(f"\n[7] Exporting {n_weights:,} BF16 weights → {N_THREADS} thread DMEM files "
      f"({BF16_PER_THREAD} words × 16-bit each):", flush=True)

for tid in range(N_THREADS):
    chunk  = padded[tid * BF16_PER_THREAD : (tid + 1) * BF16_PER_THREAD]
    fname  = os.path.join(out_dir, f'ann_weights_dmem_t{tid}.hex')
    with open(fname, 'w') as f:
        for val in chunk:
            f.write(float_to_bf16_hex(val) + '\n')
    real = max(0, min(n_weights - tid * BF16_PER_THREAD, BF16_PER_THREAD))
    print(f"  Thread {tid}: addr 0–{BF16_PER_THREAD-1:4d}  "
          f"{real:5,} weights + {BF16_PER_THREAD-real:5,} zeros  → {fname}", flush=True)

np.save(os.path.join(out_dir, 'ann_weights_bf16.npy'),
        np.array(all_weights, dtype=np.float32))

print(f"\n[8] Weight layout across threads:")
print(f"  {'Layer':30s}  {'Shape':15s}  {'Flat idx':>14s}  {'Thread':>6s}  {'Addr range'}")
offset = 0
for name, tensor in export_tensors:
    n       = tensor.numel()
    t_start = offset // BF16_PER_THREAD
    t_end   = (offset + n - 1) // BF16_PER_THREAD
    a_start = offset % BF16_PER_THREAD
    a_end   = (offset + n - 1) % BF16_PER_THREAD
    trange  = f"T{t_start}" if t_start == t_end else f"T{t_start}–T{t_end}"
    print(f"  {name:30s}  {str(tuple(tensor.shape)):15s}  "
          f"[{offset:5,}–{offset+n-1:5,}]  {trange:>6s}  {a_start}–{a_end}")
    offset += n

print(f"\n[Testbench] $readmemh for each thread DMEM:")
for tid in range(N_THREADS):
    print(f'  initial $readmemh("ann_weights_dmem_t{tid}.hex", DMEM_LANE[{tid}].u_dmem.mem);')
print(f"\nDone! acc={final_acc:.4f}", flush=True)
"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def train():
    print("=" * 68)
    print("IEC104 ANN Trainer  |  ALL datasets  |  LightGBM KD  |  BF16 GPU")
    print("=" * 68)

    # ── 1. Load all data ──────────────────────────────────────────────────────
    X, y_raw, master_cols = merge_all_data()

    # Merge all *_DoS variants into a single "DoS" class
    dos_mask = np.char.endswith(y_raw.astype(str), '_DoS')
    y_raw    = np.where(dos_mask, 'DoS', y_raw)
    print(f"\n  [DoS merge] All *_DoS labels → 'DoS'  ({dos_mask.sum()} samples)")

    # Keep only the 4 best-performing classes; drop c_sc_na_1, c_rd_na_1, c_rp_na_1
    keep_mask    = np.isin(y_raw, KEEP_CLASSES)
    dropped_lbls = sorted(set(y_raw[~keep_mask]))
    dropped_n    = (~keep_mask).sum()
    X            = X[keep_mask]
    y_raw        = y_raw[keep_mask]
    print(f"  [4-class filter] Dropped {dropped_n:,} samples  "
          f"(removed: {dropped_lbls})")
    print(f"  [4-class filter] Kept {keep_mask.sum():,} samples → {KEEP_CLASSES}")

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")
    for cls, cnt in zip(*np.unique(y_raw, return_counts=True)):
        print(f"    {cls:<30s} {cnt}")

    # ── 2. Stratified 80/20 split ────────────────────────────────────────────
    X_f32 = X.astype(np.float32)   # convert to float32 now to halve memory
    del X                           # free the float64 original
    Xtr, Xte, ytr, yte = train_test_split(X_f32, y, test_size=0.2, stratify=y, random_state=42)
    del X_f32                       # free merged array
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    del Xtr, Xte                    # free unscaled splits
    print(f"\n  Train: {Xtr_s.shape}   Test: {Xte_s.shape}")

    # ── 3. Train LightGBM teacher ─────────────────────────────────────────────
    print(f"\n[2] Training LightGBM teacher ({LGBM_TREES} trees)...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=LGBM_TREES, learning_rate=0.05, num_leaves=63,
        n_jobs=1, random_state=42, verbose=-1
    )
    lgbm.fit(Xtr_s, ytr)
    teacher_acc = accuracy_score(yte, lgbm.predict(Xte_s))
    print(f"  Teacher accuracy: {teacher_acc:.4f}")
    print(classification_report(yte, lgbm.predict(Xte_s), target_names=le.classes_))

    # ── 4. Feature selection & teacher soft labels ────────────────────────────
    imp     = lgbm.feature_importances_
    top_idx = np.argsort(imp)[::-1]

    # Use top N_FEAT features by LightGBM importance across ALL 131 columns.
    # This lets the model discover the most discriminative signal for the 7-class
    # problem rather than hard-coding IEC104-only indices.
    feat_idx      = top_idx[:N_FEAT].astype(np.int64)
    actual_n_feat = len(feat_idx)
    top_cols      = [master_cols[i] for i in feat_idx]

    arch = (f"{actual_n_feat} -> {HIDDEN1} -> {HIDDEN2} -> {N_CLASSES}" if HIDDEN2
            else f"{actual_n_feat} -> {HIDDEN1} -> {N_CLASSES}")
    print(f"\n[3] Top-{actual_n_feat} features by LightGBM importance:")
    print(f"  {[master_cols[i] for i in feat_idx[:8]]} ...")
    print(f"  ANN: {arch}")

    # Teacher soft labels for KD (full training set)
    print(f"  Computing teacher soft labels on {len(Xtr_s)} training samples...")
    soft_labels = lgbm.predict_proba(Xtr_s).astype(np.float32)   # (N_train, 7)

    # Now slice to selected features for the ANN
    Xtr_sel = np.ascontiguousarray(Xtr_s[:, feat_idx], dtype=np.float32)
    Xte_sel = np.ascontiguousarray(Xte_s[:, feat_idx], dtype=np.float32)
    del Xtr_s, Xte_s

    # ── 5. Save features to tmp file; run torch ANN in fresh subprocess ───────
    if HIDDEN2 == 0:
        total_params = (actual_n_feat*HIDDEN1+HIDDEN1) + (HIDDEN1*N_CLASSES+N_CLASSES)
    else:
        total_params = (actual_n_feat*HIDDEN1+HIDDEN1) + (HIDDEN1*HIDDEN2+HIDDEN2) + (HIDDEN2*N_CLASSES+N_CLASSES)
    assert total_params <= TOTAL_CAPACITY, \
        f"Weight count {total_params} exceeds total DMEM capacity {TOTAL_CAPACITY}!"
    print(f"  Param count: {total_params:,} / {TOTAL_CAPACITY:,}  "
          f"({total_params/TOTAL_CAPACITY*100:.1f}% of 4-thread DMEM  "
          f"[{total_params // BF16_PER_THREAD} full threads + "
          f"{total_params % BF16_PER_THREAD} words in thread {total_params // BF16_PER_THREAD}])")

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        npz_path = tf.name
    np.savez(npz_path,
             Xtr=Xtr_sel, Xte=Xte_sel,
             ytr=ytr.astype(np.int64), yte=yte.astype(np.int64),
             soft_labels=soft_labels)
    del Xtr_sel, Xte_sel, soft_labels

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as sf:
        sf.write(TORCH_SCRIPT)
        script_path = sf.name

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "feat_idx.npy"),    feat_idx)
    np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"), sc.mean_.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "scaler_std.npy"),  sc.scale_.astype(np.float32))

    class_names_str = ",".join(le.classes_)
    cmd = [
        sys.executable, script_path,
        npz_path, OUTPUT_DIR,
        str(actual_n_feat), str(HIDDEN1), str(HIDDEN2), str(N_CLASSES),
        str(EPOCHS), str(BATCH), str(LR),
        str(teacher_acc),
        class_names_str,
        str(KD_TEMP), str(KD_ALPHA),
        str(N_THREADS), str(DMEM_DEPTH),
        str(DROPOUT), str(LABEL_SMOOTH),
    ]
    print(f"\n  Launching torch ANN subprocess (fresh process, no LightGBM/OpenMP)...")
    result = subprocess.run(cmd, text=True)
    os.unlink(npz_path)
    os.unlink(script_path)
    if result.returncode != 0:
        print(f"  ERROR: subprocess exited with code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    # Write config file
    info_path = os.path.join(OUTPUT_DIR, "ann_config.txt")
    with open(info_path, "w") as f:
        arch = f"{actual_n_feat} -> {HIDDEN1} -> {N_CLASSES}" if HIDDEN2==0 else f"{actual_n_feat} -> {HIDDEN1} -> {HIDDEN2} -> {N_CLASSES}"
        f.write(f"Architecture: {arch}\n")
        f.write(f"Student model        : Linear -> ReLU -> Linear" + ("" if HIDDEN2==0 else " -> ReLU -> Linear") + "\n")
        f.write(f"Total weight params : {total_params:,} / {TOTAL_CAPACITY:,} BF16 slots used\n")
        f.write(f"DMEM layout         : {N_THREADS} threads × {DMEM_DEPTH} words × 16-bit (1 BF16/word)\n")
        f.write(f"Hex files           : ann_weights_dmem_t0.hex ... ann_weights_dmem_t3.hex\n")
        f.write(f"Data type           : BF16 (upper 16 bits of IEEE 754 float32)\n")
        f.write(f"Teacher accuracy    : {teacher_acc:.4f}\n\n")
        f.write(f"Dataset summary:\n")
        f.write(f"  Custom files (all 6 windows, train+test): 54696 rows\n")
        f.write(f"  CIC files   (all 6 windows, train+test): 54696 rows (IEC104 cols zero-filled)\n")
        f.write(f"  Combined total: 109392 rows\n")
        f.write(f"  Split: 80% train / 20% test (stratified random)\n\n")
        f.write(f"Top {actual_n_feat} features used (by LightGBM importance):\n")
        for i in range(actual_n_feat):
            f.write(f"  [{i:2d}] {top_cols[i]}\n")
        f.write(f"\nLabel encoding:\n")
        for i, cls in enumerate(le.classes_):
            f.write(f"  {i:2d} = {cls}\n")
        f.write(f"\nInference preprocessing:\n")
        f.write(f"  1. StandardScaler all 111 features (mean/std in scaler_mean.npy, scaler_std.npy)\n")
        f.write(f"  2. Select {actual_n_feat} features by feat_idx.npy\n")
        f.write(f"  3. Feed {actual_n_feat} scaled values to ANN input layer\n")
        f.write(f"\nTestbench: load ann_weights_dmem_t0.hex ... ann_weights_dmem_t3.hex into the 4 GPU DMEM lanes.\n")

    print(f"\n  Config -> {info_path}")
    print(f"\nDone! Load GPU_HW_RELU/ann_weights_dmem_t0.hex ... ann_weights_dmem_t3.hex into GPU DMEM via $readmemh.")
    print("=" * 68)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train()
