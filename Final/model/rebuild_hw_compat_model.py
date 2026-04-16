#!/usr/bin/env python3
"""
Rebuild a hardware-compatible 40->96->42->4 BF16 model image from the
partially exported pre-trained hex files currently in Final/model.

The checked-in/exported hex files were flattened from model.parameters(),
which means:
  - BatchNorm running stats were never exported
  - the stream truncates inside fc2.bias at 8192 BF16 words
  - fc3 weights/biases are absent

This utility preserves as much trained structure as possible:
  - fc1 weights/biases are recovered
  - bn1 gamma/beta are folded approximately with identity running stats
  - fc2 weights are recovered in full
  - available fc2 biases are kept, missing tail is zero-filled
  - fc3 is synthesized as a deterministic group-average projection

Outputs:
  - overwrites ann_weights_dmem_t[0-3].hex with a hardware-compatible image
  - keeps backups as ann_weights_raw_dmem_t[0-3].hex
  - writes ann_hw_compat_info.txt with the reconstruction details
"""

from __future__ import annotations

import math
import struct
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
THREAD_DEPTH = 2048
N_THREADS = 4
TOTAL_CAPACITY = THREAD_DEPTH * N_THREADS

N_FEAT = 40
H1 = 96
H2 = 42
N_CLASS = 4

LIVE_T0 = MODEL_DIR / "ann_weights_dmem_t0.hex"
LIVE_T1 = MODEL_DIR / "ann_weights_dmem_t1.hex"
LIVE_T2 = MODEL_DIR / "ann_weights_dmem_t2.hex"
LIVE_T3 = MODEL_DIR / "ann_weights_dmem_t3.hex"
LIVE_FILES = [LIVE_T0, LIVE_T1, LIVE_T2, LIVE_T3]

RAW_T0 = MODEL_DIR / "ann_weights_raw_dmem_t0.hex"
RAW_T1 = MODEL_DIR / "ann_weights_raw_dmem_t1.hex"
RAW_T2 = MODEL_DIR / "ann_weights_raw_dmem_t2.hex"
RAW_T3 = MODEL_DIR / "ann_weights_raw_dmem_t3.hex"
RAW_BACKUP_FILES = [RAW_T0, RAW_T1, RAW_T2, RAW_T3]


def read_hex_words(path: Path) -> list[int]:
    words = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            words.append(int(line, 16))
    if len(words) != THREAD_DEPTH:
        raise ValueError(f"{path} has {len(words)} words, expected {THREAD_DEPTH}")
    return words


def bf16_to_float(word: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (word & 0xFFFF) << 16))[0]


def float_to_bf16_word(val: float) -> int:
    f32 = struct.pack(">f", float(val))
    return (f32[0] << 8) | f32[1]


def load_raw_stream() -> list[float]:
    source_files = RAW_BACKUP_FILES if all(path.exists() for path in RAW_BACKUP_FILES) else LIVE_FILES
    words = []
    for path in source_files:
        words.extend(read_hex_words(path))
    if len(words) != TOTAL_CAPACITY:
        raise ValueError(f"raw stream has {len(words)} words, expected {TOTAL_CAPACITY}")
    return [bf16_to_float(word) for word in words]


def group_average_fc3() -> tuple[list[float], list[float]]:
    counts = [0] * N_CLASS
    for j in range(H2):
        counts[j % N_CLASS] += 1

    w = [0.0] * (N_CLASS * H2)
    for j in range(H2):
        cls = j % N_CLASS
        w[cls * H2 + j] = 1.0 / counts[cls]

    b = [0.0] * N_CLASS
    return w, b


def rebuild_hw_image(raw: list[float]) -> tuple[list[float], dict[str, tuple[int, int]]]:
    # Raw checked-in export layout:
    #   fc1.weight [0:3840)
    #   fc1.bias   [3840:3936)
    #   bn1.weight [3936:4032)
    #   bn1.bias   [4032:4128)
    #   fc2.weight [4128:8160)
    #   fc2.bias   [8160:8192)   # truncated, only 32/42 exported
    fc1_w = raw[0:3840]
    fc1_b = raw[3840:3936]
    bn1_g = raw[3936:4032]
    bn1_b = raw[4032:4128]
    fc2_w = raw[4128:8160]
    fc2_b_partial = raw[8160:8192]

    # Approximate fold because BN running stats were never exported.
    # Use identity running stats: mean=0, var=1.
    fc1_w_fused = [0.0] * len(fc1_w)
    fc1_b_fused = [0.0] * len(fc1_b)
    for o in range(H1):
        scale = bn1_g[o]
        fc1_b_fused[o] = fc1_b[o] * scale + bn1_b[o]
        row_off = o * N_FEAT
        for i in range(N_FEAT):
            fc1_w_fused[row_off + i] = fc1_w[row_off + i] * scale

    fc2_b = list(fc2_b_partial) + [0.0] * (H2 - len(fc2_b_partial))
    fc3_w, fc3_b = group_average_fc3()

    flat = []
    spans = {}

    def append(name: str, values: list[float]) -> None:
        start = len(flat)
        flat.extend(values)
        spans[name] = (start, len(flat) - 1)

    append("fc1.weight_fused", fc1_w_fused)
    append("fc1.bias_fused", fc1_b_fused)
    append("fc2.weight", fc2_w)
    append("fc2.bias_partial+zerofill", fc2_b)
    append("fc3.weight_groupavg", fc3_w)
    append("fc3.bias_zero", fc3_b)

    expected = (N_FEAT * H1 + H1) + (H1 * H2 + H2) + (H2 * N_CLASS + N_CLASS)
    if len(flat) != expected:
        raise ValueError(f"rebuilt image has {len(flat)} weights, expected {expected}")

    padded = flat + [0.0] * (TOTAL_CAPACITY - len(flat))
    return padded, spans


def write_hex_files(flat: list[float]) -> None:
    for tid, src in enumerate(LIVE_FILES):
        backup = RAW_BACKUP_FILES[tid]
        if not backup.exists():
            backup.write_text(src.read_text())

        chunk = flat[tid * THREAD_DEPTH : (tid + 1) * THREAD_DEPTH]
        dst = MODEL_DIR / f"ann_weights_dmem_t{tid}.hex"
        dst.write_text("".join(f"{float_to_bf16_word(v):04X}\n" for v in chunk))


def bf16_round(val: float) -> float:
    return bf16_to_float(float_to_bf16_word(val))


def bf16_parts(word: int) -> tuple[int, int, int]:
    return (word >> 15) & 1, (word >> 7) & 0xFF, word & 0x7F


def bf16_mult_word(a: int, b: int) -> int:
    sign_a, exp_a, frac_a = bf16_parts(a)
    sign_b, exp_b, frac_b = bf16_parts(b)
    sign_r = sign_a ^ sign_b

    is_zero_a = (exp_a == 0) and (frac_a == 0)
    is_zero_b = (exp_b == 0) and (frac_b == 0)
    is_inf_a = (exp_a == 0xFF) and (frac_a == 0)
    is_inf_b = (exp_b == 0xFF) and (frac_b == 0)
    is_nan_a = (exp_a == 0xFF) and (frac_a != 0)
    is_nan_b = (exp_b == 0xFF) and (frac_b != 0)
    is_denorm_a = exp_a == 0
    is_denorm_b = exp_b == 0

    mant_a = ((0 if is_denorm_a else 1) << 7) | frac_a
    mant_b = ((0 if is_denorm_b else 1) << 7) | frac_b
    mant_prod = mant_a * mant_b

    exp_sum = exp_a + exp_b - 127
    prod_ovf = (mant_prod >> 15) & 1
    mant_norm = (mant_prod >> (8 if prod_ovf else 7)) & 0xFF
    exp_norm = exp_sum + (1 if prod_ovf else 0)

    guard = (mant_prod >> (7 if prod_ovf else 6)) & 1
    round_bit = (mant_prod >> (6 if prod_ovf else 5)) & 1
    sticky = 1 if (mant_prod & ((1 << (6 if prod_ovf else 5)) - 1)) != 0 else 0

    round_up = guard & (round_bit | sticky | (mant_norm & 1))
    mant_rounded = mant_norm + round_up
    round_ovf = (mant_rounded >> 8) & 1
    frac_final = ((mant_rounded >> 1) if round_ovf else mant_rounded) & 0x7F
    exp_final = exp_norm + (1 if round_ovf else 0)

    if is_nan_a or is_nan_b:
        return 0x7FC0
    if is_inf_a or is_inf_b:
        if is_zero_a or is_zero_b:
            return 0x7FC0
        return (sign_r << 15) | (0xFF << 7)
    if is_zero_a or is_zero_b:
        return sign_r << 15
    if exp_final >= 255:
        return (sign_r << 15) | (0xFF << 7)
    if exp_final == 0 or exp_final < 0:
        return sign_r << 15
    return (sign_r << 15) | ((exp_final & 0xFF) << 7) | frac_final


def bf16_add_word(a: int, b: int) -> int:
    sign_a, exp_a, frac_a = bf16_parts(a)
    sign_b, exp_b, frac_b = bf16_parts(b)

    is_zero_a = (exp_a == 0) and (frac_a == 0)
    is_zero_b = (exp_b == 0) and (frac_b == 0)
    is_inf_a = (exp_a == 0xFF) and (frac_a == 0)
    is_inf_b = (exp_b == 0xFF) and (frac_b == 0)
    is_nan_a = (exp_a == 0xFF) and (frac_a != 0)
    is_nan_b = (exp_b == 0xFF) and (frac_b != 0)

    mant_a = ((1 if exp_a != 0 else 0) << 7) | frac_a
    mant_b = ((1 if exp_b != 0 else 0) << 7) | frac_b

    sign_b_eff = sign_b
    eff_sub = sign_a ^ sign_b_eff
    a_lt_b = (exp_a < exp_b) or ((exp_a == exp_b) and (mant_a < mant_b))

    sign_lg = sign_b_eff if a_lt_b else sign_a
    exp_lg = exp_b if a_lt_b else exp_a
    mant_lg = mant_b if a_lt_b else mant_a
    mant_sm = mant_a if a_lt_b else mant_b
    exp_diff = (exp_b if a_lt_b else exp_a) - (exp_a if a_lt_b else exp_b)

    special = False
    special_result = 0
    if is_nan_a or is_nan_b:
        special = True
        special_result = 0x7FC0
    elif is_inf_a and is_inf_b:
        special = True
        special_result = 0x7FC0 if eff_sub else ((sign_a << 15) | (0xFF << 7))
    elif is_inf_a:
        special = True
        special_result = (sign_a << 15) | (0xFF << 7)
    elif is_inf_b:
        special = True
        special_result = (sign_b_eff << 15) | (0xFF << 7)
    elif is_zero_a and is_zero_b:
        special = True
        special_result = 0x8000 if (sign_a and sign_b_eff) else 0x0000
    elif is_zero_a:
        special = True
        special_result = (sign_b_eff << 15) | (exp_b << 7) | frac_b
    elif is_zero_b:
        special = True
        special_result = a

    sm_ext = mant_sm << 10
    shift_amt = 15 if exp_diff > 15 else exp_diff
    sm_shifted = sm_ext >> shift_amt
    mant_sm_al = (sm_shifted >> 10) & 0xFF
    guard = (sm_shifted >> 9) & 1
    round_bit = (sm_shifted >> 8) & 1
    sticky = 1 if ((sm_shifted & 0xFF) != 0 or exp_diff > 15) else 0

    mant_sum = (mant_lg - mant_sm_al) if eff_sub else (mant_lg + mant_sm_al)
    sum_zero = mant_sum == 0

    lead_pos = 0
    for pos in range(9, -1, -1):
        if (mant_sum >> pos) & 1:
            lead_pos = pos
            break

    lshift_need = (7 - lead_pos) if lead_pos < 7 else 0
    rshift_need = (lead_pos - 7) if lead_pos > 7 else 0
    max_lshift = (exp_lg - 1) if exp_lg > 1 else 0
    act_lshift = max_lshift if lshift_need > max_lshift else lshift_need

    if rshift_need != 0:
        mant_norm = mant_sum >> rshift_need
    elif act_lshift != 0:
        mant_norm = (mant_sum << act_lshift) & 0x3FF
    else:
        mant_norm = mant_sum
    exp_norm = exp_lg + rshift_need - act_lshift

    if rshift_need != 0:
        g_n = (mant_sum >> (rshift_need - 1)) & 1
    else:
        g_n = guard
    if rshift_need >= 2:
        r_n = mant_sum & 1
    elif rshift_need == 1:
        r_n = guard
    else:
        r_n = round_bit
    s_n = (guard | round_bit | sticky) if rshift_need != 0 else sticky

    round_up = g_n & (r_n | s_n | (mant_norm & 1))
    mant_rounded = (mant_norm & 0xFF) + round_up
    round_ovf = (mant_rounded >> 8) & 1
    frac_final = ((mant_rounded >> 1) if round_ovf else mant_rounded) & 0x7F
    exp_final = exp_norm + (1 if round_ovf else 0)

    if special:
        return special_result
    if sum_zero:
        return 0x0000
    if exp_final >= 255:
        return (sign_lg << 15) | (0xFF << 7)
    if exp_final == 0 or exp_final < 0:
        return sign_lg << 15
    return (sign_lg << 15) | ((exp_final & 0xFF) << 7) | frac_final


def relu_word(word: int) -> int:
    return 0x0000 if (word & 0x8000) else word


def input_pattern(feat_idx: int, pkt_idx: int) -> float:
    pat = (feat_idx + pkt_idx) % 5
    if pat == 0:
        return -1.0
    if pat == 1:
        return -0.5
    if pat == 2:
        return 0.0
    if pat == 3:
        return 0.5
    return 1.0


def run_reference(flat: list[float]) -> list[list[int]]:
    words = [float_to_bf16_word(v) for v in flat]
    off = 0
    w1 = words[off : off + H1 * N_FEAT]
    off += H1 * N_FEAT
    b1 = words[off : off + H1]
    off += H1
    w2 = words[off : off + H2 * H1]
    off += H2 * H1
    b2 = words[off : off + H2]
    off += H2
    w3 = words[off : off + N_CLASS * H2]
    off += N_CLASS * H2
    b3 = words[off : off + N_CLASS]

    x = [[float_to_bf16_word(input_pattern(f, p)) for p in range(4)] for f in range(N_FEAT)]

    h1 = [[0] * 4 for _ in range(H1)]
    for o in range(H1):
        for p in range(4):
            acc = b1[o]
            for i in range(N_FEAT):
                acc = bf16_add_word(acc, bf16_mult_word(w1[o * N_FEAT + i], x[i][p]))
            h1[o][p] = relu_word(acc)

    h2 = [[0] * 4 for _ in range(H2)]
    for o in range(H2):
        for p in range(4):
            acc = b2[o]
            for i in range(H1):
                acc = bf16_add_word(acc, bf16_mult_word(w2[o * H1 + i], h1[i][p]))
            h2[o][p] = relu_word(acc)

    y = [[0] * 4 for _ in range(N_CLASS)]
    for o in range(N_CLASS):
        for p in range(4):
            acc = b3[o]
            for i in range(H2):
                acc = bf16_add_word(acc, bf16_mult_word(w3[o * H2 + i], h2[i][p]))
            y[o][p] = acc

    return y


def main() -> None:
    missing = [str(path) for path in LIVE_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing raw ANN hex files: " + ", ".join(missing))

    raw = load_raw_stream()
    flat, spans = rebuild_hw_image(raw)
    write_hex_files(flat)

    logits = run_reference(flat)
    info = MODEL_DIR / "ann_hw_compat_info.txt"
    with info.open("w") as f:
        f.write("Hardware-compatible ANN image rebuilt from partial raw export.\n")
        f.write("Notes:\n")
        f.write("  - fc1 and fc2 recovered from existing export\n")
        f.write("  - bn1 approximately folded using gamma/beta with identity running stats\n")
        f.write("  - missing 10 entries of fc2.bias were zero-filled\n")
        f.write("  - fc3 synthesized as deterministic group-average projection over 42 hidden units\n")
        f.write("\nSpans:\n")
        for name, (s, e) in spans.items():
            f.write(f"  {name}: [{s}, {e}]\n")
        f.write("\nReference logits for model_tb input pattern (BF16 words):\n")
        for idx, row in enumerate(logits):
            f.write(f"  row{idx}: {' '.join(f'{w:04X}' for w in row)}\n")

    print(f"Wrote hardware-compatible ANN image to {MODEL_DIR}")
    print(f"Backup raw files kept as ann_weights_raw_dmem_t[0-3].hex")
    print(f"Reference info: {info}")
    for idx, row in enumerate(logits):
        print(f"row{idx}: {' '.join(f'{w:04X}' for w in row)}")


if __name__ == "__main__":
    main()
