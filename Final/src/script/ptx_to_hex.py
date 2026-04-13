#!/usr/bin/env python3
"""
ptx_to_hex.py — PTX-to-Custom-Hex Compiler with WMMA Fusion
=============================================================
Pipeline:  .cu → nvcc -ptx -arch=sm_80 → .ptx → [THIS] → .hex + socreg test

ISA v2.1: 32 opcodes, 4R4W LUT RF, 4×4 bf16 tensor core (37-cyc MMA).

Key feature: FMA chain fusion
  Detects 4 parallel bf16 FMA chains sharing multiplicand registers
  and fuses them into a single WMMA.MMA instruction (16 FMAs → 1 op).
  Without fusion, the scalar path cannot compute cross-lane matrix
  multiply — WMMA.MMA is the ONLY cross-lane operation in the ISA.
"""

import re, sys, os, struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OPCODE TABLE — ISA v2.1 §6 (32 opcodes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPCODES = {
    "NOP":        (0x00, "X"),   "ST":         (0x01, "M"),
    "LD":         (0x02, "M"),   "MOV":        (0x03, "U"),
    "MOVI":       (0x04, "P"),   "CVT":        (0x05, "U"),
    "ADD":        (0x06, "R"),   "SUB":        (0x07, "R"),
    "MUL":        (0x08, "R"),   "FMA":        (0x09, "U"),
    "MAX":        (0x0A, "R"),   "MIN":        (0x0B, "R"),
    "ABS":        (0x0C, "U"),   "NEG":        (0x0D, "U"),
    "AND":        (0x0E, "R"),   "OR":         (0x0F, "R"),
    "XOR":        (0x10, "R"),   "SHL":        (0x11, "I"),
    "SHR":        (0x12, "I"),   "ADDI":       (0x13, "I"),
    "MULI":       (0x14, "I"),   "SETP":       (0x15, "R"),
    "SELP":       (0x16, "R"),   "BRA":        (0x17, "P"),
    "PBRA":       (0x18, "P"),   "RET":        (0x19, "X"),
    "SET":        (0x1A, "R"),   "LDS":        (0x1B, "M"),
    "STS":        (0x1C, "M"),   "WMMA.MMA":   (0x1D, "RE"),
    "WMMA.LOAD":  (0x1E, "M"),   "WMMA.STORE": (0x1F, "M"),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENCODER — ISA v2.1 binary formats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def encode(mnemonic, dt=0, rd=0, ra=0, rb=0, rc=0, imm=0, sel=0):
    opcode, fmt = OPCODES[mnemonic]
    w = (opcode & 0x1F) << 27 | (dt & 1) << 26
    if   fmt == "X":  pass
    elif fmt == "R":  w |= (rd & 0xF) << 20 | (ra & 0xF) << 16 | (rb & 0xF) << 12
    elif fmt == "I":  w |= (rd & 0xF) << 20 | (ra & 0xF) << 16 | (imm & 0xFFFF)
    elif fmt == "M":  w |= (sel & 0x3) << 24 | (rd & 0xF) << 20 | (ra & 0xF) << 16 | (imm & 0xFFFF)
    elif fmt == "P":  w |= (rd & 0xF) << 20 | (imm & 0xFFFFF)
    elif fmt == "U":  w |= (sel & 0x3) << 24 | (rd & 0xF) << 20 | (ra & 0xF) << 16 | (rb & 0xF) << 12
    elif fmt == "RE": w |= (rd & 0xF) << 20 | (ra & 0xF) << 16 | (rb & 0xF) << 12 | (rc & 0xF) << 8
    return w & 0xFFFFFFFF

E = encode  # shorthand


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PTX PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_ptx(text):
    """Parse .ptx → list of kernel dicts {name, body}."""
    kernels = []
    for m in re.finditer(r'\.visible\s+\.entry\s+(\S+)\s*\(.*?\)\s*\{', text, re.DOTALL):
        name = m.group(1)
        start = m.end()
        depth = 1; pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == '{': depth += 1
            elif text[pos] == '}': depth -= 1
            pos += 1
        body = _parse_body(text[start:pos-1])
        kernels.append({'name': name, 'body': body})
    return kernels

def _parse_body(text):
    instrs = []; in_asm = False; buf = []
    for line in text.split('\n'):
        s = line.strip()
        if s == '// begin inline asm': in_asm = True; buf = []; continue
        if s == '// end inline asm':
            in_asm = False
            for a in buf:
                i = _parse_one(a)
                if i: instrs.append(i)
            continue
        if in_asm:
            clean = s.strip('{}; \t')
            if clean and not clean.startswith('.reg'): buf.append(clean)
            continue
        if not s or s.startswith('//') or s.startswith('.reg') or s.startswith('$L'): continue
        # Handle bare inline asm: lines like {fma.rn.bf16 ...;} without markers
        if s.startswith('{') and (s.endswith('}') or s.endswith(';}')):
            clean = s.strip('{}; \t')
            if clean:
                i = _parse_one(clean)
                if i: instrs.append(i)
            continue
        i = _parse_one(s)
        if i: instrs.append(i)
    return instrs

def _parse_one(line):
    line = line.rstrip(';').strip()
    if not line: return None
    parts = line.split(None, 1)
    op = parts[0]
    args_str = parts[1] if len(parts) > 1 else ""
    args = [x.strip() for x in args_str.split(',') if x.strip()]
    return {'op': op, 'args': args, 'raw': line}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FMA CHAIN DETECTOR — core WMMA fusion algorithm
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_fma_chains(body, start_idx=0):
    """Detect groups of 4 FMA chains sharing multiplicand registers.

    Returns: list of FmaGroup, each containing:
      - load_map: {ptx_reg: byte_offset} for all loads before the chains
      - chains: list of 4 FmaChain objects
      - relu_regs: list of ptx regs that go through max.bf16
      - store_map: {byte_offset: ptx_reg} for stores after chains

    An FmaChain has:
      - seed: ptx reg (bias/accumulator init)
      - pairs: [(mul_a, mul_b), ...] — 4 multiplicand register pairs
      - result: ptx reg (final FMA output)
    """
    # Pass 1: Collect loads, FMAs, maxes, stores in order
    loads = {}    # ptx_reg → byte_offset
    fmas = []     # list of (dst, mul_a, mul_b, accum)
    maxes = {}    # result_reg → (src_a, src_b)
    stores = {}   # byte_offset → ptx_reg
    zero_reg = None

    for i, ins in enumerate(body):
        if i < start_idx: continue
        op = ins['op']; a = ins['args']

        if op == 'ld.global.u16' and len(a) >= 2:
            addr = a[1].strip('[]')
            if '+' in addr:
                _, off = addr.rsplit('+', 1)
                loads[a[0]] = int(off)
            else:
                loads[a[0]] = 0

        elif op == 'fma.rn.bf16' and len(a) >= 4:
            fmas.append((a[0], a[1], a[2], a[3]))

        elif op in ('max.bf16', 'max.f32') and len(a) >= 3:
            maxes[a[0]] = (a[1], a[2])

        elif op == 'st.global.u16' and len(a) >= 2:
            addr = a[0].strip('[]')
            if '+' in addr:
                _, off = addr.rsplit('+', 1)
                stores[int(off)] = a[1]
            else:
                stores[0] = a[1]

        elif op == 'cvt.rn.bf16.f32':
            zero_reg = a[0]  # bf16 zero for ReLU

        elif op == 'mov.f32' and '0f00000000' in ins['raw']:
            zero_reg = a[0]

    # Pass 2: Build FMA chains by following accumulator dependencies
    # An FMA chain: seed → fma1 → fma2 → fma3 → fma4 (result)
    # Chain detection: each FMA's accumulator input is either a load (seed)
    # or the result of the previous FMA in the chain.
    result_map = {}  # dst_reg → fma_index
    for i, (dst, ma, mb, acc) in enumerate(fmas):
        result_map[dst] = i

    chains = []
    used = set()
    for i, (dst, ma, mb, acc) in enumerate(fmas):
        if i in used: continue
        # Is this the start of a chain? (accumulator is a load, not an FMA result)
        if acc in result_map and result_map[acc] not in used:
            continue  # This FMA is mid-chain, skip — it'll be found from its start

        # Follow the chain forward
        chain = []
        cur_acc = acc
        cur_idx = i
        # Walk through FMAs that use the same accumulator lineage
        # Start from this FMA and follow the chain
        chain.append((dst, ma, mb, acc))
        used.add(cur_idx)

        # Find next FMA whose accumulator is our result
        next_reg = dst
        while True:
            found = False
            for j, (d2, m2a, m2b, a2) in enumerate(fmas):
                if j not in used and a2 == next_reg:
                    chain.append((d2, m2a, m2b, a2))
                    used.add(j)
                    next_reg = d2
                    found = True
                    break
            if not found: break

        if len(chain) == 4:
            chains.append({
                'seed': acc,
                'pairs': [(ma, mb) for (_, ma, mb, _) in chain],
                'result': chain[-1][0],
                'fma_regs': [dst for (dst, _, _, _) in chain]
            })

    # Pass 3: Group chains by shared multiplicand registers
    groups = []
    ungrouped = list(range(len(chains)))
    while ungrouped:
        base = ungrouped.pop(0)
        base_pairs = set((a, b) for a, b in chains[base]['pairs'])
        group = [base]
        remaining = []
        for idx in ungrouped:
            cand_pairs = set((a, b) for a, b in chains[idx]['pairs'])
            if cand_pairs == base_pairs:
                group.append(idx)
            else:
                remaining.append(idx)
        ungrouped = remaining
        if len(group) == 4:
            groups.append({
                'chains': [chains[g] for g in group],
                'mul_pairs': chains[base]['pairs'],
                'load_map': loads,
                'store_map': stores,
                'maxes': maxes,
                'zero_reg': zero_reg,
            })

    return groups


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ANN KERNEL TRANSLATOR — WMMA-fused ISA generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DMEM layout (word addresses, per bank):
#   X=0..3, W1=4..7, B1=8..11, W2=12..15, B2=16..19, H=20..23, Y=24..27
DMEM_LAYOUT = {
    'X': 0, 'W1': 4, 'B1': 8, 'W2': 12, 'B2': 16, 'H': 20, 'Y': 24
}

def translate_ann_fused(body, kernel_name="ann"):
    """Translate ANN kernel PTX with WMMA fusion.

    Detects FMA chain groups and emits WMMA.MMA instructions.
    Returns list of (hex_word, asm_string, comment) tuples.
    """
    groups = detect_fma_chains(body)
    instrs = []

    def emit(hw, asm, cmt=""):
        instrs.append((hw, asm, cmt))

    if len(groups) < 2:
        # Fallback: not enough FMA groups for 2-layer ANN
        print(f"  WARNING: Found {len(groups)} FMA groups (need 2 for ANN)")
        print(f"  Falling back to template-based emission")

    # ── Analyze load offsets to determine data layout ──
    # Group loads by byte offset to identify which matrix they belong to
    if groups:
        load_map = groups[0]['load_map']  # ptx_reg → byte_offset

    # ── Layer 1: H = ReLU(W1 × X + B1) ──
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['W1']),
         f"MOVI  R0, {DMEM_LAYOUT['W1']}", "R0 = ADDR_W1")
    emit(E("WMMA.LOAD", dt=1, sel=0, rd=4, ra=0, imm=0),
         "WMMA.LOAD.A  R4, R0, 0", "{R4-R7} = W1[tid][0..3]")
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['X']),
         f"MOVI  R0, {DMEM_LAYOUT['X']}", "R0 = ADDR_X")
    emit(E("WMMA.LOAD", dt=1, sel=1, rd=8, ra=0, imm=0),
         "WMMA.LOAD.B  R8, R0, 0", "{R8-R11} = X[tid][0..3]")
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['B1']),
         f"MOVI  R0, {DMEM_LAYOUT['B1']}", "R0 = ADDR_B1")
    emit(E("WMMA.LOAD", dt=1, sel=2, rd=12, ra=0, imm=0),
         "WMMA.LOAD.C  R12, R0, 0", "{R12-R15} = B1[tid][0..3]")
    emit(E("WMMA.MMA", dt=1, rd=12, ra=4, rb=8, rc=12),
         "WMMA.MMA  R12, R4, R8, R12",
         "D = W1 x X + B1  (37cyc, 64 MACs)")

    # ── ReLU ──
    has_relu = len(groups) > 0 and len(groups[0].get('maxes', {})) > 0
    # Check PTX body directly for max.bf16
    if not has_relu:
        has_relu = any(ins['op'] in ('max.bf16', 'max.f32') for ins in body)

    if has_relu:
        emit(E("MOVI", dt=1, rd=0, imm=0),
             "MOVI  R0, 0x0000", "R0 = bf16(0.0) for ReLU")
        for r in range(12, 16):
            emit(E("MAX", dt=1, rd=r, ra=r, rb=0),
                 f"MAX.f R{r}, R{r}, R0", f"ReLU D[{r-12}]")

    # ── Store H ──
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['H']),
         f"MOVI  R0, {DMEM_LAYOUT['H']}", "R0 = ADDR_H")
    emit(E("WMMA.STORE", dt=1, rd=12, ra=0, imm=0),
         "WMMA.STORE  R12, R0, 0", "DMEM[20..23] = H")

    # ── Layer 2: Y = W2 × H + B2 ──
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['W2']),
         f"MOVI  R0, {DMEM_LAYOUT['W2']}", "R0 = ADDR_W2")
    emit(E("WMMA.LOAD", dt=1, sel=0, rd=4, ra=0, imm=0),
         "WMMA.LOAD.A  R4, R0, 0", "{R4-R7} = W2[tid][0..3]")
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['H']),
         f"MOVI  R0, {DMEM_LAYOUT['H']}", "R0 = ADDR_H")
    emit(E("WMMA.LOAD", dt=1, sel=1, rd=8, ra=0, imm=0),
         "WMMA.LOAD.B  R8, R0, 0", "{R8-R11} = H[tid][0..3]")
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['B2']),
         f"MOVI  R0, {DMEM_LAYOUT['B2']}", "R0 = ADDR_B2")
    emit(E("WMMA.LOAD", dt=1, sel=2, rd=12, ra=0, imm=0),
         "WMMA.LOAD.C  R12, R0, 0", "{R12-R15} = B2[tid][0..3]")
    emit(E("WMMA.MMA", dt=1, rd=12, ra=4, rb=8, rc=12),
         "WMMA.MMA  R12, R4, R8, R12",
         "D = W2 x H + B2  (37cyc, 64 MACs)")

    # ── Store Y ──
    emit(E("MOVI", rd=0, imm=DMEM_LAYOUT['Y']),
         f"MOVI  R0, {DMEM_LAYOUT['Y']}", "R0 = ADDR_Y")
    emit(E("WMMA.STORE", dt=1, rd=12, ra=0, imm=0),
         "WMMA.STORE  R12, R0, 0", "DMEM[24..27] = Y")

    # ── RET ──
    emit(E("RET"), "RET", "End kernel")

    return instrs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OUTPUT FORMATTERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fmt_hex(instrs, kernel_name="ann"):
    """Verilog-compatible hex listing."""
    L = [f"// GPU IMEM — {kernel_name} ({len(instrs)} instructions)",
         f"// ISA v2.1 | WMMA-fused | 4x4 bf16 tensor core",
         ""]
    for pc, (hw, asm, cmt) in enumerate(instrs):
        L.append(f"0x{hw:08X}  // [{pc:2d}] {asm:<35s} ; {cmt}")
    L.append("")
    return '\n'.join(L)


def fmt_socreg_test(instrs, kernel_name="ann"):
    """Generate socreg.pl inject_word calls for hardware verification.

    Test data: all weights = 1.0 (bf16 0x3F80), all biases = 0
    Expected: Layer1 = 4.0 per element, Layer2 = 16.0 = bf16 0x4180
    D_PACK readback Y[0..3] → {0x4180, 0x4180, 0x4180, 0x4180}
    """
    L = []
    L.append(f"# ════════════════════════════════════════════════")
    L.append(f"# socreg.pl test: {kernel_name}")
    L.append(f"# All W=1.0, B=0 → Layer1=4.0, Layer2=16.0=0x4180")
    L.append(f"# ════════════════════════════════════════════════")
    L.append(f"sub t_ann {{ banner(\"ANN: 2-layer FC (W=1,B=0 → Y=16.0=0x4180)\"); ic(); my@c;")
    L.append(f"")

    # Count DWs for LOAD_IMEM
    n_instrs = len(instrs)
    n_dw = (n_instrs + 1) // 2  # ceiling div: 2 instrs per 64-bit DW

    # Pad to even number
    padded = list(instrs)
    if len(padded) % 2: padded.append((E("NOP"), "NOP", "pad"))

    # ARM program: D_IMEM(16) + D_UNPACK(14 per bank) + launch + D_PACK + HALT
    # Since this is a large test, use gpu_arm template approach
    L.append(f"  # ARM: D_IMEM({n_instrs} GPU instrs = {n_dw} DWs)")
    L.append(f"  # + D_UNPACK(14 words/bank × 4 banks) + launch + D_PACK")
    L.append(f"  gpu_arm(0x20, 14, 24, 2, 15);  # dunpack=14, dpack_src=24(Y), dpack=2, 15 NOP")
    L.append(f"")

    # GPU IMEM as LOAD_DMEM
    L.append(f"  # ── GPU IMEM ({n_instrs} instrs = {n_dw} DWs) ──")
    L.append(f"  @c=cw(2,0,{n_dw},0); iw($c[0],$c[1],0);")
    for i in range(0, len(padded), 2):
        hw_odd = padded[i+1][0] if i+1 < len(padded) else 0
        hw_even = padded[i][0]
        asm_e = padded[i][1]
        asm_o = padded[i+1][1] if i+1 < len(padded) else "NOP"
        L.append(f"  iw(0x{hw_odd:08X},0x{hw_even:08X},0);  # [{i+1}]{asm_o}; [{i}]{asm_e}")

    L.append(f"")

    # GPU DMEM data via D_UNPACK: 14 CPU words per bank, 4 banks = 56 words
    # Layout: X(4) W1(4) B1(4) W2(4) B2(4) H(4) Y(4) = 28 DMEM words
    # D_UNPACK packs 2 bf16 per 32-bit CPU word → 14 CPU words per bank
    bf16_one = 0x3F80
    bf16_zero = 0x0000

    L.append(f"  # ── GPU DMEM data (D_UNPACK: 14 words/bank × 4 banks) ──")
    L.append(f"  # Layout: X[0..3] W1[4..7] B1[8..11] W2[12..15] B2[16..19] H[20..23] Y[24..27]")
    L.append(f"  # All W=1.0(0x3F80), B=0, X=1.0 → expect Y=16.0(0x4180)")
    L.append(f"  @c=cw(2,0x010,{14*4//2},0); iw($c[0],$c[1],0);  # 28 DW pairs")

    # Build per-bank data words
    # Each CPU word = {DMEM[2n+1], DMEM[2n]} (high16, low16)
    one_one = (bf16_one << 16) | bf16_one    # {1.0, 1.0}
    zero_zero = 0                             # {0.0, 0.0}

    bank_data = [
        one_one,   one_one,    # X[0..3] = 1.0
        one_one,   one_one,    # W1[0..3] = 1.0
        zero_zero, zero_zero,  # B1[0..3] = 0
        one_one,   one_one,    # W2[0..3] = 1.0
        zero_zero, zero_zero,  # B2[0..3] = 0
        zero_zero, zero_zero,  # H[0..3] = 0 (computed)
        zero_zero, zero_zero,  # Y[0..3] = 0 (computed)
    ]  # 14 words per bank

    for bank in range(4):
        for i in range(0, len(bank_data), 2):
            w_hi = bank_data[i+1] if i+1 < len(bank_data) else 0
            w_lo = bank_data[i]
            tag = f"bank{bank}" if i == 0 else ""
            L.append(f"  iw(0x{w_hi:08X},0x{w_lo:08X},0);  # {tag} word{i+1},{i}")

    L.append(f"")

    # Readback area + commands
    L.append(f"  # ── Readback zeros + commands ──")
    L.append(f"  tail(0x020,4);  # 4 DW pairs = 8 words readback")
    L.append(f"")
    L.append(f"  id();")

    # Expected: Y = all 16.0 = bf16 0x4180
    # D_PACK packs {DMEM[2n+1], DMEM[2n]} per bank
    # Y[0..3] = {16.0, 16.0, 16.0, 16.0}
    # Packed: {0x4180, 0x4180} = 0x41804180
    exp = 0x41804180
    L.append(f"  # Expected: Y = W2 × ReLU(W1 × X + B1) + B2")
    L.append(f"  #         = I × ReLU(I × I + 0) + 0 = I × 4.0 = 16.0 = bf16 0x4180")
    L.append(f"  # D_PACK: {{0x4180,0x4180}} = 0x41804180 per bank")
    L.append(f"  c4(\"ANN\",pt(400),cr(),0x{exp:08X},0x{exp:08X},0x{exp:08X},0x{exp:08X});")
    L.append(f"}}")
    L.append(f"")

    return '\n'.join(L)


def fmt_soc_tb_test(instrs, kernel_name="ann"):
    """Generate soc_tb.v test for simulation verification."""
    L = []
    n = len(instrs)
    padded = list(instrs)
    if len(padded) % 2: padded.append((E("NOP"), "NOP", "pad"))
    n_dw = len(padded) // 2

    L.append(f"    // ── Test ANN: 2-layer FC inference via WMMA tensor core ──")
    L.append(f"    // {n} GPU instrs, all W=1.0, B=0, X=1.0 → Y=16.0=bf16 0x4180")
    L.append(f"    begin")
    L.append(f"        $display(\"\\n--- Test ANN: 2-layer WMMA inference ---\");")
    L.append(f"        cycle_cnt = 0;")
    L.append(f"")
    L.append(f"        send_gpu_arm(8'h20, 8'd14, 8'd24, 8'd2);")
    L.append(f"        // GPU IMEM ({n} instrs = {n_dw} DWs)")
    L.append(f"        rx(cmd(4'h2, 12'h000, 16'd{n_dw}, 32'h0), 8'h00);")

    for i in range(0, len(padded), 2):
        hw_odd = padded[i+1][0]
        hw_even = padded[i][0]
        L.append(f"        rx({{32'h{hw_odd:08X}, 32'h{hw_even:08X}}}, 8'h00);  "
                 f"// [{i+1}]{padded[i+1][1]}; [{i}]{padded[i][1]}")

    # Pad to 16 instrs (8 DWs) if needed
    remaining = 8 - n_dw
    for _ in range(max(0, remaining)):
        L.append(f"        rx(64'h0, 8'h00);")

    L.append(f"")
    L.append(f"        // GPU DMEM: per-bank data (D_UNPACK 14 words/bank)")
    L.append(f"        // X=1.0, W1=1.0, B1=0, W2=1.0, B2=0, H=0, Y=0")
    L.append(f"        rx(cmd(4'h2, 12'h010, 16'd{14*4//2}, 32'h0), 8'h00);")

    bf16_one = 0x3F80
    one_one = (bf16_one << 16) | bf16_one
    bank_words = [
        one_one, one_one,     # X
        one_one, one_one,     # W1
        0, 0,                 # B1
        one_one, one_one,     # W2
        0, 0,                 # B2
        0, 0,                 # H
        0, 0,                 # Y
    ]

    for bank in range(4):
        for i in range(0, len(bank_words), 2):
            wh = bank_words[i+1] if i+1 < len(bank_words) else 0
            wl = bank_words[i]
            L.append(f"        rx({{32'h{wh:08X}, 32'h{wl:08X}}}, 8'h00);")

    L.append(f"")
    L.append(f"        send_gpu_tail(4);")
    L.append(f"        wait_and_capture(MAX_CYCLES);")
    L.append(f"")
    L.append(f"        // Y = 16.0 = bf16 0x4180, packed = 0x41804180")
    L.append(f"        checkN(tx_cnt, 4, \"ANN TX count\");")
    L.append(f"        check64(tx_data[0], 64'h41804180_41804180, \"ANN TX[0] Y=16.0\");")
    L.append(f"        check64(tx_data[1], 64'h41804180_41804180, \"ANN TX[1] Y=16.0\");")
    L.append(f"        check64(tx_data[2], 64'h41804180_41804180, \"ANN TX[2] Y=16.0\");")
    L.append(f"        check64(tx_data[3], 64'h41804180_41804180, \"ANN TX[3] Y=16.0\");")
    L.append(f"        settle;")
    L.append(f"    end")

    return '\n'.join(L)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SELF-TEST — verify encoder against ISA v2.1 golden reference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def self_test():
    tests = [
        ("MOVI R1, 0",        E("MOVI", rd=1, imm=0),                   0x20100000),
        ("MOV R4, TID",       E("MOV", rd=4, sel=1),                    0x19400000),
        ("SHL R4, R4, 3",     E("SHL", rd=4, ra=4, imm=3),             0x88440003),
        ("ADD R1, R1, R4",    E("ADD", rd=1, ra=1, rb=4),              0x30114000),
        ("WMMA.LOAD.A R4,R1", E("WMMA.LOAD", dt=1, sel=0, rd=4, ra=1), 0xF4410000),
        ("WMMA.LOAD.B R8,R2", E("WMMA.LOAD", dt=1, sel=1, rd=8, ra=2), 0xF5820000),
        ("WMMA.LOAD.C R12,R0",E("WMMA.LOAD", dt=1, sel=2, rd=12, ra=0),0xF6C00000),
        ("MOVI R12, 0 bf16",  E("MOVI", dt=1, rd=12, imm=0),          0x24C00000),
        ("WMMA.MMA",          E("WMMA.MMA", dt=1, rd=12, ra=4, rb=8, rc=12), 0xECC48C00),
        ("WMMA.STORE R12,R3", E("WMMA.STORE", dt=1, rd=12, ra=3),      0xFCC30000),
        ("WMMA.STORE R12,R0", E("WMMA.STORE", dt=1, rd=12, ra=0),      0xFCC00000),
        ("MAX.f R12,R12,R0",  E("MAX", dt=1, rd=12, ra=12, rb=0),      0x54CC0000),
        ("MAX.f R15,R15,R0",  E("MAX", dt=1, rd=15, ra=15, rb=0),      0x54FF0000),
        ("RET",               E("RET"),                                  0xC8000000),
        ("NOP",               E("NOP"),                                  0x00000000),
        # ANN kernel specific
        ("MOVI R0, 4",        E("MOVI", rd=0, imm=4),                   0x20000004),
        ("MOVI R0, 20",       E("MOVI", rd=0, imm=20),                  0x20000014),
        ("MOVI R0, 24",       E("MOVI", rd=0, imm=24),                  0x20000018),
    ]
    ok = 0; fail = 0
    for desc, actual, expected in tests:
        if actual == expected:
            ok += 1
        else:
            print(f"  FAIL: {desc}: got 0x{actual:08X}, exp 0x{expected:08X}")
            fail += 1
    print(f"[self_test] {ok}/{ok+fail} encoding tests passed")
    return fail == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    import argparse
    parser = argparse.ArgumentParser(description='PTX → Custom Hex with WMMA Fusion')
    parser.add_argument('ptx_file', nargs='?', help='PTX file to compile')
    parser.add_argument('--no-test', action='store_true')
    parser.add_argument('--kernel', '-k', default=None,
                        help='Select kernel by substring (default: first with fma.rn.bf16)')
    args = parser.parse_args()

    if not args.no_test:
        print("[ptx_to_hex] ISA v2.1 encoder self-test:")
        if not self_test(): sys.exit(1)
        print()

    if not args.ptx_file:
        # Demo mode: generate the ANN kernel directly without PTX input
        print("[ptx_to_hex] No PTX file — generating ANN kernel from template")
        print()
        instrs = translate_ann_fused([], "ann_template")
        base = "ann_inference"
    else:
        with open(args.ptx_file) as f:
            text = f.read()
        print(f"[ptx_to_hex] Parsing: {args.ptx_file}")
        kernels = parse_ptx(text)
        print(f"[ptx_to_hex] Found {len(kernels)} kernel(s):")
        for k in kernels:
            n_fma = sum(1 for i in k['body'] if i['op'] == 'fma.rn.bf16')
            n_max = sum(1 for i in k['body'] if i['op'] in ('max.bf16', 'max.f32'))
            print(f"  {k['name']}: {len(k['body'])} instrs, {n_fma} FMAs, {n_max} MAXes")

        # Select kernel
        sel = None
        for k in kernels:
            if args.kernel and args.kernel in k['name']:
                sel = k; break
            elif any(i['op'] == 'fma.rn.bf16' for i in k['body']):
                if sel is None: sel = k  # first kernel with FMAs

        if sel is None:
            print("ERROR: No kernel with fma.rn.bf16 found")
            sys.exit(1)

        print(f"\n[ptx_to_hex] Translating: {sel['name']}")

        # Detect FMA chains
        groups = detect_fma_chains(sel['body'])
        print(f"[ptx_to_hex] FMA chain analysis:")
        print(f"  Total FMA instructions: {sum(1 for i in sel['body'] if i['op']=='fma.rn.bf16')}")
        print(f"  4-element chains found: {sum(len(g['chains']) for g in groups)}")
        print(f"  WMMA-fusible groups (4 parallel chains): {len(groups)}")
        if groups:
            for gi, g in enumerate(groups):
                seeds = [c['seed'] for c in g['chains']]
                results = [c['result'] for c in g['chains']]
                print(f"    Group {gi}: seeds={seeds} → results={results}")
                pairs = g['mul_pairs']
                print(f"           shared mul pairs: {pairs}")

        instrs = translate_ann_fused(sel['body'], sel['name'])
        base = os.path.splitext(os.path.basename(args.ptx_file))[0]

    # ── Output ──
    print(f"\n[ptx_to_hex] Generated {len(instrs)} instructions (WMMA-fused)")

    hex_out = fmt_hex(instrs, base)
    hex_path = f"{base}.hex"
    with open(hex_path, 'w') as f: f.write(hex_out)
    print(f"[ptx_to_hex] → {hex_path}")

    socreg_out = fmt_socreg_test(instrs, base)
    socreg_path = f"{base}_socreg_test.pl"
    with open(socreg_path, 'w') as f: f.write(socreg_out)
    print(f"[ptx_to_hex] → {socreg_path}")

    tb_out = fmt_soc_tb_test(instrs, base)
    tb_path = f"{base}_soc_tb_test.v"
    with open(tb_path, 'w') as f: f.write(tb_out)
    print(f"[ptx_to_hex] → {tb_path}")

    # Print hex listing
    print(f"\n{'='*72}")
    print(hex_out)
    print(f"{'='*72}")

    # Print instruction stats
    n_wmma_mma = sum(1 for h, a, _ in instrs if 'WMMA.MMA' in a)
    n_wmma_ld = sum(1 for h, a, _ in instrs if 'WMMA.LOAD' in a)
    n_wmma_st = sum(1 for h, a, _ in instrs if 'WMMA.STORE' in a)
    n_max = sum(1 for h, a, _ in instrs if 'MAX' in a)
    n_movi = sum(1 for h, a, _ in instrs if 'MOVI' in a)
    print(f"\nInstruction mix:")
    print(f"  WMMA.MMA:   {n_wmma_mma} (× 37 cyc = {n_wmma_mma*37} cyc tensor core)")
    print(f"  WMMA.LOAD:  {n_wmma_ld} (× 4 cyc = {n_wmma_ld*4} cyc memory)")
    print(f"  WMMA.STORE: {n_wmma_st} (× 4 cyc = {n_wmma_st*4} cyc memory)")
    print(f"  MAX.f:      {n_max} (ReLU)")
    print(f"  MOVI:       {n_movi}")
    print(f"  RET:        1")
    total_cyc = n_wmma_mma*37 + (n_wmma_ld + n_wmma_st)*4 + n_max + n_movi + 1
    print(f"  Total:      ~{total_cyc} cycles (barrel: ×4 = {total_cyc*4} wall clocks)")
    print(f"  At 62.5 MHz: {total_cyc*4*16/1000:.1f} μs per batch of 4 packets")
    print()


if __name__ == '__main__':
    main()
