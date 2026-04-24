#!/usr/bin/env python
"""
Send model_test-emitted SoC command frames over raw Ethernet.

This script is written for the old NetFPGA Fedora 14 environment, including
Python 2.4.3.  It avoids argparse, f-strings, dataclasses, annotations,
conditional expressions, pathlib, and other newer Python features.

Frame format:
  - Input file lines are produced by `model_test emit_*`.
  - This script wraps each command stream as:
      dst_mac | src_mac | EtherType 0x88B5 | 2-byte pad | 64-bit command words
  - On hardware the NetFPGA pipeline supplies FIFO word 0 as a module header.
    The raw Ethernet bytes become FIFO words 1 and 2.  The 2-byte pad is
    required because pkt_proc drops the first three 64-bit FIFO words before
    command decode.  FIFO word 2 is:
      src_mac[31:0] | ethertype[15:0] | pad[15:0]
"""

import optparse
import os
import select
import socket
import struct
import sys
import time


ETHERTYPE_SOC = 0x88B5
CLASS_NAMES = ["DoS", "NORMAL", "c_ci_na_1", "c_se_na_1"]


def out(text):
    sys.stdout.write(str(text) + "\n")


def err(text):
    sys.stderr.write(str(text) + "\n")


def bstr(text):
    # Python 2: str is already bytes.  Python 3 fallback: encode as latin-1.
    if sys.version_info[0] >= 3 and isinstance(text, str):
        return text.encode("latin-1")
    return text


class Frame(object):
    def __init__(self, index, expect, cpu, rb_words, words):
        self.index = index
        self.expect = expect
        self.cpu = cpu
        self.rb_words = rb_words
        self.words = words


def parse_mac(text):
    parts = text.split(":")
    if len(parts) != 6:
        raise ValueError("bad MAC address: %s" % text)
    vals = []
    for p in parts:
        vals.append(int(p, 16))
    return struct.pack("!BBBBBB", vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def iface_mac(iface):
    path = "/sys/class/net/%s/address" % iface
    if not os.path.exists(path):
        raise IOError("cannot read interface MAC at %s" % path)
    fh = open(path, "r")
    try:
        return fh.read().strip()
    finally:
        fh.close()


def parse_frame_file(path):
    frames = []
    fh = open(path, "r")
    try:
        lineno = 0
        for raw in fh:
            lineno += 1
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 3 or toks[0] != "FRAME":
                raise ValueError("%s:%d: expected FRAME line" % (path, lineno))
            idx = int(toks[1], 0)
            meta = {}
            pos = 2
            while pos < len(toks) and toks[pos].find("=") >= 0:
                key, val = toks[pos].split("=", 1)
                meta[key] = val
                pos += 1
            words = []
            for w in toks[pos:]:
                words.append(int(w, 16))
            declared = int(meta.get("words", str(len(words))), 0)
            if declared != len(words):
                raise ValueError(
                    "%s:%d: metadata says words=%d, got %d"
                    % (path, lineno, declared, len(words))
                )
            frames.append(
                Frame(
                    idx,
                    meta.get("expect", "0") == "1",
                    meta.get("cpu", "0") == "1",
                    int(meta.get("rb", "0"), 0),
                    words,
                )
            )
    finally:
        fh.close()
    return frames


def smoke_cpu_add():
    words = [
        0x1000000300000000,  # LOAD_IMEM addr=0 count=3
        0xE5901000E3A00000,
        0xE0813002E5902004,
        0xEAFFFFFEE5803008,
        0x2000000200000000,  # LOAD_DMEM addr=0 count=2
        0x000000140000000A,
        0x0000000000000000,
        0x3000000000000000,  # CPU_START
        0x4000000200000000,  # READBACK addr=0 count=2
        0x5000000000000000,  # SEND_PKT
    ]
    return [Frame(0, True, True, 2, words)]


def pack_u64(word):
    hi = (word >> 32) & 0xFFFFFFFF
    lo = word & 0xFFFFFFFF
    return struct.pack("!II", hi, lo)


def build_ether_frame(frame, dst_mac, src_mac, ethertype):
    payload_words = []
    for w in frame.words:
        payload_words.append(pack_u64(w))
    payload = bstr("\x00\x00") + bstr("").join(payload_words)
    pkt = dst_mac + src_mac + struct.pack("!H", ethertype) + payload
    if len(pkt) < 64:
        pkt += bstr("\x00") * (64 - len(pkt))
    return pkt


def parse_rx_words(pkt, rb_words):
    if len(pkt) < 16:
        return []
    payload = pkt[16:]  # Ethernet header (14) + pkt_proc alignment pad (2)
    usable = len(payload) - (len(payload) % 8)
    words = []
    i = 0
    while i < usable:
        hi, lo = struct.unpack("!II", payload[i : i + 8])
        words.append((hi << 32) | lo)
        i += 8
    if rb_words > 0:
        words = words[:rb_words]
    return words


def open_raw_socket(iface):
    if not hasattr(socket, "AF_PACKET"):
        raise RuntimeError("raw Ethernet send requires Linux AF_PACKET sockets")
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
    sock.bind((iface, 0))
    return sock


def wait_for_response(sock, timeout_s, rb_words, ethertype):
    deadline = time.time() + timeout_s
    outgoing_type = getattr(socket, "PACKET_OUTGOING", 4)
    wanted_type = struct.pack("!H", ethertype)
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 0.0:
            remaining = 0.0
        ready, unused_w, unused_x = select.select([sock], [], [], remaining)
        if not ready:
            break
        pkt, addr = sock.recvfrom(65535)
        if len(addr) > 2 and addr[2] == outgoing_type:
            continue
        if len(pkt) < 16:
            continue
        if pkt[12:14] != wanted_type:
            continue
        words = parse_rx_words(pkt, rb_words)
        if words:
            return words
    return None


def bf16_to_float(bits):
    word = (bits & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", word))[0]


def decode_logit_word(word):
    hi = (word >> 32) & 0xFFFFFFFF
    lo = word & 0xFFFFFFFF
    vals = [lo & 0xFFFF, (lo >> 16) & 0xFFFF, hi & 0xFFFF, (hi >> 16) & 0xFFFF]
    floats = []
    for v in vals:
        floats.append(bf16_to_float(v))
    return vals, floats


def argmax4(vals):
    best = 0
    for i in range(1, 4):
        if vals[i] > vals[best]:
            best = i
    return best


def print_ann_decode(rx_records, expected):
    for rec in rx_records:
        frame_idx, words = rec
        if len(words) >= 4:
            out("")
            out("ANN decode from response to frame %d:" % frame_idx)
            correct = 0
            for sample in range(0, 4):
                hx, fl = decode_logit_word(words[sample])
                pred = argmax4(fl)
                exp_txt = ""
                if expected and sample < len(expected):
                    ok = pred == expected[sample]
                    if ok:
                        correct += 1
                        verdict = "PASS"
                    else:
                        verdict = "FAIL"
                    exp_txt = " expected=%d:%s %s" % (
                        expected[sample],
                        CLASS_NAMES[expected[sample]],
                        verdict,
                    )
                out(
                    "  sample %d: pred=%d:%-10s "
                    "logits=%04X/%+7.2f %04X/%+7.2f "
                    "%04X/%+7.2f %04X/%+7.2f%s"
                    % (
                        sample,
                        pred,
                        CLASS_NAMES[pred],
                        hx[0],
                        fl[0],
                        hx[1],
                        fl[1],
                        hx[2],
                        fl[2],
                        hx[3],
                        fl[3],
                        exp_txt,
                    )
                )
            if expected:
                total = min(4, len(expected))
                out("  accuracy: %d/%d" % (correct, total))
            return
    out("")
    out("ANN decode skipped: no response with at least 4 readback words was captured")


def parse_expected_classes(text):
    vals = []
    if not text:
        return None
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(int(part, 0))
    return vals


def build_parser():
    usage = "%prog (--frames FILE | --smoke cpu-add) [options]"
    parser = optparse.OptionParser(usage=usage, description=__doc__)
    parser.add_option("--iface", dest="iface", help="Linux Ethernet interface connected to NetFPGA")
    parser.add_option("--frames", dest="frames", help="frame file from model_test emit_*")
    parser.add_option("--smoke", dest="smoke", help="built-in smoke test: cpu-add")
    parser.add_option("--dst-mac", dest="dst_mac", default="ff:ff:ff:ff:ff:ff", help="destination MAC")
    parser.add_option("--src-mac", dest="src_mac", help="source MAC; default reads interface address")
    parser.add_option("--ethertype", dest="ethertype", default="0x%04X" % ETHERTYPE_SOC, help="SoC EtherType")
    parser.add_option("--gap", dest="gap", type="float", default=0.02, help="delay after non-response frames")
    parser.add_option("--cpu-gap", dest="cpu_gap", type="float", default=0.25, help="delay after CPU frames without response")
    parser.add_option("--timeout", dest="timeout", type="float", default=10.0, help="response timeout per expect=1 frame")
    parser.add_option("--rx-out", dest="rx_out", help="write captured RX words")
    parser.add_option("--dry-run", dest="dry_run", action="store_true", default=False, help="parse and print without sending")
    parser.add_option("--decode-ann", dest="decode_ann", action="store_true", default=False, help="decode first 4-word ANN response")
    parser.add_option("--expected-classes", dest="expected_classes", help="comma-separated expected ANN class IDs")
    return parser


def main():
    parser = build_parser()
    opts, args = parser.parse_args()
    if args:
        parser.error("unexpected positional arguments")
    if bool(opts.frames) == bool(opts.smoke):
        parser.error("provide exactly one of --frames or --smoke cpu-add")
    if opts.smoke and opts.smoke != "cpu-add":
        parser.error("--smoke only supports cpu-add")
    if not opts.dry_run and not opts.iface:
        parser.error("--iface is required unless --dry-run is used")

    if opts.frames:
        frames = parse_frame_file(opts.frames)
    else:
        frames = smoke_cpu_add()

    ethertype = int(opts.ethertype, 0)
    dst = parse_mac(opts.dst_mac)
    if opts.src_mac:
        src_text = opts.src_mac
    elif opts.iface:
        src_text = iface_mac(opts.iface)
    else:
        src_text = "02:00:00:00:00:01"
    src = parse_mac(src_text)

    total_bytes = 0
    expect_count = 0
    for f in frames:
        total_bytes += len(build_ether_frame(f, dst, src, ethertype))
        if f.expect:
            expect_count += 1
    out(
        "frames=%d expect_responses=%d bytes=%d dst=%s src=%s ethertype=0x%04X"
        % (len(frames), expect_count, total_bytes, opts.dst_mac, src_text, ethertype)
    )

    if opts.dry_run:
        show = frames[:5]
        for f in show:
            out(
                "  frame %d: words=%d expect=%d cpu=%d rb=%d"
                % (f.index, len(f.words), int(f.expect), int(f.cpu), f.rb_words)
            )
        if len(frames) > 5:
            out("  ... %d more frames" % (len(frames) - 5))
        return 0

    rx_records = []
    sock = open_raw_socket(opts.iface)
    try:
        rx_fh = None
        if opts.rx_out:
            rx_fh = open(opts.rx_out, "w")
        try:
            for f in frames:
                pkt = build_ether_frame(f, dst, src, ethertype)
                sock.send(pkt)
                out(
                    "[TX frame=%04d] bytes=%d words=%d expect=%d"
                    % (f.index, len(pkt), len(f.words), int(f.expect))
                )
                if f.expect:
                    words = wait_for_response(sock, opts.timeout, f.rb_words, ethertype)
                    if words is None:
                        out("[RX frame=%04d] TIMEOUT after %.1fs" % (f.index, opts.timeout))
                        return 2
                    parts = []
                    for w in words:
                        parts.append("%016X" % w)
                    word_text = " ".join(parts)
                    out("[RX frame=%04d] %s" % (f.index, word_text))
                    rx_records.append((f.index, words))
                    if rx_fh:
                        rx_fh.write("RX %d %s\n" % (f.index, word_text))
                else:
                    if f.cpu:
                        time.sleep(opts.cpu_gap)
                    else:
                        time.sleep(opts.gap)
        finally:
            if rx_fh:
                rx_fh.close()
    finally:
        sock.close()

    if opts.smoke == "cpu-add" and rx_records:
        got = rx_records[0][1]
        exp = [0x000000140000000A, 0x000000000000001E]
        if got[:2] == exp:
            out("smoke cpu-add: PASS")
        else:
            out("smoke cpu-add: FAIL")

    expected = parse_expected_classes(opts.expected_classes)
    if opts.decode_ann:
        print_ann_decode(rx_records, expected)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (socket.error, OSError):
        exc = sys.exc_info()[1]
        err("socket/os error: %s" % exc)
        err("raw Ethernet usually requires root; run with sudo")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
