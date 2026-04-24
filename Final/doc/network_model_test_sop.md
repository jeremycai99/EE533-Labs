# Network ANN Hardware Test SOP

This procedure drives the existing `pkt_proc` command stream over Ethernet instead of MMIO registers.  The SoC wrapper and CPU/GPU/DMA datapath are unchanged.  The only RTL contract used here is the existing `pkt_proc` network command format:

- EtherType: `0x88B5`
- Ethernet payload begins with a 2-byte zero pad
- 64-bit command words start immediately after that pad
- Command word layout: `cmd[63:60] addr[59:48] count[47:32] param[31:0]`

The raw Ethernet sender controls the Ethernet frame only.  On hardware, the NetFPGA pipeline prepends the module header that `pkt_proc` saves as FIFO word 0; the Ethernet header becomes FIFO words 1 and 2.

The sender script is Python 2.4.3 compatible for the Fedora 14 NetFPGA environment.  Use `python`, not `python3`, on that machine.

If Python is still a blocker, compile the C sender:

```bash
gcc -O2 -Wall -o /tmp/net_model_send Final/src/script/net_model_send.c
```

The C sender uses the same arguments as the Python sender.  It still needs raw-socket privilege.  Without `sudo` or `su`, ask an admin/TA to either run it or install it as a setuid-root lab binary:

```bash
chown root:root /tmp/net_model_send
chmod 4755 /tmp/net_model_send
```

If raw-socket execution is blocked, the C sender can still generate a pcap file as a normal user:

```bash
/tmp/net_model_send --smoke cpu-add --pcap-out /tmp/smoke_cpu_add.pcap
```

Replaying that pcap still requires a privileged tool such as `tcpreplay` or a lab-provided wrapper, but pcap generation itself does not require root.

There is also a Perl pcap-only path that mirrors the older lab `gen_test_pcap.pl` method while using the Final command stream:

```bash
perl Final/src/script/net_model_pcap.pl \
  --in /tmp/ann_classes.frames \
  --out /tmp/ann_classes.pcap \
  --src-mac 00:4e:46:32:43:00 \
  --dst-mac ff:ff:ff:ff:ff:ff
```

Important differences from the old Lab6 pcap generator:

- Use Final `model_test emit_*` output as the payload source.
- Use EtherType `0x88B5`, not the old test EtherType.
- Keep the 2-byte alignment pad before the 64-bit command words.
- Do not reuse the old hardcoded CPU/GPU test packets for Final ANN testing.

## 1. Hardware Setup

1. Program the FPGA bitfile that includes the current `Final/src/rtl/network/pkt_proc.v`.
2. Connect the host Linux Ethernet interface directly to the NetFPGA data port.
3. Identify the host interface.  On newer Linux:

   ```bash
   ip link
   ```

   On the Fedora 14 NetFPGA image, `ip` may not exist.  Use:

   ```bash
   /sbin/ifconfig -a
   cat /proc/net/dev
   ```

4. Bring the interface up and enable promiscuous capture if you have privileges.  Promiscuous mode is recommended because `pkt_proc` replays the received Ethernet header unchanged in responses.

   ```bash
   sudo ip link set dev <iface> up
   sudo ip link set dev <iface> promisc on
   ```

   Fedora 14 fallback:

   ```bash
   sudo /sbin/ifconfig <iface> up
   sudo /sbin/ifconfig <iface> promisc
   ```

## 2. Sanity Check The Network Path

Run a small CPU ADD test through `pkt_proc` before the ANN flow:

```bash
sudo python Final/src/script/net_model_send.py \
  --iface <iface> \
  --smoke cpu-add \
  --dst-mac ff:ff:ff:ff:ff:ff
```

C sender equivalent:

```bash
/tmp/net_model_send \
  --iface <iface> \
  --smoke cpu-add \
  --dst-mac ff:ff:ff:ff:ff:ff
```

Pass criterion:

```text
smoke cpu-add: PASS
```

If this fails, debug the Ethernet path before running ANN: link state, interface selection, bitfile, `pkt_proc` EtherType, and response capture.

If you do not have raw-socket privilege, generate and inspect a pcap instead:

```bash
/tmp/net_model_send --smoke cpu-add --pcap-out /tmp/smoke_cpu_add.pcap
tcpdump -nn -e -xx -r /tmp/smoke_cpu_add.pcap
```

If `tcpreplay` is installed with the required permissions, replay it:

```bash
tcpreplay -i <iface> /tmp/smoke_cpu_add.pcap
```

## 3. Generate ANN Frame Files

From the repository root, generate a preload stream and a class-demo inference stream:

```bash
MODEL_DIR=Final/model SOC_EMIT_PATH=/tmp/ann_load.frames \
  perl Final/src/script/model_test emit_load -q

MODEL_DIR=Final/model SOC_EMIT_PATH=/tmp/ann_classes.frames \
  perl Final/src/script/model_test emit_classes -q
```

The emitted files contain one `FRAME` line per logical `pkt_proc` packet.  They do not contain Ethernet headers; `net_model_send.py` adds the Ethernet header and the required 2-byte pad.

## 4. Preload Weights

Send the generated weight preload frames.  This step writes the packed ANN weights into GPU DMEM and should be done once after FPGA programming.

```bash
sudo python Final/src/script/net_model_send.py \
  --iface <iface> \
  --frames /tmp/ann_load.frames \
  --dst-mac ff:ff:ff:ff:ff:ff \
  --gap 0.02 \
  --cpu-gap 0.25
```

C sender equivalent:

```bash
/tmp/net_model_send \
  --iface <iface> \
  --frames /tmp/ann_load.frames \
  --dst-mac ff:ff:ff:ff:ff:ff \
  --gap 0.02 \
  --cpu-gap 0.25
```

Pass criterion:

- No sender timeout or socket error.
- The script reaches the last frame.

If the board drops frames, increase pacing:

```bash
--gap 0.05 --cpu-gap 0.50
```

Pcap replay equivalent:

```bash
perl Final/src/script/net_model_pcap.pl \
  --in /tmp/ann_load.frames \
  --out /tmp/ann_load.pcap \
  --src-mac 00:4e:46:32:43:00 \
  --dst-mac ff:ff:ff:ff:ff:ff \
  --gap-us 50000 \
  --cpu-gap-us 500000

tcpreplay -i nf2c0 /tmp/ann_load.pcap
```

## 5. Run The Class-Demo Inference

Send the class-demo inference frames and decode the four returned logit packets:

```bash
sudo python Final/src/script/net_model_send.py \
  --iface <iface> \
  --frames /tmp/ann_classes.frames \
  --dst-mac ff:ff:ff:ff:ff:ff \
  --timeout 120 \
  --rx-out /tmp/ann_classes.rx \
  --decode-ann \
  --expected-classes 0,1,2,3
```

C sender equivalent:

```bash
/tmp/net_model_send \
  --iface <iface> \
  --frames /tmp/ann_classes.frames \
  --dst-mac ff:ff:ff:ff:ff:ff \
  --timeout 120 \
  --rx-out /tmp/ann_classes.rx \
  --decode-ann \
  --expected-classes 0,1,2,3
```

Pass criterion:

- One response frame is captured from the final `READBACK + SEND_PKT`.
- The decoder prints four samples.
- Expected class IDs match:
  - `0 = DoS`
  - `1 = NORMAL`
  - `2 = c_ci_na_1`
  - `3 = c_se_na_1`

Pcap replay equivalent:

```bash
perl Final/src/script/net_model_pcap.pl \
  --in /tmp/ann_classes.frames \
  --out /tmp/ann_classes.pcap \
  --src-mac 00:4e:46:32:43:00 \
  --dst-mac ff:ff:ff:ff:ff:ff

tcpdump -i nf2c1 -nn -e -xx -c 1 'ether proto 0x88b5' -w /tmp/ann_classes_resp.pcap &
tcpreplay -i nf2c0 /tmp/ann_classes.pcap
tcpdump -nn -e -xx -r /tmp/ann_classes_resp.pcap
```

## 6. Synthetic Reference Run

For the synthetic reference pattern instead of real class rows:

```bash
MODEL_DIR=Final/model SOC_EMIT_PATH=/tmp/ann_run.frames \
  perl Final/src/script/model_test emit_run -q

sudo python Final/src/script/net_model_send.py \
  --iface <iface> \
  --frames /tmp/ann_run.frames \
  --dst-mac ff:ff:ff:ff:ff:ff \
  --timeout 120 \
  --rx-out /tmp/ann_run.rx \
  --decode-ann
```

The raw response words in `/tmp/ann_run.rx` correspond to the four expected packet-major BF16 outputs documented at the top of `model_test`.

## 7. One-File End-To-End Option

To emit preload and class-demo inference into one file:

```bash
MODEL_DIR=Final/model SOC_EMIT_PATH=/tmp/ann_model_classes.frames \
  perl Final/src/script/model_test emit_model_classes -q

sudo python Final/src/script/net_model_send.py \
  --iface <iface> \
  --frames /tmp/ann_model_classes.frames \
  --dst-mac ff:ff:ff:ff:ff:ff \
  --timeout 120 \
  --rx-out /tmp/ann_model_classes.rx \
  --decode-ann \
  --expected-classes 0,1,2,3
```

This is slower but convenient immediately after programming the FPGA.

## Troubleshooting

- `permission error`: run the sender with `sudo`.
- `sudo` not allowed: use `--pcap-out` to generate pcap files, then ask whether `tcpreplay` or another lab packet replay wrapper is installed with permissions.
- `TIMEOUT` on smoke test: verify link, interface, bitfile, and EtherType `0x88B5`.
- TX completes but no ANN response: increase `--timeout`; the final scheduler run can be long on hardware.
- Preload appears unreliable: increase `--gap` and `--cpu-gap`.
- Need a fallback status check during bring-up: the MMIO commands still work:

  ```bash
  perl Final/src/script/model_test status
  perl Final/src/script/model_test reset
  ```
