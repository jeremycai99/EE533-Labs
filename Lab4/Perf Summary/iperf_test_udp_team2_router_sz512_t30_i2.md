# Network Bandwidth Test Results (UDP)

## Test Information
* Date: Sat Feb  7 16:18:29 PST 2026
* Team Number: 2
* Bitfile Type: router
* Protocol: UDP
* Packet Size: 512 bytes
* Test Duration: 30 seconds
* Test Interval: 2 seconds
* Target Bandwidth: 1G
* RKD Enabled: Yes

### Node Mapping
* NetFPGA Router: nf2
* Control plane: SSH via hostnames (management network)
* Data plane: iperf/ping via direct IPs (through NetFPGA)
* Node Ports:
  - nf3 (Port 0): SSH=nf3.usc.edu | Data=10.0.8.3:5001
  - nf4 (Port 1): SSH=nf4.usc.edu | Data=10.0.9.3:5002
  - nf0 (Port 2): SSH=nf0.usc.edu | Data=10.0.10.3:5003
  - nf1 (Port 3): SSH=nf1.usc.edu | Data=10.0.11.3:5004

## Test Results
Found net device: nf2c0
Bit file built from: nf2_top_par.ncd;HW_TIMEOUT=FALSE
Part: 2vp50ff1152
Date: 2011/11/17
Time: 17:49:43
Error Registers: 0
Good, after resetting programming interface the FIFO is empty
Download completed -  2377668 bytes. (expected 2377668).
DONE went high - chip has been successfully programmed.
CPCI Information
----------------
Version: 4 (rev 1)

Device (Virtex) Information
---------------------------
Project directory: reference_router
Project name: Reference router
Project description: Reference IPv4 router

Device ID: 2
Version: 1.0.0
Built against CPCI version: 4 (rev 1)

Virtex design compiled against active CPCI version

### RKD Status
* RKD confirmed running on nf2

#### RKD Log (last 20 lines)
```
```


---

## Phase 1: Latency & Cross-Port Connectivity Verification

Testing RTT latency and connectivity across all port pairs via data plane IPs.

Router mode: Adjacent AND cross-port pairs should succeed.

### Latency Matrix (RTT in ms)

| Source           | Destination  | Target Address       | Pair Type  | Status   | Min RTT    | Avg RTT    | Max RTT    | Loss %   |
|------------------|--------------|----------------------|------------|----------|------------|------------|------------|----------|
| nf3              | nf4          | 10.0.9.3             | adjacent   | OK       | 0.311ms    | 1.115ms    | 4.037ms    | 0%       |
| nf3              | nf0          | 10.0.10.3            | cross-port | OK       | 0.235ms    | 0.650ms    | 2.088ms    | 0%       |
| nf3              | nf1          | 10.0.11.3            | cross-port | OK       | 0.235ms    | 0.758ms    | 2.735ms    | 0%       |
| nf4              | nf3          | 10.0.8.3             | adjacent   | OK       | 0.353ms    | 0.456ms    | 0.623ms    | 0%       |
| nf4              | nf0          | 10.0.10.3            | cross-port | OK       | 0.182ms    | 0.397ms    | 0.852ms    | 0%       |
| nf4              | nf1          | 10.0.11.3            | cross-port | OK       | 0.327ms    | 0.395ms    | 0.473ms    | 0%       |
| nf0              | nf3          | 10.0.8.3             | cross-port | OK       | 0.454ms    | 0.590ms    | 0.726ms    | 0%       |
| nf0              | nf4          | 10.0.9.3             | cross-port | OK       | 0.376ms    | 0.449ms    | 0.533ms    | 0%       |
| nf0              | nf1          | 10.0.11.3            | adjacent   | OK       | 0.347ms    | 0.492ms    | 0.644ms    | 0%       |
| nf1              | nf3          | 10.0.8.3             | cross-port | OK       | 0.410ms    | 0.464ms    | 0.533ms    | 0%       |
| nf1              | nf4          | 10.0.9.3             | cross-port | OK       | 0.351ms    | 0.409ms    | 0.472ms    | 0%       |
| nf1              | nf0          | 10.0.10.3            | adjacent   | OK       | 0.398ms    | 0.600ms    | 0.861ms    | 0%       |

### Connectivity Summary

* **Adjacent port pairs** (NIC + Router): 4 passed, 0 failed
* **Cross port pairs** (Router only):     8 passed, 0 failed

### Latency Summary

* Average RTT (adjacent pairs): .665 ms
* Average RTT (cross-port pairs): .514 ms


---

## Phase 2: Aggregate Throughput Test (Unidirectional)

Testing 2 simultaneous non-overlapping flows to measure aggregate capacity.
* Flow 1: nf3 → nf4 (10.0.8.3 → 10.0.9.3)
* Flow 2: nf0 → nf1 (10.0.10.3 → 10.0.11.3)


### Test: nf3 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 50456 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  2.0- 4.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  4.0- 6.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  6.0- 8.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  8.0-10.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 10.0-12.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 12.0-14.0 sec  59.8 MBytes   251 Mbits/sec
[  3] 14.0-16.0 sec  59.1 MBytes   248 Mbits/sec
[  3] 16.0-18.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 18.0-20.0 sec  57.8 MBytes   242 Mbits/sec
[  3] 20.0-22.0 sec  57.4 MBytes   241 Mbits/sec
[  3] 22.0-24.0 sec  56.6 MBytes   237 Mbits/sec
[  3] 24.0-26.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 26.0-28.0 sec  60.5 MBytes   254 Mbits/sec
[  3] 28.0-30.0 sec  59.8 MBytes   251 Mbits/sec
[  3]  0.0-30.0 sec   883 MBytes   247 Mbits/sec
[  3] Sent 1808595 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   187 MBytes  52.0 Mbits/sec  14.221 ms 1425448/1808569 (79%)
[  3]  0.0-30.2 sec  354 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 52.0 Mbits/sec | Loss: 79% | Jitter: 14.221 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 36102 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  2.0- 4.0 sec  59.0 MBytes   247 Mbits/sec
[  3]  4.0- 6.0 sec  58.4 MBytes   245 Mbits/sec
[  3]  6.0- 8.0 sec  57.8 MBytes   243 Mbits/sec
[  3]  8.0-10.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 10.0-12.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 12.0-14.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 14.0-16.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 16.0-18.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 18.0-20.0 sec  59.4 MBytes   249 Mbits/sec
[  3] 20.0-22.0 sec  58.8 MBytes   247 Mbits/sec
[  3] 22.0-24.0 sec  58.1 MBytes   244 Mbits/sec
[  3] 24.0-26.0 sec  57.5 MBytes   241 Mbits/sec
[  3] 26.0-28.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 28.0-30.0 sec  59.0 MBytes   247 Mbits/sec
[  3]  0.0-30.0 sec   881 MBytes   246 Mbits/sec
[  3] Sent 1803636 datagrams
[  3] Server Report:
[  3]  0.0-29.9 sec   166 MBytes  46.6 Mbits/sec   0.576 ms 1462830/1803635 (81%)
[  3]  0.0-29.9 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 46.6 Mbits/sec | Loss: 81% | Jitter: 0.576 ms


---

## Phase 3: Aggregate Throughput Test (Bidirectional)

Testing 4 simultaneous bidirectional flows to measure maximum aggregate capacity.
* Flow 1: nf3 ↔ nf4 (10.0.8.3 ↔ 10.0.9.3)
* Flow 2: nf0 ↔ nf1 (10.0.10.3 ↔ 10.0.11.3)


### Test: nf3 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 48993 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  60.4 MBytes   253 Mbits/sec
[  3]  2.0- 4.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  4.0- 6.0 sec  59.0 MBytes   247 Mbits/sec
[  3]  6.0- 8.0 sec  58.5 MBytes   245 Mbits/sec
[  3]  8.0-10.0 sec  57.7 MBytes   242 Mbits/sec
[  3] 10.0-12.0 sec  57.0 MBytes   239 Mbits/sec
[  3] 12.0-14.0 sec  56.6 MBytes   237 Mbits/sec
[  3] 14.0-16.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 16.0-18.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 18.0-20.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 20.0-22.0 sec  59.1 MBytes   248 Mbits/sec
[  3] 22.0-24.0 sec  58.5 MBytes   245 Mbits/sec
[  3] 24.0-26.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 26.0-28.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  0.0-30.0 sec   880 MBytes   246 Mbits/sec
[  3] Sent 1801940 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   209 MBytes  58.0 Mbits/sec   7.240 ms 1374648/1801897 (76%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 58.0 Mbits/sec | Loss: 76% | Jitter: 7.240 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 55034 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  2.0- 4.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  4.0- 6.0 sec  61.9 MBytes   260 Mbits/sec
[  3]  6.0- 8.0 sec  61.5 MBytes   258 Mbits/sec
[  3]  8.0-10.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 10.0-12.0 sec  56.7 MBytes   238 Mbits/sec
[  3] 12.0-14.0 sec  59.2 MBytes   248 Mbits/sec
[  3] 14.0-16.0 sec  62.0 MBytes   260 Mbits/sec
[  3] 16.0-18.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 18.0-20.0 sec  56.9 MBytes   238 Mbits/sec
[  3] 20.0-22.0 sec  59.2 MBytes   248 Mbits/sec
[  3] 22.0-24.0 sec  61.9 MBytes   260 Mbits/sec
[  3] 24.0-26.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 26.0-28.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  0.0-30.0 sec   883 MBytes   247 Mbits/sec
[  3] Sent 1808698 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   125 MBytes  34.9 Mbits/sec   0.106 ms 1551905/1808697 (86%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 34.9 Mbits/sec | Loss: 86% | Jitter: 0.106 ms


### Test: nf4 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 33998 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  2.0- 4.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  4.0- 6.0 sec  60.2 MBytes   253 Mbits/sec
[  3]  6.0- 8.0 sec  60.4 MBytes   254 Mbits/sec
[  3]  8.0-10.0 sec  59.4 MBytes   249 Mbits/sec
[  3] 10.0-12.0 sec  58.5 MBytes   245 Mbits/sec
[  3] 12.0-14.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 14.0-16.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 16.0-18.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 18.0-20.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 20.0-22.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 22.0-24.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 24.0-26.0 sec  58.0 MBytes   243 Mbits/sec
[  3] 26.0-28.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 28.0-30.0 sec  60.2 MBytes   252 Mbits/sec
[  3]  0.0-30.0 sec   886 MBytes   248 Mbits/sec
[  3] Sent 1815220 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   147 MBytes  41.0 Mbits/sec   0.106 ms 1514342/1815219 (83%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 41.0 Mbits/sec | Loss: 83% | Jitter: 0.106 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 56934 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  2.0- 4.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  4.0- 6.0 sec  57.6 MBytes   241 Mbits/sec
[  3]  6.0- 8.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  8.0-10.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 10.0-12.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 12.0-14.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 14.0-16.0 sec  59.4 MBytes   249 Mbits/sec
[  3] 16.0-18.0 sec  58.9 MBytes   247 Mbits/sec
[  3] 18.0-20.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 20.0-22.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 22.0-24.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 24.0-26.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 26.0-28.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  0.0-30.0 sec   870 MBytes   243 Mbits/sec
[  3] Sent 1782618 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   192 MBytes  53.5 Mbits/sec   0.039 ms 1390179/1782617 (78%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 53.5 Mbits/sec | Loss: 78% | Jitter: 0.039 ms


---

## Phase 4: Individual Flow Baseline Tests

Testing each flow individually (sequentially) to establish baseline performance.


### Test: nf4 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 40415 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  2.0- 4.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  4.0- 6.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  6.0- 8.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  8.0-10.0 sec  60.5 MBytes   254 Mbits/sec
[  3] 10.0-12.0 sec  59.5 MBytes   250 Mbits/sec
[  3] 12.0-14.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 14.0-16.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 16.0-18.0 sec  58.4 MBytes   245 Mbits/sec
[  3] 18.0-20.0 sec  60.8 MBytes   255 Mbits/sec
[  3] 20.0-22.0 sec  59.7 MBytes   250 Mbits/sec
[  3] 22.0-24.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 24.0-26.0 sec  57.8 MBytes   242 Mbits/sec
[  3] 26.0-28.0 sec  57.1 MBytes   239 Mbits/sec
[  3] 28.0-30.0 sec  61.0 MBytes   256 Mbits/sec
[  3]  0.0-30.0 sec   884 MBytes   247 Mbits/sec
[  3] Sent 1810616 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   362 MBytes   101 Mbits/sec   0.586 ms 1069471/1810615 (59%)
[  3]  0.0-30.0 sec  385 datagrams received out-of-order
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 101 Mbits/sec | Loss: 59% | Jitter: 0.586 ms


### Test: nf0 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 53576 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  2.0- 4.0 sec  61.0 MBytes   256 Mbits/sec
[  3]  4.0- 6.0 sec  60.7 MBytes   254 Mbits/sec
[  3]  6.0- 8.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  8.0-10.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 10.0-12.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 12.0-14.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 14.0-16.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 16.0-18.0 sec  57.0 MBytes   239 Mbits/sec
[  3] 18.0-20.0 sec  60.2 MBytes   253 Mbits/sec
[  3] 20.0-22.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 22.0-24.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 24.0-26.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 26.0-28.0 sec  59.2 MBytes   248 Mbits/sec
[  3]  0.0-30.0 sec   889 MBytes   249 Mbits/sec
[  3] Sent 1821230 datagrams
[  3] Server Report:
[  3]  0.0-29.4 sec   182 MBytes  51.8 Mbits/sec  12.816 ms 1449354/1821216 (80%)
[  3]  0.0-29.4 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf3 (10.0.8.3:5001) | BW: 51.8 Mbits/sec | Loss: 80% | Jitter: 12.816 ms


### Test: nf1 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 48540 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  60.7 MBytes   254 Mbits/sec
[  3]  2.0- 4.0 sec  60.0 MBytes   252 Mbits/sec
[  3]  4.0- 6.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  6.0- 8.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  8.0-10.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 10.0-12.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 12.0-14.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 14.0-16.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 16.0-18.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 18.0-20.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 20.0-22.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 22.0-24.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 24.0-26.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 26.0-28.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 28.0-30.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  0.0-30.0 sec   886 MBytes   248 Mbits/sec
[  3] Sent 1815502 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   317 MBytes  88.8 Mbits/sec   0.606 ms 1165627/1815501 (64%)
[  3]  0.0-30.0 sec  6 datagrams received out-of-order
```
**Completed:** nf1 → nf3 (10.0.8.3:5001) | BW: 88.8 Mbits/sec | Loss: 64% | Jitter: 0.606 ms


### Test: nf3 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 33859 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  2.0- 4.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  4.0- 6.0 sec  60.7 MBytes   254 Mbits/sec
[  3]  6.0- 8.0 sec  59.8 MBytes   251 Mbits/sec
[  3]  8.0-10.0 sec  59.1 MBytes   248 Mbits/sec
[  3] 10.0-12.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 12.0-14.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 14.0-16.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 16.0-18.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 18.0-20.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 20.0-22.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 22.0-24.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 24.0-26.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 26.0-28.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 28.0-30.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  0.0-30.0 sec   885 MBytes   248 Mbits/sec
[  3] Sent 1813408 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   187 MBytes  52.3 Mbits/sec   0.627 ms 1430454/1813407 (79%)
[  3]  0.0-30.0 sec  40 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 52.3 Mbits/sec | Loss: 79% | Jitter: 0.627 ms


### Test: nf0 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 33239 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  2.0- 4.0 sec  61.3 MBytes   257 Mbits/sec
[  3]  4.0- 6.0 sec  60.7 MBytes   255 Mbits/sec
[  3]  6.0- 8.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  8.0-10.0 sec  59.5 MBytes   250 Mbits/sec
[  3] 10.0-12.0 sec  59.1 MBytes   248 Mbits/sec
[  3] 12.0-14.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 14.0-16.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 16.0-18.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 18.0-20.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 20.0-22.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 22.0-24.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 24.0-26.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 26.0-28.0 sec  59.4 MBytes   249 Mbits/sec
[  3] 28.0-30.0 sec  58.8 MBytes   247 Mbits/sec
[  3]  0.0-30.0 sec   889 MBytes   249 Mbits/sec
[  3] Sent 1821086 datagrams
[  3] Server Report:
[  3]  0.0-29.9 sec   412 MBytes   116 Mbits/sec   0.634 ms 977135/1821085 (54%)
[  3]  0.0-29.9 sec  4 datagrams received out-of-order
```
**Completed:** nf0 → nf4 (10.0.9.3:5002) | BW: 116 Mbits/sec | Loss: 54% | Jitter: 0.634 ms


### Test: nf1 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 49549 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  56.6 MBytes   237 Mbits/sec
[  3]  2.0- 4.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  4.0- 6.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  6.0- 8.0 sec  60.2 MBytes   252 Mbits/sec
[  3]  8.0-10.0 sec  59.1 MBytes   248 Mbits/sec
[  3] 10.0-12.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 12.0-14.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 14.0-16.0 sec  57.8 MBytes   242 Mbits/sec
[  3] 16.0-18.0 sec  57.3 MBytes   241 Mbits/sec
[  3] 18.0-20.0 sec  56.6 MBytes   238 Mbits/sec
[  3] 20.0-22.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 22.0-24.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 24.0-26.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 26.0-28.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 28.0-30.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  0.0-30.0 sec   885 MBytes   247 Mbits/sec
[  3] Sent 1812012 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   203 MBytes  56.8 Mbits/sec   0.519 ms 1395789/1812011 (77%)
[  3]  0.0-30.0 sec  4 datagrams received out-of-order
```
**Completed:** nf1 → nf4 (10.0.9.3:5002) | BW: 56.8 Mbits/sec | Loss: 77% | Jitter: 0.519 ms


### Test: nf3 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 52888 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  60.7 MBytes   254 Mbits/sec
[  3]  2.0- 4.0 sec  60.0 MBytes   252 Mbits/sec
[  3]  4.0- 6.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  6.0- 8.0 sec  58.9 MBytes   247 Mbits/sec
[  3]  8.0-10.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 10.0-12.0 sec  57.4 MBytes   241 Mbits/sec
[  3] 12.0-14.0 sec  56.7 MBytes   238 Mbits/sec
[  3] 14.0-16.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 16.0-18.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 18.0-20.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 20.0-22.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 22.0-24.0 sec  58.7 MBytes   246 Mbits/sec
[  3] 24.0-26.0 sec  58.0 MBytes   243 Mbits/sec
[  3] 26.0-28.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  0.0-30.0 sec   884 MBytes   247 Mbits/sec
[  3] Sent 1810366 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   411 MBytes   115 Mbits/sec  14.186 ms 968393/1810352 (53%)
[  3]  0.0-30.0 sec  147 datagrams received out-of-order
```
**Completed:** nf3 → nf0 (10.0.10.3:5003) | BW: 115 Mbits/sec | Loss: 53% | Jitter: 14.186 ms


### Test: nf4 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 41685 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  60.7 MBytes   254 Mbits/sec
[  3]  2.0- 4.0 sec  59.4 MBytes   249 Mbits/sec
[  3]  4.0- 6.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  6.0- 8.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  8.0-10.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 10.0-12.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 12.0-14.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 14.0-16.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 16.0-18.0 sec  57.7 MBytes   242 Mbits/sec
[  3] 18.0-20.0 sec  57.5 MBytes   241 Mbits/sec
[  3] 20.0-22.0 sec  61.2 MBytes   257 Mbits/sec
[  3] 22.0-24.0 sec  60.1 MBytes   252 Mbits/sec
[  3] 24.0-26.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 26.0-28.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 28.0-30.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  0.0-30.0 sec   885 MBytes   247 Mbits/sec
[  3] Sent 1812710 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   317 MBytes  88.6 Mbits/sec   0.684 ms 1162812/1812709 (64%)
[  3]  0.0-30.0 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf0 (10.0.10.3:5003) | BW: 88.6 Mbits/sec | Loss: 64% | Jitter: 0.684 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 53142 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  2.0- 4.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  4.0- 6.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  6.0- 8.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  8.0-10.0 sec  59.9 MBytes   251 Mbits/sec
[  3] 10.0-12.0 sec  60.8 MBytes   255 Mbits/sec
[  3] 12.0-14.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 14.0-16.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 16.0-18.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 18.0-20.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 20.0-22.0 sec  58.0 MBytes   243 Mbits/sec
[  3] 22.0-24.0 sec  57.5 MBytes   241 Mbits/sec
[  3] 24.0-26.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 26.0-28.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 28.0-30.0 sec  61.0 MBytes   256 Mbits/sec
[  3]  0.0-30.0 sec   883 MBytes   247 Mbits/sec
[  3] Sent 1807824 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   276 MBytes  76.6 Mbits/sec  12.206 ms 1242639/1807822 (69%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 76.6 Mbits/sec | Loss: 69% | Jitter: 12.206 ms


### Test: nf3 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 51548 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  7.28 MBytes  30.6 Mbits/sec
[  3]  2.0- 4.0 sec  7.35 MBytes  30.8 Mbits/sec
[  3]  4.0- 6.0 sec  7.37 MBytes  30.9 Mbits/sec
[  3]  6.0- 8.0 sec  8.00 MBytes  33.6 Mbits/sec
[  3]  8.0-10.0 sec  7.40 MBytes  31.0 Mbits/sec
[  3] 10.0-12.0 sec  7.20 MBytes  30.2 Mbits/sec
[  3] 12.0-14.0 sec  7.23 MBytes  30.3 Mbits/sec
[  3] 14.0-16.0 sec  7.30 MBytes  30.6 Mbits/sec
[  3] 16.0-18.0 sec  7.35 MBytes  30.8 Mbits/sec
[  3] 18.0-20.0 sec  7.36 MBytes  30.9 Mbits/sec
[  3] 20.0-22.0 sec  7.38 MBytes  31.0 Mbits/sec
[  3] 22.0-24.0 sec  7.39 MBytes  31.0 Mbits/sec
[  3] 24.0-26.0 sec  7.38 MBytes  30.9 Mbits/sec
[  3] 26.0-28.0 sec  7.19 MBytes  30.2 Mbits/sec
[  3] 28.0-30.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  0.0-30.0 sec   160 MBytes  44.9 Mbits/sec
[  3] Sent 328585 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   123 MBytes  34.5 Mbits/sec   0.056 ms 76065/328584 (23%)
[  3]  0.0-30.0 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf1 (10.0.11.3:5004) | BW: 34.5 Mbits/sec | Loss: 23% | Jitter: 0.056 ms


### Test: nf4 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 52765 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  2.0- 4.0 sec  60.5 MBytes   254 Mbits/sec
[  3]  4.0- 6.0 sec  59.8 MBytes   251 Mbits/sec
[  3]  6.0- 8.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  8.0-10.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 10.0-12.0 sec  58.3 MBytes   245 Mbits/sec
[  3] 12.0-14.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 14.0-16.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 16.0-18.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 18.0-20.0 sec  58.0 MBytes   243 Mbits/sec
[  3] 20.0-22.0 sec  57.2 MBytes   240 Mbits/sec
[  3] 22.0-24.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 24.0-26.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 26.0-28.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  0.0-30.0 sec   888 MBytes   248 Mbits/sec
[  3] Sent 1818189 datagrams
[  3] Server Report:
[  3]  0.0-29.8 sec   176 MBytes  49.4 Mbits/sec  11.639 ms 1458005/1818173 (80%)
[  3]  0.0-29.8 sec  231 datagrams received out-of-order
```
**Completed:** nf4 → nf1 (10.0.11.3:5004) | BW: 49.4 Mbits/sec | Loss: 80% | Jitter: 11.639 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 57720 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  2.0- 4.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  4.0- 6.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  6.0- 8.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  8.0-10.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 10.0-12.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 12.0-14.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 14.0-16.0 sec  59.2 MBytes   248 Mbits/sec
[  3] 16.0-18.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 18.0-20.0 sec  58.1 MBytes   244 Mbits/sec
[  3] 20.0-22.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 22.0-24.0 sec  56.6 MBytes   238 Mbits/sec
[  3] 24.0-26.0 sec  60.9 MBytes   256 Mbits/sec
[  3] 26.0-28.0 sec  60.6 MBytes   254 Mbits/sec
[  3] 28.0-30.0 sec  60.1 MBytes   252 Mbits/sec
[  3]  0.0-30.0 sec   884 MBytes   247 Mbits/sec
[  3] Sent 1810616 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   314 MBytes  87.7 Mbits/sec   0.649 ms 1167719/1810615 (64%)
[  3]  0.0-30.0 sec  6 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 87.7 Mbits/sec | Loss: 64% | Jitter: 0.649 ms


---

## Summary Statistics

### Phase 1: Latency & Cross-Port Connectivity

* Bitfile type: **router**
* Adjacent port pairs: 4 OK, 0 FAIL
* Cross-port pairs:    8 OK, 0 FAIL
* Average RTT (adjacent): .665 ms
* Average RTT (cross-port): .514 ms

### Phases 2-3: Aggregate Throughput Tests
* Total flows tested: 6
* Average per-flow bandwidth: 47.66 Mbits/sec
* **Total aggregate bandwidth: 286.0 Mbits/sec**

### Phase 4: Individual Flow Baseline Tests
* Total flows tested: 12
* **Peak individual flow bandwidth: 116 Mbits/sec**
* **Min individual flow bandwidth: 34.5 Mbits/sec**
* **Average individual flow bandwidth: 76.54 Mbits/sec**

