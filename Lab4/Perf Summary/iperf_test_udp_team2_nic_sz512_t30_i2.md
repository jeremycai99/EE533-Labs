# Network Bandwidth Test Results (UDP)

## Test Information
* Date: Sat Feb  7 15:38:10 PST 2026
* Team Number: 2
* Bitfile Type: nic
* Protocol: UDP
* Packet Size: 512 bytes
* Test Duration: 30 seconds
* Test Interval: 2 seconds
* Target Bandwidth: 1G
* RKD Enabled: No

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
Time: 16:21:17
Error Registers: 0
Good, after resetting programming interface the FIFO is empty
Download completed -  2377668 bytes. (expected 2377668).
DONE went high - chip has been successfully programmed.
CPCI Information
----------------
Version: 4 (rev 1)

Device (Virtex) Information
---------------------------
Project directory: reference_nic
Project name: Reference NIC
Project description: Reference NIC

Device ID: 1
Version: 1.1.0
Built against CPCI version: 4 (rev 1)

Virtex design compiled against active CPCI version

---

## Phase 1: Latency & Cross-Port Connectivity Verification

Testing RTT latency and connectivity across all port pairs via data plane IPs.

NIC mode: Only adjacent port pairs should succeed. Cross-port expected to FAIL.

### Latency Matrix (RTT in ms)

| Source           | Destination  | Target Address       | Pair Type  | Status   | Min RTT    | Avg RTT    | Max RTT    | Loss %   |
|------------------|--------------|----------------------|------------|----------|------------|------------|------------|----------|
| nf3              | nf4          | 10.0.9.3             | adjacent   | OK       | 0.915ms    | 1.898ms    | 3.831ms    | 0%       |
| nf3              | nf0          | 10.0.10.3            | cross-port | OK       | 1.075ms    | 1.606ms    | 3.538ms    | 0%       |
| nf3              | nf1          | 10.0.11.3            | cross-port | OK       | 1.060ms    | 1.678ms    | 3.767ms    | 0%       |
| nf4              | nf3          | 10.0.8.3             | adjacent   | OK       | 1.005ms    | 1.071ms    | 1.202ms    | 0%       |
| nf4              | nf0          | 10.0.10.3            | cross-port | OK       | 0.861ms    | 1.077ms    | 1.528ms    | 0%       |
| nf4              | nf1          | 10.0.11.3            | cross-port | OK       | 0.875ms    | 1.036ms    | 1.430ms    | 0%       |
| nf0              | nf3          | 10.0.8.3             | cross-port | OK       | 0.951ms    | 1.183ms    | 1.648ms    | 0%       |
| nf0              | nf4          | 10.0.9.3             | cross-port | OK       | 0.740ms    | 0.809ms    | 1.071ms    | 0%       |
| nf0              | nf1          | 10.0.11.3            | adjacent   | OK       | 0.798ms    | 1.332ms    | 2.765ms    | 0%       |
| nf1              | nf3          | 10.0.8.3             | cross-port | OK       | 0.981ms    | 1.195ms    | 1.509ms    | 0%       |
| nf1              | nf4          | 10.0.9.3             | cross-port | OK       | 0.832ms    | 0.930ms    | 1.092ms    | 0%       |
| nf1              | nf0          | 10.0.10.3            | adjacent   | OK       | 0.995ms    | 1.157ms    | 1.287ms    | 0%       |

### Connectivity Summary

* **Adjacent port pairs** (NIC + Router): 4 passed, 0 failed
* **Cross port pairs** (Router only):     8 passed, 0 failed

### Latency Summary

* Average RTT (adjacent pairs): 1.364 ms
* Average RTT (cross-port pairs): 1.189 ms

* **WARNING:** Cross-port pairs succeeded in NIC mode. Verify topology.

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
[  3] local 10.0.8.3 port 54224 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  2.0- 4.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  4.0- 6.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  6.0- 8.0 sec  61.1 MBytes   256 Mbits/sec
[  3]  8.0-10.0 sec  60.6 MBytes   254 Mbits/sec
[  3] 10.0-12.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 12.0-14.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 14.0-16.0 sec  58.7 MBytes   246 Mbits/sec
[  3] 16.0-18.0 sec  58.2 MBytes   244 Mbits/sec
[  3] 18.0-20.0 sec  57.4 MBytes   241 Mbits/sec
[  3] 20.0-22.0 sec  56.8 MBytes   238 Mbits/sec
[  3] 22.0-24.0 sec  60.9 MBytes   255 Mbits/sec
[  3] 24.0-26.0 sec  60.7 MBytes   255 Mbits/sec
[  3] 26.0-28.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 28.0-30.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  0.0-30.0 sec   886 MBytes   248 Mbits/sec
[  3] Sent 1815502 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   155 MBytes  42.9 Mbits/sec  10.286 ms 1498295/1815501 (83%)
[  3]  0.0-30.3 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 42.9 Mbits/sec | Loss: 83% | Jitter: 10.286 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 53937 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  2.0- 4.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  4.0- 6.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  6.0- 8.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  8.0-10.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 10.0-12.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 12.0-14.0 sec  60.4 MBytes   253 Mbits/sec
[  3] 14.0-16.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 16.0-18.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 18.0-20.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 20.0-22.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 22.0-24.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 24.0-26.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 26.0-28.0 sec  59.5 MBytes   249 Mbits/sec
[  3] 28.0-30.0 sec  60.8 MBytes   255 Mbits/sec
[  3]  0.0-30.0 sec   883 MBytes   247 Mbits/sec
[  3] Sent 1808522 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   135 MBytes  37.7 Mbits/sec   0.510 ms 1531593/1808521 (85%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 37.7 Mbits/sec | Loss: 85% | Jitter: 0.510 ms


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
[  3] local 10.0.8.3 port 33985 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  2.0- 4.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  4.0- 6.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  6.0- 8.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  8.0-10.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 10.0-12.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 12.0-14.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 14.0-16.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 16.0-18.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 18.0-20.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 20.0-22.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 22.0-24.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 24.0-26.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 26.0-28.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 28.0-30.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  0.0-30.0 sec   880 MBytes   246 Mbits/sec
[  3] Sent 1802240 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec  63.9 MBytes  17.7 Mbits/sec   3.993 ms 1671282/1802238 (93%)
[  3]  0.0-30.3 sec  3 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 17.7 Mbits/sec | Loss: 93% | Jitter: 3.993 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 50938 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  2.0- 4.0 sec  60.7 MBytes   254 Mbits/sec
[  3]  4.0- 6.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  6.0- 8.0 sec  59.5 MBytes   250 Mbits/sec
[  3]  8.0-10.0 sec  58.8 MBytes   246 Mbits/sec
[  3] 10.0-12.0 sec  58.0 MBytes   243 Mbits/sec
[  3] 12.0-14.0 sec  57.6 MBytes   241 Mbits/sec
[  3] 14.0-16.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 16.0-18.0 sec  56.2 MBytes   236 Mbits/sec
[  3] 18.0-20.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 20.0-22.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 22.0-24.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 24.0-26.0 sec  59.2 MBytes   248 Mbits/sec
[  3] 26.0-28.0 sec  58.8 MBytes   247 Mbits/sec
[  3] 28.0-30.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  0.0-30.0 sec   884 MBytes   247 Mbits/sec
[  3] Sent 1809918 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec  78.0 MBytes  21.6 Mbits/sec   9.594 ms 1650239/1809916 (91%)
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 21.6 Mbits/sec | Loss: 91% | Jitter: 9.594 ms


### Test: nf4 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 60781 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  61.7 MBytes   259 Mbits/sec
[  3]  2.0- 4.0 sec  62.0 MBytes   260 Mbits/sec
[  3]  4.0- 6.0 sec  57.8 MBytes   242 Mbits/sec
[  3]  6.0- 8.0 sec  58.4 MBytes   245 Mbits/sec
[  3]  8.0-10.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 10.0-12.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 12.0-14.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 14.0-16.0 sec  59.7 MBytes   251 Mbits/sec
[  3] 16.0-18.0 sec  58.5 MBytes   246 Mbits/sec
[  3] 18.0-20.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 20.0-22.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 22.0-24.0 sec  62.0 MBytes   260 Mbits/sec
[  3] 24.0-26.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 26.0-28.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 28.0-30.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  0.0-30.0 sec   894 MBytes   250 Mbits/sec
[  3] Sent 1830160 datagrams
[  3] Server Report:
[  3]  0.0-30.7 sec  56.6 MBytes  15.5 Mbits/sec  35.556 ms 1714170/1830158 (94%)
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 15.5 Mbits/sec | Loss: 94% | Jitter: 35.556 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 50877 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  2.0- 4.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  4.0- 6.0 sec  56.8 MBytes   238 Mbits/sec
[  3]  6.0- 8.0 sec  56.7 MBytes   238 Mbits/sec
[  3]  8.0-10.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 10.0-12.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 12.0-14.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 14.0-16.0 sec  62.0 MBytes   260 Mbits/sec
[  3] 16.0-18.0 sec  61.7 MBytes   259 Mbits/sec
[  3] 18.0-20.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 20.0-22.0 sec  59.5 MBytes   250 Mbits/sec
[  3] 22.0-24.0 sec  58.7 MBytes   246 Mbits/sec
[  3] 24.0-26.0 sec  57.0 MBytes   239 Mbits/sec
[  3] 26.0-28.0 sec  61.3 MBytes   257 Mbits/sec
[  3]  0.0-30.0 sec   881 MBytes   246 Mbits/sec
[  3] Sent 1805198 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec  78.0 MBytes  21.7 Mbits/sec   0.059 ms 1645468/1805197 (91%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 21.7 Mbits/sec | Loss: 91% | Jitter: 0.059 ms


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
[  3] local 10.0.9.3 port 37134 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  2.0- 4.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  4.0- 6.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  6.0- 8.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  8.0-10.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 10.0-12.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 12.0-14.0 sec  59.2 MBytes   248 Mbits/sec
[  3] 14.0-16.0 sec  60.5 MBytes   254 Mbits/sec
[  3] 16.0-18.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 18.0-20.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 20.0-22.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 22.0-24.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 24.0-26.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 26.0-28.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 28.0-30.0 sec  59.0 MBytes   247 Mbits/sec
[  3]  0.0-30.0 sec   885 MBytes   248 Mbits/sec
[  3] Sent 1813408 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   186 MBytes  51.7 Mbits/sec  11.178 ms 1431463/1813406 (79%)
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 51.7 Mbits/sec | Loss: 79% | Jitter: 11.178 ms


### Test: nf0 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 39625 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  60.7 MBytes   255 Mbits/sec
[  3]  2.0- 4.0 sec  60.2 MBytes   253 Mbits/sec
[  3]  4.0- 6.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  6.0- 8.0 sec  59.0 MBytes   247 Mbits/sec
[  3]  8.0-10.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 10.0-12.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 12.0-14.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 14.0-16.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 16.0-18.0 sec  61.3 MBytes   257 Mbits/sec
[  3] 18.0-20.0 sec  60.6 MBytes   254 Mbits/sec
[  3] 20.0-22.0 sec  60.1 MBytes   252 Mbits/sec
[  3] 22.0-24.0 sec  59.7 MBytes   251 Mbits/sec
[  3] 24.0-26.0 sec  59.1 MBytes   248 Mbits/sec
[  3] 26.0-28.0 sec  58.0 MBytes   243 Mbits/sec
[  3]  0.0-30.0 sec   887 MBytes   248 Mbits/sec
[  3] Sent 1815600 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   216 MBytes  60.3 Mbits/sec   0.045 ms 1373457/1815599 (76%)
[  3]  0.0-30.0 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf3 (10.0.8.3:5001) | BW: 60.3 Mbits/sec | Loss: 76% | Jitter: 0.045 ms


### Test: nf1 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 42879 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  61.3 MBytes   257 Mbits/sec
[  3]  2.0- 4.0 sec  61.0 MBytes   256 Mbits/sec
[  3]  4.0- 6.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  6.0- 8.0 sec  59.7 MBytes   250 Mbits/sec
[  3]  8.0-10.0 sec  59.0 MBytes   248 Mbits/sec
[  3] 10.0-12.0 sec  58.5 MBytes   246 Mbits/sec
[  3] 12.0-14.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 14.0-16.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 16.0-18.0 sec  56.7 MBytes   238 Mbits/sec
[  3] 18.0-20.0 sec  60.9 MBytes   255 Mbits/sec
[  3] 20.0-22.0 sec  60.7 MBytes   255 Mbits/sec
[  3] 22.0-24.0 sec  60.6 MBytes   254 Mbits/sec
[  3] 24.0-26.0 sec  59.7 MBytes   250 Mbits/sec
[  3] 26.0-28.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 28.0-30.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  0.0-30.0 sec   892 MBytes   249 Mbits/sec
[  3] Sent 1826670 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   180 MBytes  49.8 Mbits/sec  10.101 ms 1458821/1826668 (80%)
```
**Completed:** nf1 → nf3 (10.0.8.3:5001) | BW: 49.8 Mbits/sec | Loss: 80% | Jitter: 10.101 ms


### Test: nf3 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 46357 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  56.0 MBytes   235 Mbits/sec
[  3]  2.0- 4.0 sec  60.0 MBytes   252 Mbits/sec
[  3]  4.0- 6.0 sec  60.4 MBytes   253 Mbits/sec
[  3]  6.0- 8.0 sec  59.7 MBytes   250 Mbits/sec
[  3]  8.0-10.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 10.0-12.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 12.0-14.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 14.0-16.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 16.0-18.0 sec  57.8 MBytes   242 Mbits/sec
[  3] 18.0-20.0 sec  61.2 MBytes   257 Mbits/sec
[  3] 20.0-22.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 22.0-24.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 24.0-26.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 26.0-28.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  0.0-30.0 sec   885 MBytes   247 Mbits/sec
[  3] Sent 1812226 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   184 MBytes  51.0 Mbits/sec   9.184 ms 1435740/1812119 (79%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 51.0 Mbits/sec | Loss: 79% | Jitter: 9.184 ms


### Test: nf0 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 54978 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  54.4 MBytes   228 Mbits/sec
[  3]  2.0- 4.0 sec  7.58 MBytes  31.8 Mbits/sec
[  3]  4.0- 6.0 sec  7.57 MBytes  31.7 Mbits/sec
[  3]  6.0- 8.0 sec  44.7 MBytes   188 Mbits/sec
[  3]  8.0-10.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 10.0-12.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 12.0-14.0 sec  55.8 MBytes   234 Mbits/sec
[  3] 14.0-16.0 sec  7.61 MBytes  31.9 Mbits/sec
[  3] 16.0-18.0 sec  7.58 MBytes  31.8 Mbits/sec
[  3] 18.0-20.0 sec  7.59 MBytes  31.8 Mbits/sec
[  3] 20.0-22.0 sec  7.55 MBytes  31.7 Mbits/sec
[  3] 22.0-24.0 sec  7.56 MBytes  31.7 Mbits/sec
[  3] 24.0-26.0 sec  54.8 MBytes   230 Mbits/sec
[  3] 26.0-28.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 28.0-30.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  0.0-30.0 sec   496 MBytes   139 Mbits/sec
[  3] Sent 1014994 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   107 MBytes  29.9 Mbits/sec   0.584 ms 795872/1014993 (78%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf4 (10.0.9.3:5002) | BW: 29.9 Mbits/sec | Loss: 78% | Jitter: 0.584 ms


### Test: nf1 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 48665 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  57.9 MBytes   243 Mbits/sec
[  3]  2.0- 4.0 sec  57.3 MBytes   240 Mbits/sec
[  3]  4.0- 6.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  6.0- 8.0 sec  61.3 MBytes   257 Mbits/sec
[  3]  8.0-10.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 10.0-12.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 12.0-14.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 14.0-16.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 16.0-18.0 sec  58.4 MBytes   245 Mbits/sec
[  3] 18.0-20.0 sec  57.8 MBytes   242 Mbits/sec
[  3] 20.0-22.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 22.0-24.0 sec  57.7 MBytes   242 Mbits/sec
[  3] 24.0-26.0 sec  61.3 MBytes   257 Mbits/sec
[  3] 26.0-28.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 28.0-30.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  0.0-30.0 sec   887 MBytes   248 Mbits/sec
[  3] Sent 1816898 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   174 MBytes  48.5 Mbits/sec   0.546 ms 1460543/1816897 (80%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf1 → nf4 (10.0.9.3:5002) | BW: 48.5 Mbits/sec | Loss: 80% | Jitter: 0.546 ms


### Test: nf3 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 58091 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  60.0 MBytes   252 Mbits/sec
[  3]  2.0- 4.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  4.0- 6.0 sec  58.7 MBytes   246 Mbits/sec
[  3]  6.0- 8.0 sec  58.2 MBytes   244 Mbits/sec
[  3]  8.0-10.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 10.0-12.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 12.0-14.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 14.0-16.0 sec  60.6 MBytes   254 Mbits/sec
[  3] 16.0-18.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 18.0-20.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 20.0-22.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 22.0-24.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 24.0-26.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 26.0-28.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 28.0-30.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  0.0-30.0 sec   880 MBytes   246 Mbits/sec
[  3] Sent 1802240 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   216 MBytes  60.0 Mbits/sec  12.418 ms 1359287/1802239 (75%)
[  3]  0.0-30.2 sec  4 datagrams received out-of-order
```
**Completed:** nf3 → nf0 (10.0.10.3:5003) | BW: 60.0 Mbits/sec | Loss: 75% | Jitter: 12.418 ms


### Test: nf4 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 34295 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  2.0- 4.0 sec  57.4 MBytes   241 Mbits/sec
[  3]  4.0- 6.0 sec  59.3 MBytes   249 Mbits/sec
[  3]  6.0- 8.0 sec  60.5 MBytes   254 Mbits/sec
[  3]  8.0-10.0 sec  59.6 MBytes   250 Mbits/sec
[  3] 10.0-12.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 12.0-14.0 sec  57.4 MBytes   241 Mbits/sec
[  3] 14.0-16.0 sec  58.1 MBytes   244 Mbits/sec
[  3] 16.0-18.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 18.0-20.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 20.0-22.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 22.0-24.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 24.0-26.0 sec  57.0 MBytes   239 Mbits/sec
[  3] 26.0-28.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 28.0-30.0 sec  60.2 MBytes   253 Mbits/sec
[  3]  0.0-30.0 sec   885 MBytes   247 Mbits/sec
[  3] Sent 1812012 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   169 MBytes  47.1 Mbits/sec   0.578 ms 1466615/1812011 (81%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf0 (10.0.10.3:5003) | BW: 47.1 Mbits/sec | Loss: 81% | Jitter: 0.578 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 40970 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  59.4 MBytes   249 Mbits/sec
[  3]  2.0- 4.0 sec  58.6 MBytes   246 Mbits/sec
[  3]  4.0- 6.0 sec  58.3 MBytes   244 Mbits/sec
[  3]  6.0- 8.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  8.0-10.0 sec  56.9 MBytes   239 Mbits/sec
[  3] 10.0-12.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 12.0-14.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 14.0-16.0 sec  60.4 MBytes   253 Mbits/sec
[  3] 16.0-18.0 sec  59.9 MBytes   251 Mbits/sec
[  3] 18.0-20.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 20.0-22.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 22.0-24.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 24.0-26.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 26.0-28.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  0.0-30.0 sec   881 MBytes   246 Mbits/sec
[  3] Sent 1804835 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   256 MBytes  71.3 Mbits/sec   0.071 ms 1281086/1804834 (71%)
[  3]  0.0-30.1 sec  4 datagrams received out-of-order
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 71.3 Mbits/sec | Loss: 71% | Jitter: 0.071 ms


### Test: nf3 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 58701 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  59.0 MBytes   247 Mbits/sec
[  3]  2.0- 4.0 sec  58.1 MBytes   244 Mbits/sec
[  3]  4.0- 6.0 sec  57.5 MBytes   241 Mbits/sec
[  3]  6.0- 8.0 sec  56.9 MBytes   239 Mbits/sec
[  3]  8.0-10.0 sec  59.8 MBytes   251 Mbits/sec
[  3] 10.0-12.0 sec  60.5 MBytes   254 Mbits/sec
[  3] 12.0-14.0 sec  60.3 MBytes   253 Mbits/sec
[  3] 14.0-16.0 sec  59.3 MBytes   249 Mbits/sec
[  3] 16.0-18.0 sec  59.0 MBytes   247 Mbits/sec
[  3] 18.0-20.0 sec  58.3 MBytes   244 Mbits/sec
[  3] 20.0-22.0 sec  57.6 MBytes   242 Mbits/sec
[  3] 22.0-24.0 sec  57.2 MBytes   240 Mbits/sec
[  3] 24.0-26.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 26.0-28.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 28.0-30.0 sec  60.3 MBytes   253 Mbits/sec
[  3]  0.0-30.0 sec   883 MBytes   247 Mbits/sec
[  3] Sent 1808522 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   160 MBytes  44.6 Mbits/sec   0.547 ms 1481209/1808521 (82%)
[  3]  0.0-30.0 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf1 (10.0.11.3:5004) | BW: 44.6 Mbits/sec | Loss: 82% | Jitter: 0.547 ms


### Test: nf4 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 45414 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  9.22 MBytes  38.7 Mbits/sec
[  3]  2.0- 4.0 sec  9.23 MBytes  38.7 Mbits/sec
[  3]  4.0- 6.0 sec  9.21 MBytes  38.6 Mbits/sec
[  3]  6.0- 8.0 sec  9.22 MBytes  38.7 Mbits/sec
[  3]  8.0-10.0 sec  9.21 MBytes  38.6 Mbits/sec
[  3] 10.0-12.0 sec  9.23 MBytes  38.7 Mbits/sec
[  3] 12.0-14.0 sec  9.23 MBytes  38.7 Mbits/sec
[  3] 14.0-16.0 sec  9.24 MBytes  38.8 Mbits/sec
[  3] 16.0-18.0 sec  9.23 MBytes  38.7 Mbits/sec
[  3] 18.0-20.0 sec  9.21 MBytes  38.6 Mbits/sec
[  3] 20.0-22.0 sec  9.21 MBytes  38.6 Mbits/sec
[  3] 22.0-24.0 sec  9.25 MBytes  38.8 Mbits/sec
[  3] 24.0-26.0 sec  9.25 MBytes  38.8 Mbits/sec
[  3] 26.0-28.0 sec  9.22 MBytes  38.7 Mbits/sec
[  3]  0.0-30.0 sec   138 MBytes  38.7 Mbits/sec
[  3] Sent 283417 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec  87.1 MBytes  24.2 Mbits/sec  11.356 ms 104972/283388 (37%)
```
**Completed:** nf4 → nf1 (10.0.11.3:5004) | BW: 24.2 Mbits/sec | Loss: 37% | Jitter: 11.356 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 512 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 53686 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec  57.6 MBytes   242 Mbits/sec
[  3]  2.0- 4.0 sec  56.6 MBytes   238 Mbits/sec
[  3]  4.0- 6.0 sec  59.9 MBytes   251 Mbits/sec
[  3]  6.0- 8.0 sec  60.7 MBytes   254 Mbits/sec
[  3]  8.0-10.0 sec  60.1 MBytes   252 Mbits/sec
[  3] 10.0-12.0 sec  59.8 MBytes   251 Mbits/sec
[  3] 12.0-14.0 sec  59.0 MBytes   248 Mbits/sec
[  3] 14.0-16.0 sec  58.6 MBytes   246 Mbits/sec
[  3] 16.0-18.0 sec  57.9 MBytes   243 Mbits/sec
[  3] 18.0-20.0 sec  57.3 MBytes   240 Mbits/sec
[  3] 20.0-22.0 sec  56.6 MBytes   237 Mbits/sec
[  3] 22.0-24.0 sec  61.0 MBytes   256 Mbits/sec
[  3] 24.0-26.0 sec  60.7 MBytes   254 Mbits/sec
[  3] 26.0-28.0 sec  60.0 MBytes   252 Mbits/sec
[  3] 28.0-30.0 sec  59.6 MBytes   250 Mbits/sec
[  3]  0.0-30.0 sec   885 MBytes   248 Mbits/sec
[  3] Sent 1813408 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   105 MBytes  29.3 Mbits/sec   0.603 ms 1598230/1813407 (88%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 29.3 Mbits/sec | Loss: 88% | Jitter: 0.603 ms


---

## Summary Statistics

### Phase 1: Latency & Cross-Port Connectivity

* Bitfile type: **nic**
* Adjacent port pairs: 4 OK, 0 FAIL
* Cross-port pairs:    8 OK, 0 FAIL
* Average RTT (adjacent): 1.364 ms
* Average RTT (cross-port): 1.189 ms

### Phases 2-3: Aggregate Throughput Tests
* Total flows tested: 6
* Average per-flow bandwidth: 26.18 Mbits/sec
* **Total aggregate bandwidth: 157.1 Mbits/sec**

### Phase 4: Individual Flow Baseline Tests
* Total flows tested: 12
* **Peak individual flow bandwidth: 71.3 Mbits/sec**
* **Min individual flow bandwidth: 24.2 Mbits/sec**
* **Average individual flow bandwidth: 47.30 Mbits/sec**

