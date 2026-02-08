# Network Bandwidth Test Results (UDP)

## Test Information
* Date: Sat Feb  7 15:53:12 PST 2026
* Team Number: 2
* Bitfile Type: nic
* Protocol: UDP
* Packet Size: 1472 bytes
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
| nf3              | nf4          | 10.0.9.3             | adjacent   | OK       | 0.944ms    | 1.832ms    | 5.326ms    | 0%       |
| nf3              | nf0          | 10.0.10.3            | cross-port | OK       | 0.990ms    | 1.632ms    | 3.769ms    | 0%       |
| nf3              | nf1          | 10.0.11.3            | cross-port | OK       | 1.008ms    | 1.816ms    | 3.304ms    | 0%       |
| nf4              | nf3          | 10.0.8.3             | adjacent   | OK       | 0.871ms    | 0.980ms    | 1.143ms    | 0%       |
| nf4              | nf0          | 10.0.10.3            | cross-port | OK       | 1.014ms    | 1.140ms    | 1.510ms    | 0%       |
| nf4              | nf1          | 10.0.11.3            | cross-port | OK       | 0.921ms    | 0.994ms    | 1.122ms    | 0%       |
| nf0              | nf3          | 10.0.8.3             | cross-port | OK       | 0.756ms    | 0.851ms    | 1.089ms    | 0%       |
| nf0              | nf4          | 10.0.9.3             | cross-port | OK       | 0.698ms    | 0.773ms    | 0.933ms    | 0%       |
| nf0              | nf1          | 10.0.11.3            | adjacent   | OK       | 0.801ms    | 0.901ms    | 1.167ms    | 0%       |
| nf1              | nf3          | 10.0.8.3             | cross-port | OK       | 0.950ms    | 0.989ms    | 1.054ms    | 0%       |
| nf1              | nf4          | 10.0.9.3             | cross-port | OK       | 0.778ms    | 0.886ms    | 1.032ms    | 0%       |
| nf1              | nf0          | 10.0.10.3            | adjacent   | OK       | 1.079ms    | 1.550ms    | 3.002ms    | 0%       |

### Connectivity Summary

* **Adjacent port pairs** (NIC + Router): 4 passed, 0 failed
* **Cross port pairs** (Router only):     8 passed, 0 failed

### Latency Summary

* Average RTT (adjacent pairs): 1.315 ms
* Average RTT (cross-port pairs): 1.135 ms

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
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 41339 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   171 MBytes   718 Mbits/sec
[  3]  2.0- 4.0 sec   171 MBytes   717 Mbits/sec
[  3]  4.0- 6.0 sec   169 MBytes   710 Mbits/sec
[  3]  6.0- 8.0 sec   168 MBytes   703 Mbits/sec
[  3]  8.0-10.0 sec   166 MBytes   695 Mbits/sec
[  3] 10.0-12.0 sec   164 MBytes   687 Mbits/sec
[  3] 12.0-14.0 sec   171 MBytes   718 Mbits/sec
[  3] 14.0-16.0 sec   175 MBytes   734 Mbits/sec
[  3] 16.0-18.0 sec   173 MBytes   726 Mbits/sec
[  3] 18.0-20.0 sec   171 MBytes   718 Mbits/sec
[  3] 20.0-22.0 sec   170 MBytes   711 Mbits/sec
[  3] 22.0-24.0 sec   168 MBytes   704 Mbits/sec
[  3] 24.0-26.0 sec   166 MBytes   697 Mbits/sec
[  3] 26.0-28.0 sec   164 MBytes   690 Mbits/sec
[  3] 28.0-30.0 sec   168 MBytes   704 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   709 Mbits/sec
[  3] Sent 1805730 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   263 MBytes  72.9 Mbits/sec   9.947 ms 1618122/1805728 (90%)
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 72.9 Mbits/sec | Loss: 90% | Jitter: 9.947 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 45735 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   175 MBytes   733 Mbits/sec
[  3]  2.0- 4.0 sec   173 MBytes   726 Mbits/sec
[  3]  4.0- 6.0 sec   171 MBytes   718 Mbits/sec
[  3]  6.0- 8.0 sec   170 MBytes   712 Mbits/sec
[  3]  8.0-10.0 sec   168 MBytes   705 Mbits/sec
[  3] 10.0-12.0 sec   166 MBytes   697 Mbits/sec
[  3] 12.0-14.0 sec   165 MBytes   694 Mbits/sec
[  3] 14.0-16.0 sec   163 MBytes   685 Mbits/sec
[  3] 16.0-18.0 sec   175 MBytes   732 Mbits/sec
[  3] 18.0-20.0 sec   174 MBytes   731 Mbits/sec
[  3] 20.0-22.0 sec   172 MBytes   722 Mbits/sec
[  3] 22.0-24.0 sec   171 MBytes   717 Mbits/sec
[  3] 24.0-26.0 sec   169 MBytes   709 Mbits/sec
[  3] 26.0-28.0 sec   167 MBytes   702 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   712 Mbits/sec
[  3] Sent 1813863 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   273 MBytes  75.7 Mbits/sec  12.276 ms 1619056/1813854 (89%)
[  3]  0.0-30.2 sec  4 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 75.7 Mbits/sec | Loss: 89% | Jitter: 12.276 ms


---

## Phase 3: Aggregate Throughput Test (Bidirectional)

Testing 4 simultaneous bidirectional flows to measure maximum aggregate capacity.
* Flow 1: nf3 ↔ nf4 (10.0.8.3 ↔ 10.0.9.3)
* Flow 2: nf0 ↔ nf1 (10.0.10.3 ↔ 10.0.11.3)


### Test: nf3 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 56192 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   164 MBytes   686 Mbits/sec
[  3]  2.0- 4.0 sec   172 MBytes   720 Mbits/sec
[  3]  4.0- 6.0 sec   177 MBytes   742 Mbits/sec
[  3]  6.0- 8.0 sec   175 MBytes   736 Mbits/sec
[  3]  8.0-10.0 sec   170 MBytes   712 Mbits/sec
[  3] 10.0-12.0 sec   174 MBytes   732 Mbits/sec
[  3] 12.0-14.0 sec   174 MBytes   728 Mbits/sec
[  3] 14.0-16.0 sec   168 MBytes   706 Mbits/sec
[  3] 16.0-18.0 sec   164 MBytes   690 Mbits/sec
[  3] 18.0-20.0 sec   169 MBytes   708 Mbits/sec
[  3] 20.0-22.0 sec   177 MBytes   743 Mbits/sec
[  3] 22.0-24.0 sec   175 MBytes   733 Mbits/sec
[  3] 24.0-26.0 sec   167 MBytes   699 Mbits/sec
[  3] 26.0-28.0 sec   169 MBytes   711 Mbits/sec
[  3]  0.0-30.0 sec  2.50 GBytes   717 Mbits/sec
[  3] Sent 1826704 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec  41.4 MBytes  11.5 Mbits/sec  10.643 ms 1797151/1826703 (98%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 11.5 Mbits/sec | Loss: 98% | Jitter: 10.643 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 47428 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   168 MBytes   706 Mbits/sec
[  3]  2.0- 4.0 sec   167 MBytes   698 Mbits/sec
[  3]  4.0- 6.0 sec   164 MBytes   690 Mbits/sec
[  3]  6.0- 8.0 sec   164 MBytes   688 Mbits/sec
[  3]  8.0-10.0 sec   176 MBytes   737 Mbits/sec
[  3] 10.0-12.0 sec   175 MBytes   733 Mbits/sec
[  3] 12.0-14.0 sec   172 MBytes   723 Mbits/sec
[  3] 14.0-16.0 sec   171 MBytes   717 Mbits/sec
[  3] 16.0-18.0 sec   169 MBytes   711 Mbits/sec
[  3] 18.0-20.0 sec   167 MBytes   702 Mbits/sec
[  3] 20.0-22.0 sec   166 MBytes   695 Mbits/sec
[  3] 22.0-24.0 sec   164 MBytes   688 Mbits/sec
[  3] 24.0-26.0 sec   167 MBytes   700 Mbits/sec
[  3] 26.0-28.0 sec   175 MBytes   736 Mbits/sec
[  3] 28.0-30.0 sec   172 MBytes   720 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   710 Mbits/sec
[  3] Sent 1807824 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   155 MBytes  42.9 Mbits/sec   8.366 ms 1697402/1807822 (94%)
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 42.9 Mbits/sec | Loss: 94% | Jitter: 8.366 ms


### Test: nf4 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 42674 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   170 MBytes   714 Mbits/sec
[  3]  2.0- 4.0 sec   167 MBytes   699 Mbits/sec
[  3]  4.0- 6.0 sec   165 MBytes   691 Mbits/sec
[  3]  6.0- 8.0 sec   175 MBytes   734 Mbits/sec
[  3]  8.0-10.0 sec   173 MBytes   725 Mbits/sec
[  3] 10.0-12.0 sec   170 MBytes   711 Mbits/sec
[  3] 12.0-14.0 sec   167 MBytes   699 Mbits/sec
[  3] 14.0-16.0 sec   164 MBytes   690 Mbits/sec
[  3] 16.0-18.0 sec   174 MBytes   731 Mbits/sec
[  3] 18.0-20.0 sec   173 MBytes   726 Mbits/sec
[  3] 20.0-22.0 sec   171 MBytes   716 Mbits/sec
[  3] 22.0-24.0 sec   168 MBytes   704 Mbits/sec
[  3] 24.0-26.0 sec   165 MBytes   690 Mbits/sec
[  3] 26.0-28.0 sec   170 MBytes   715 Mbits/sec
[  3] 28.0-30.0 sec   174 MBytes   732 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   712 Mbits/sec
[  3] Sent 1813408 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   147 MBytes  40.8 Mbits/sec   9.878 ms 1708421/1813406 (94%)
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 40.8 Mbits/sec | Loss: 94% | Jitter: 9.878 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 52718 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   165 MBytes   694 Mbits/sec
[  3]  2.0- 4.0 sec   163 MBytes   683 Mbits/sec
[  3]  4.0- 6.0 sec   173 MBytes   726 Mbits/sec
[  3]  6.0- 8.0 sec   174 MBytes   730 Mbits/sec
[  3]  8.0-10.0 sec   173 MBytes   726 Mbits/sec
[  3] 10.0-12.0 sec   171 MBytes   719 Mbits/sec
[  3] 12.0-14.0 sec   170 MBytes   711 Mbits/sec
[  3] 14.0-16.0 sec   168 MBytes   707 Mbits/sec
[  3] 16.0-18.0 sec   167 MBytes   699 Mbits/sec
[  3] 18.0-20.0 sec   165 MBytes   694 Mbits/sec
[  3] 20.0-22.0 sec   164 MBytes   686 Mbits/sec
[  3] 22.0-24.0 sec   172 MBytes   720 Mbits/sec
[  3] 24.0-26.0 sec   175 MBytes   735 Mbits/sec
[  3] 26.0-28.0 sec   173 MBytes   727 Mbits/sec
[  3] 28.0-30.0 sec   171 MBytes   719 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   712 Mbits/sec
[  3] Sent 1813408 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   103 MBytes  28.6 Mbits/sec   9.142 ms 1739947/1813406 (96%)
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 28.6 Mbits/sec | Loss: 96% | Jitter: 9.142 ms


---

## Phase 4: Individual Flow Baseline Tests

Testing each flow individually (sequentially) to establish baseline performance.


### Test: nf4 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 35778 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   172 MBytes   720 Mbits/sec
[  3]  2.0- 4.0 sec   169 MBytes   710 Mbits/sec
[  3]  4.0- 6.0 sec   167 MBytes   699 Mbits/sec
[  3]  6.0- 8.0 sec   164 MBytes   686 Mbits/sec
[  3]  8.0-10.0 sec   175 MBytes   734 Mbits/sec
[  3] 10.0-12.0 sec   172 MBytes   723 Mbits/sec
[  3] 12.0-14.0 sec   170 MBytes   713 Mbits/sec
[  3] 14.0-16.0 sec   167 MBytes   701 Mbits/sec
[  3] 16.0-18.0 sec   164 MBytes   688 Mbits/sec
[  3] 18.0-20.0 sec   171 MBytes   718 Mbits/sec
[  3] 20.0-22.0 sec   173 MBytes   725 Mbits/sec
[  3] 22.0-24.0 sec   171 MBytes   716 Mbits/sec
[  3] 24.0-26.0 sec   168 MBytes   706 Mbits/sec
[  3] 26.0-28.0 sec   164 MBytes   689 Mbits/sec
[  3] 28.0-30.0 sec   168 MBytes   704 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   709 Mbits/sec
[  3] Sent 1805749 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   303 MBytes  84.6 Mbits/sec   0.205 ms 1589414/1805748 (88%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 84.6 Mbits/sec | Loss: 88% | Jitter: 0.205 ms


### Test: nf0 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 52893 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   147 MBytes   616 Mbits/sec
[  3]  2.0- 4.0 sec   146 MBytes   612 Mbits/sec
[  3]  4.0- 6.0 sec   153 MBytes   642 Mbits/sec
[  3]  6.0- 8.0 sec   175 MBytes   733 Mbits/sec
[  3]  8.0-10.0 sec   172 MBytes   721 Mbits/sec
[  3] 10.0-12.0 sec   169 MBytes   710 Mbits/sec
[  3] 12.0-14.0 sec   168 MBytes   703 Mbits/sec
[  3] 14.0-16.0 sec   166 MBytes   695 Mbits/sec
[  3] 16.0-18.0 sec   165 MBytes   690 Mbits/sec
[  3] 18.0-20.0 sec   163 MBytes   684 Mbits/sec
[  3] 20.0-22.0 sec   162 MBytes   680 Mbits/sec
[  3] 22.0-24.0 sec   168 MBytes   707 Mbits/sec
[  3] 24.0-26.0 sec   172 MBytes   723 Mbits/sec
[  3] 26.0-28.0 sec   171 MBytes   717 Mbits/sec
[  3] 28.0-30.0 sec   169 MBytes   707 Mbits/sec
[  3]  0.0-30.0 sec  2.41 GBytes   689 Mbits/sec
[  3] Sent 1756172 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   365 MBytes   101 Mbits/sec  11.602 ms 1495786/1756171 (85%)
[  3]  0.0-30.3 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf3 (10.0.8.3:5001) | BW: 101 Mbits/sec | Loss: 85% | Jitter: 11.602 ms


### Test: nf1 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 33295 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   169 MBytes   707 Mbits/sec
[  3]  2.0- 4.0 sec   167 MBytes   700 Mbits/sec
[  3]  4.0- 6.0 sec   166 MBytes   695 Mbits/sec
[  3]  6.0- 8.0 sec   164 MBytes   686 Mbits/sec
[  3]  8.0-10.0 sec   171 MBytes   719 Mbits/sec
[  3] 10.0-12.0 sec   175 MBytes   735 Mbits/sec
[  3] 12.0-14.0 sec   174 MBytes   729 Mbits/sec
[  3] 14.0-16.0 sec   172 MBytes   720 Mbits/sec
[  3] 16.0-18.0 sec   170 MBytes   715 Mbits/sec
[  3] 18.0-20.0 sec   168 MBytes   706 Mbits/sec
[  3] 20.0-22.0 sec   168 MBytes   704 Mbits/sec
[  3] 22.0-24.0 sec   165 MBytes   694 Mbits/sec
[  3] 24.0-26.0 sec   164 MBytes   688 Mbits/sec
[  3] 26.0-28.0 sec   170 MBytes   714 Mbits/sec
[  3] 28.0-30.0 sec   175 MBytes   735 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   710 Mbits/sec
[  3] Sent 1808223 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   359 MBytes   100 Mbits/sec   0.091 ms 1552441/1808222 (86%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf1 → nf3 (10.0.8.3:5001) | BW: 100 Mbits/sec | Loss: 86% | Jitter: 0.091 ms


### Test: nf3 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 40591 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   168 MBytes   703 Mbits/sec
[  3]  2.0- 4.0 sec   166 MBytes   698 Mbits/sec
[  3]  4.0- 6.0 sec   165 MBytes   692 Mbits/sec
[  3]  6.0- 8.0 sec   166 MBytes   694 Mbits/sec
[  3]  8.0-10.0 sec   176 MBytes   739 Mbits/sec
[  3] 10.0-12.0 sec   174 MBytes   730 Mbits/sec
[  3] 12.0-14.0 sec   172 MBytes   721 Mbits/sec
[  3] 14.0-16.0 sec   170 MBytes   715 Mbits/sec
[  3] 16.0-18.0 sec   169 MBytes   710 Mbits/sec
[  3] 18.0-20.0 sec   167 MBytes   700 Mbits/sec
[  3] 20.0-22.0 sec   166 MBytes   695 Mbits/sec
[  3] 22.0-24.0 sec   163 MBytes   686 Mbits/sec
[  3] 24.0-26.0 sec   175 MBytes   732 Mbits/sec
[  3] 26.0-28.0 sec   175 MBytes   732 Mbits/sec
[  3] 28.0-30.0 sec   173 MBytes   724 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   711 Mbits/sec
[  3] Sent 1812346 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   339 MBytes  94.1 Mbits/sec   9.886 ms 1570308/1812344 (87%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 94.1 Mbits/sec | Loss: 87% | Jitter: 9.886 ms


### Test: nf0 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 48522 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   166 MBytes   698 Mbits/sec
[  3]  2.0- 4.0 sec   164 MBytes   690 Mbits/sec
[  3]  4.0- 6.0 sec   163 MBytes   684 Mbits/sec
[  3]  6.0- 8.0 sec   175 MBytes   732 Mbits/sec
[  3]  8.0-10.0 sec   171 MBytes   718 Mbits/sec
[  3] 10.0-12.0 sec   176 MBytes   740 Mbits/sec
[  3] 12.0-14.0 sec   163 MBytes   685 Mbits/sec
[  3] 14.0-16.0 sec   164 MBytes   690 Mbits/sec
[  3] 16.0-18.0 sec   167 MBytes   699 Mbits/sec
[  3] 18.0-20.0 sec   166 MBytes   696 Mbits/sec
[  3] 20.0-22.0 sec   164 MBytes   688 Mbits/sec
[  3] 22.0-24.0 sec   167 MBytes   700 Mbits/sec
[  3] 24.0-26.0 sec   176 MBytes   737 Mbits/sec
[  3] 26.0-28.0 sec   174 MBytes   732 Mbits/sec
[  3] 28.0-30.0 sec   172 MBytes   723 Mbits/sec
[  3]  0.0-30.0 sec  2.47 GBytes   707 Mbits/sec
[  3] Sent 1802240 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   358 MBytes  99.7 Mbits/sec   0.143 ms 1547038/1802239 (86%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf4 (10.0.9.3:5002) | BW: 99.7 Mbits/sec | Loss: 86% | Jitter: 0.143 ms


### Test: nf1 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 51494 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   173 MBytes   725 Mbits/sec
[  3]  2.0- 4.0 sec   171 MBytes   717 Mbits/sec
[  3]  4.0- 6.0 sec   170 MBytes   711 Mbits/sec
[  3]  6.0- 8.0 sec   168 MBytes   705 Mbits/sec
[  3]  8.0-10.0 sec   166 MBytes   697 Mbits/sec
[  3] 10.0-12.0 sec   165 MBytes   690 Mbits/sec
[  3] 12.0-14.0 sec   164 MBytes   687 Mbits/sec
[  3] 14.0-16.0 sec   176 MBytes   739 Mbits/sec
[  3] 16.0-18.0 sec   175 MBytes   732 Mbits/sec
[  3] 18.0-20.0 sec   173 MBytes   727 Mbits/sec
[  3] 20.0-22.0 sec   171 MBytes   719 Mbits/sec
[  3] 22.0-24.0 sec   169 MBytes   711 Mbits/sec
[  3] 24.0-26.0 sec   168 MBytes   704 Mbits/sec
[  3] 26.0-28.0 sec   167 MBytes   699 Mbits/sec
[  3] 28.0-30.0 sec   165 MBytes   690 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   710 Mbits/sec
[  3] Sent 1809220 datagrams
[  3] Server Report:
[  3]  0.0-30.6 sec   336 MBytes  92.1 Mbits/sec   0.507 ms 1569258/1809219 (87%)
[  3]  0.0-30.6 sec  4 datagrams received out-of-order
```
**Completed:** nf1 → nf4 (10.0.9.3:5002) | BW: 92.1 Mbits/sec | Loss: 87% | Jitter: 0.507 ms


### Test: nf3 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 50521 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   162 MBytes   679 Mbits/sec
[  3]  2.0- 4.0 sec   168 MBytes   704 Mbits/sec
[  3]  4.0- 6.0 sec   170 MBytes   713 Mbits/sec
[  3]  6.0- 8.0 sec   168 MBytes   704 Mbits/sec
[  3]  8.0-10.0 sec   165 MBytes   692 Mbits/sec
[  3] 10.0-12.0 sec   165 MBytes   692 Mbits/sec
[  3] 12.0-14.0 sec   164 MBytes   686 Mbits/sec
[  3] 14.0-16.0 sec   161 MBytes   677 Mbits/sec
[  3] 16.0-18.0 sec   164 MBytes   686 Mbits/sec
[  3] 18.0-20.0 sec   172 MBytes   723 Mbits/sec
[  3] 20.0-22.0 sec   175 MBytes   734 Mbits/sec
[  3] 22.0-24.0 sec   173 MBytes   727 Mbits/sec
[  3] 24.0-26.0 sec   172 MBytes   720 Mbits/sec
[  3] 26.0-28.0 sec   170 MBytes   712 Mbits/sec
[  3]  0.0-30.0 sec  2.46 GBytes   704 Mbits/sec
[  3] Sent 1792334 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   310 MBytes  85.9 Mbits/sec   7.818 ms 1571516/1792333 (88%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf0 (10.0.10.3:5003) | BW: 85.9 Mbits/sec | Loss: 88% | Jitter: 7.818 ms


### Test: nf4 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 53897 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   174 MBytes   732 Mbits/sec
[  3]  2.0- 4.0 sec   172 MBytes   721 Mbits/sec
[  3]  4.0- 6.0 sec   169 MBytes   707 Mbits/sec
[  3]  6.0- 8.0 sec   166 MBytes   697 Mbits/sec
[  3]  8.0-10.0 sec   166 MBytes   695 Mbits/sec
[  3] 10.0-12.0 sec   175 MBytes   735 Mbits/sec
[  3] 12.0-14.0 sec   172 MBytes   723 Mbits/sec
[  3] 14.0-16.0 sec   170 MBytes   712 Mbits/sec
[  3] 16.0-18.0 sec   167 MBytes   700 Mbits/sec
[  3] 18.0-20.0 sec   164 MBytes   689 Mbits/sec
[  3] 20.0-22.0 sec   175 MBytes   732 Mbits/sec
[  3] 22.0-24.0 sec   173 MBytes   727 Mbits/sec
[  3] 24.0-26.0 sec   170 MBytes   714 Mbits/sec
[  3] 26.0-28.0 sec   167 MBytes   702 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   712 Mbits/sec
[  3] Sent 1812812 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   358 MBytes  99.4 Mbits/sec  10.989 ms 1557221/1812811 (86%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf0 (10.0.10.3:5003) | BW: 99.4 Mbits/sec | Loss: 86% | Jitter: 10.989 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 41938 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   173 MBytes   727 Mbits/sec
[  3]  2.0- 4.0 sec   175 MBytes   736 Mbits/sec
[  3]  4.0- 6.0 sec   173 MBytes   727 Mbits/sec
[  3]  6.0- 8.0 sec   172 MBytes   721 Mbits/sec
[  3]  8.0-10.0 sec   170 MBytes   713 Mbits/sec
[  3] 10.0-12.0 sec   169 MBytes   708 Mbits/sec
[  3] 12.0-14.0 sec   167 MBytes   699 Mbits/sec
[  3] 14.0-16.0 sec   165 MBytes   694 Mbits/sec
[  3] 16.0-18.0 sec   164 MBytes   687 Mbits/sec
[  3] 18.0-20.0 sec   172 MBytes   720 Mbits/sec
[  3] 20.0-22.0 sec   175 MBytes   735 Mbits/sec
[  3] 22.0-24.0 sec   174 MBytes   728 Mbits/sec
[  3] 24.0-26.0 sec   172 MBytes   721 Mbits/sec
[  3] 26.0-28.0 sec   170 MBytes   713 Mbits/sec
[  3] 28.0-30.0 sec   169 MBytes   709 Mbits/sec
[  3]  0.0-30.0 sec  2.50 GBytes   716 Mbits/sec
[  3] Sent 1823878 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   334 MBytes  92.6 Mbits/sec   9.565 ms 1585624/1823876 (87%)
[  3]  0.0-30.3 sec  2 datagrams received out-of-order
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 92.6 Mbits/sec | Loss: 87% | Jitter: 9.565 ms


### Test: nf3 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 45402 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   172 MBytes   720 Mbits/sec
[  3]  2.0- 4.0 sec   172 MBytes   721 Mbits/sec
[  3]  4.0- 6.0 sec   170 MBytes   714 Mbits/sec
[  3]  6.0- 8.0 sec   169 MBytes   707 Mbits/sec
[  3]  8.0-10.0 sec   167 MBytes   701 Mbits/sec
[  3] 10.0-12.0 sec   165 MBytes   691 Mbits/sec
[  3] 12.0-14.0 sec   164 MBytes   688 Mbits/sec
[  3] 14.0-16.0 sec   176 MBytes   738 Mbits/sec
[  3] 16.0-18.0 sec   174 MBytes   732 Mbits/sec
[  3] 18.0-20.0 sec   173 MBytes   724 Mbits/sec
[  3] 20.0-22.0 sec   170 MBytes   715 Mbits/sec
[  3] 22.0-24.0 sec   170 MBytes   711 Mbits/sec
[  3] 24.0-26.0 sec   167 MBytes   701 Mbits/sec
[  3] 26.0-28.0 sec   165 MBytes   693 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   710 Mbits/sec
[  3] Sent 1807926 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   483 MBytes   135 Mbits/sec   0.082 ms 1463630/1807925 (81%)
[  3]  0.0-30.1 sec  4 datagrams received out-of-order
```
**Completed:** nf3 → nf1 (10.0.11.3:5004) | BW: 135 Mbits/sec | Loss: 81% | Jitter: 0.082 ms


### Test: nf4 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 33491 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   170 MBytes   711 Mbits/sec
[  3]  2.0- 4.0 sec   174 MBytes   729 Mbits/sec
[  3]  4.0- 6.0 sec   171 MBytes   718 Mbits/sec
[  3]  6.0- 8.0 sec   169 MBytes   707 Mbits/sec
[  3]  8.0-10.0 sec   166 MBytes   695 Mbits/sec
[  3] 10.0-12.0 sec   167 MBytes   699 Mbits/sec
[  3] 12.0-14.0 sec   174 MBytes   728 Mbits/sec
[  3] 14.0-16.0 sec   171 MBytes   719 Mbits/sec
[  3] 16.0-18.0 sec   169 MBytes   710 Mbits/sec
[  3] 18.0-20.0 sec   166 MBytes   698 Mbits/sec
[  3] 20.0-22.0 sec   164 MBytes   687 Mbits/sec
[  3] 22.0-24.0 sec   175 MBytes   734 Mbits/sec
[  3] 24.0-26.0 sec   173 MBytes   725 Mbits/sec
[  3] 26.0-28.0 sec   170 MBytes   712 Mbits/sec
[  3] 28.0-30.0 sec   167 MBytes   699 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   711 Mbits/sec
[  3] Sent 1812005 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   310 MBytes  85.9 Mbits/sec  11.444 ms 1591035/1811998 (88%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf1 (10.0.11.3:5004) | BW: 85.9 Mbits/sec | Loss: 88% | Jitter: 11.444 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 45914 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   170 MBytes   715 Mbits/sec
[  3]  2.0- 4.0 sec   168 MBytes   704 Mbits/sec
[  3]  4.0- 6.0 sec   166 MBytes   697 Mbits/sec
[  3]  6.0- 8.0 sec   165 MBytes   692 Mbits/sec
[  3]  8.0-10.0 sec   164 MBytes   689 Mbits/sec
[  3] 10.0-12.0 sec   176 MBytes   738 Mbits/sec
[  3] 12.0-14.0 sec   174 MBytes   732 Mbits/sec
[  3] 14.0-16.0 sec   173 MBytes   724 Mbits/sec
[  3] 16.0-18.0 sec   171 MBytes   716 Mbits/sec
[  3] 18.0-20.0 sec   169 MBytes   711 Mbits/sec
[  3] 20.0-22.0 sec   168 MBytes   703 Mbits/sec
[  3] 22.0-24.0 sec   165 MBytes   693 Mbits/sec
[  3] 24.0-26.0 sec   164 MBytes   689 Mbits/sec
[  3] 26.0-28.0 sec   168 MBytes   703 Mbits/sec
[  3] 28.0-30.0 sec   175 MBytes   735 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   709 Mbits/sec
[  3] Sent 1807277 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   435 MBytes   121 Mbits/sec   0.098 ms 1497066/1807276 (83%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 121 Mbits/sec | Loss: 83% | Jitter: 0.098 ms


---

## Summary Statistics

### Phase 1: Latency & Cross-Port Connectivity

* Bitfile type: **nic**
* Adjacent port pairs: 4 OK, 0 FAIL
* Cross-port pairs:    8 OK, 0 FAIL
* Average RTT (adjacent): 1.315 ms
* Average RTT (cross-port): 1.135 ms

### Phases 2-3: Aggregate Throughput Tests
* Total flows tested: 6
* Average per-flow bandwidth: 45.40 Mbits/sec
* **Total aggregate bandwidth: 272.4 Mbits/sec**

### Phase 4: Individual Flow Baseline Tests
* Total flows tested: 12
* **Peak individual flow bandwidth: 135 Mbits/sec**
* **Min individual flow bandwidth: 84.6 Mbits/sec**
* **Average individual flow bandwidth: 99.27 Mbits/sec**

