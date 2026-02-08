# Network Bandwidth Test Results (UDP)

## Test Information
* Date: Sat Feb  7 16:03:47 PST 2026
* Team Number: 2
* Bitfile Type: router
* Protocol: UDP
* Packet Size: 1472 bytes
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
| nf3              | nf4          | 10.0.9.3             | adjacent   | OK       | 0.177ms    | 1.214ms    | 5.198ms    | 0%       |
| nf3              | nf0          | 10.0.10.3            | cross-port | OK       | 0.251ms    | 0.736ms    | 2.534ms    | 0%       |
| nf3              | nf1          | 10.0.11.3            | cross-port | OK       | 0.250ms    | 0.927ms    | 3.076ms    | 0%       |
| nf4              | nf3          | 10.0.8.3             | adjacent   | OK       | 0.412ms    | 0.496ms    | 0.605ms    | 0%       |
| nf4              | nf0          | 10.0.10.3            | cross-port | OK       | 0.442ms    | 0.510ms    | 0.630ms    | 0%       |
| nf4              | nf1          | 10.0.11.3            | cross-port | OK       | 0.311ms    | 0.412ms    | 0.567ms    | 0%       |
| nf0              | nf3          | 10.0.8.3             | cross-port | OK       | 0.178ms    | 0.241ms    | 0.420ms    | 0%       |
| nf0              | nf4          | 10.0.9.3             | cross-port | OK       | 0.172ms    | 0.255ms    | 0.368ms    | 0%       |
| nf0              | nf1          | 10.0.11.3            | adjacent   | OK       | 0.347ms    | 0.413ms    | 0.449ms    | 0%       |
| nf1              | nf3          | 10.0.8.3             | cross-port | OK       | 0.354ms    | 0.436ms    | 0.587ms    | 0%       |
| nf1              | nf4          | 10.0.9.3             | cross-port | OK       | 0.289ms    | 0.778ms    | 2.147ms    | 0%       |
| nf1              | nf0          | 10.0.10.3            | adjacent   | OK       | 0.401ms    | 0.511ms    | 0.745ms    | 0%       |

### Connectivity Summary

* **Adjacent port pairs** (NIC + Router): 4 passed, 0 failed
* **Cross port pairs** (Router only):     8 passed, 0 failed

### Latency Summary

* Average RTT (adjacent pairs): .658 ms
* Average RTT (cross-port pairs): .536 ms


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
[  3] local 10.0.8.3 port 37827 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   159 MBytes   668 Mbits/sec
[  3]  2.0- 4.0 sec   171 MBytes   717 Mbits/sec
[  3]  4.0- 6.0 sec   170 MBytes   714 Mbits/sec
[  3]  6.0- 8.0 sec   169 MBytes   710 Mbits/sec
[  3]  8.0-10.0 sec   166 MBytes   696 Mbits/sec
[  3] 10.0-12.0 sec   165 MBytes   692 Mbits/sec
[  3] 12.0-14.0 sec   164 MBytes   688 Mbits/sec
[  3] 14.0-16.0 sec   169 MBytes   710 Mbits/sec
[  3] 16.0-18.0 sec   175 MBytes   734 Mbits/sec
[  3] 18.0-20.0 sec   174 MBytes   728 Mbits/sec
[  3] 20.0-22.0 sec   172 MBytes   721 Mbits/sec
[  3] 22.0-24.0 sec   170 MBytes   713 Mbits/sec
[  3] 24.0-26.0 sec   168 MBytes   706 Mbits/sec
[  3] 26.0-28.0 sec   166 MBytes   697 Mbits/sec
[  3] 28.0-30.0 sec   165 MBytes   690 Mbits/sec
[  3]  0.0-30.0 sec  2.46 GBytes   706 Mbits/sec
[  3] Sent 1797810 datagrams
[  3] Server Report:
[  3]  0.0-29.9 sec   542 MBytes   152 Mbits/sec   0.334 ms 1411510/1797809 (79%)
[  3]  0.0-29.9 sec  144 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 152 Mbits/sec | Loss: 79% | Jitter: 0.334 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 41068 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   165 MBytes   690 Mbits/sec
[  3]  2.0- 4.0 sec   165 MBytes   691 Mbits/sec
[  3]  4.0- 6.0 sec   175 MBytes   735 Mbits/sec
[  3]  6.0- 8.0 sec   174 MBytes   728 Mbits/sec
[  3]  8.0-10.0 sec   172 MBytes   719 Mbits/sec
[  3] 10.0-12.0 sec   170 MBytes   715 Mbits/sec
[  3] 12.0-14.0 sec   169 MBytes   710 Mbits/sec
[  3] 14.0-16.0 sec   167 MBytes   700 Mbits/sec
[  3] 16.0-18.0 sec   166 MBytes   694 Mbits/sec
[  3] 18.0-20.0 sec   164 MBytes   687 Mbits/sec
[  3] 20.0-22.0 sec   168 MBytes   706 Mbits/sec
[  3] 22.0-24.0 sec   174 MBytes   732 Mbits/sec
[  3] 24.0-26.0 sec   173 MBytes   727 Mbits/sec
[  3] 26.0-28.0 sec   172 MBytes   720 Mbits/sec
[  3] 28.0-30.0 sec   170 MBytes   715 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   711 Mbits/sec
[  3] Sent 1812012 datagrams
[  3] Server Report:
[  3]  0.0-29.9 sec   477 MBytes   134 Mbits/sec   0.386 ms 1472037/1812011 (81%)
[  3]  0.0-29.9 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 134 Mbits/sec | Loss: 81% | Jitter: 0.386 ms


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
[  3] local 10.0.8.3 port 35640 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   164 MBytes   687 Mbits/sec
[  3]  2.0- 4.0 sec   164 MBytes   687 Mbits/sec
[  3]  4.0- 6.0 sec   173 MBytes   726 Mbits/sec
[  3]  6.0- 8.0 sec   174 MBytes   732 Mbits/sec
[  3]  8.0-10.0 sec   173 MBytes   725 Mbits/sec
[  3] 10.0-12.0 sec   171 MBytes   717 Mbits/sec
[  3] 12.0-14.0 sec   170 MBytes   711 Mbits/sec
[  3] 14.0-16.0 sec   168 MBytes   703 Mbits/sec
[  3] 16.0-18.0 sec   166 MBytes   696 Mbits/sec
[  3] 18.0-20.0 sec   164 MBytes   688 Mbits/sec
[  3] 20.0-22.0 sec   169 MBytes   710 Mbits/sec
[  3] 22.0-24.0 sec   175 MBytes   734 Mbits/sec
[  3] 24.0-26.0 sec   173 MBytes   728 Mbits/sec
[  3] 26.0-28.0 sec   171 MBytes   719 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   712 Mbits/sec
[  3] Sent 1813194 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   409 MBytes   114 Mbits/sec   0.114 ms 1521200/1813193 (84%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 114 Mbits/sec | Loss: 84% | Jitter: 0.114 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 43310 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   171 MBytes   717 Mbits/sec
[  3]  2.0- 4.0 sec   170 MBytes   715 Mbits/sec
[  3]  4.0- 6.0 sec   168 MBytes   705 Mbits/sec
[  3]  6.0- 8.0 sec   167 MBytes   700 Mbits/sec
[  3]  8.0-10.0 sec   166 MBytes   695 Mbits/sec
[  3] 10.0-12.0 sec   162 MBytes   679 Mbits/sec
[  3] 12.0-14.0 sec   173 MBytes   727 Mbits/sec
[  3] 14.0-16.0 sec   174 MBytes   730 Mbits/sec
[  3] 16.0-18.0 sec   173 MBytes   724 Mbits/sec
[  3] 18.0-20.0 sec   171 MBytes   717 Mbits/sec
[  3] 20.0-22.0 sec   168 MBytes   704 Mbits/sec
[  3] 22.0-24.0 sec   168 MBytes   703 Mbits/sec
[  3] 24.0-26.0 sec   166 MBytes   697 Mbits/sec
[  3] 26.0-28.0 sec   165 MBytes   692 Mbits/sec
[  3] 28.0-30.0 sec   163 MBytes   684 Mbits/sec
[  3]  0.0-30.0 sec  2.47 GBytes   706 Mbits/sec
[  3] Sent 1798207 datagrams
[  3] Server Report:
[  3]  0.0-30.1 sec   365 MBytes   102 Mbits/sec   0.126 ms 1537856/1798206 (86%)
[  3]  0.0-30.1 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 102 Mbits/sec | Loss: 86% | Jitter: 0.126 ms


### Test: nf4 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 53158 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   167 MBytes   699 Mbits/sec
[  3]  2.0- 4.0 sec   166 MBytes   696 Mbits/sec
[  3]  4.0- 6.0 sec   164 MBytes   689 Mbits/sec
[  3]  6.0- 8.0 sec   164 MBytes   689 Mbits/sec
[  3]  8.0-10.0 sec   165 MBytes   691 Mbits/sec
[  3] 10.0-12.0 sec   166 MBytes   697 Mbits/sec
[  3] 12.0-14.0 sec   166 MBytes   697 Mbits/sec
[  3] 14.0-16.0 sec   166 MBytes   695 Mbits/sec
[  3] 16.0-18.0 sec   164 MBytes   686 Mbits/sec
[  3] 18.0-20.0 sec   164 MBytes   686 Mbits/sec
[  3] 20.0-22.0 sec   164 MBytes   689 Mbits/sec
[  3] 22.0-24.0 sec   170 MBytes   711 Mbits/sec
[  3] 24.0-26.0 sec   165 MBytes   692 Mbits/sec
[  3] 26.0-28.0 sec   165 MBytes   691 Mbits/sec
[  3] 28.0-30.0 sec   164 MBytes   687 Mbits/sec
[  3]  0.0-30.0 sec  2.42 GBytes   693 Mbits/sec
[  3] Sent 1765551 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   546 MBytes   152 Mbits/sec   0.129 ms 1375983/1765550 (78%)
[  3]  0.0-30.0 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 152 Mbits/sec | Loss: 78% | Jitter: 0.129 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 50461 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   168 MBytes   703 Mbits/sec
[  3]  2.0- 4.0 sec   173 MBytes   727 Mbits/sec
[  3]  4.0- 6.0 sec   171 MBytes   719 Mbits/sec
[  3]  6.0- 8.0 sec   170 MBytes   711 Mbits/sec
[  3]  8.0-10.0 sec   168 MBytes   703 Mbits/sec
[  3] 10.0-12.0 sec   167 MBytes   702 Mbits/sec
[  3] 12.0-14.0 sec   166 MBytes   695 Mbits/sec
[  3] 14.0-16.0 sec   165 MBytes   692 Mbits/sec
[  3] 16.0-18.0 sec   163 MBytes   685 Mbits/sec
[  3] 18.0-20.0 sec   165 MBytes   691 Mbits/sec
[  3] 20.0-22.0 sec   175 MBytes   733 Mbits/sec
[  3] 22.0-24.0 sec   172 MBytes   721 Mbits/sec
[  3] 24.0-26.0 sec   170 MBytes   712 Mbits/sec
[  3] 26.0-28.0 sec   169 MBytes   708 Mbits/sec
[  3]  0.0-30.0 sec  2.47 GBytes   707 Mbits/sec
[  3] Sent 1800719 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   399 MBytes   111 Mbits/sec   0.327 ms 1516358/1800718 (84%)
[  3]  0.0-30.2 sec  1 datagrams received out-of-order
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 111 Mbits/sec | Loss: 84% | Jitter: 0.327 ms


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
[  3] local 10.0.9.3 port 58633 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   166 MBytes   695 Mbits/sec
[  3]  2.0- 4.0 sec   168 MBytes   703 Mbits/sec
[  3]  4.0- 6.0 sec   175 MBytes   735 Mbits/sec
[  3]  6.0- 8.0 sec   172 MBytes   720 Mbits/sec
[  3]  8.0-10.0 sec   169 MBytes   708 Mbits/sec
[  3] 10.0-12.0 sec   166 MBytes   698 Mbits/sec
[  3] 12.0-14.0 sec   165 MBytes   690 Mbits/sec
[  3] 14.0-16.0 sec   176 MBytes   738 Mbits/sec
[  3] 16.0-18.0 sec   173 MBytes   724 Mbits/sec
[  3] 18.0-20.0 sec   170 MBytes   712 Mbits/sec
[  3] 20.0-22.0 sec   167 MBytes   701 Mbits/sec
[  3] 22.0-24.0 sec   165 MBytes   691 Mbits/sec
[  3] 24.0-26.0 sec   173 MBytes   724 Mbits/sec
[  3] 26.0-28.0 sec   174 MBytes   728 Mbits/sec
[  3] 28.0-30.0 sec   171 MBytes   716 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   712 Mbits/sec
[  3] Sent 1814611 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   887 MBytes   248 Mbits/sec   0.097 ms 1182073/1814610 (65%)
[  3]  0.0-30.0 sec  2 datagrams received out-of-order
```
**Completed:** nf4 → nf3 (10.0.8.3:5001) | BW: 248 Mbits/sec | Loss: 65% | Jitter: 0.097 ms


### Test: nf0 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 37719 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   170 MBytes   711 Mbits/sec
[  3]  2.0- 4.0 sec   169 MBytes   708 Mbits/sec
[  3]  4.0- 6.0 sec   167 MBytes   700 Mbits/sec
[  3]  6.0- 8.0 sec   166 MBytes   694 Mbits/sec
[  3]  8.0-10.0 sec   163 MBytes   684 Mbits/sec
[  3] 10.0-12.0 sec   173 MBytes   724 Mbits/sec
[  3] 12.0-14.0 sec   174 MBytes   732 Mbits/sec
[  3] 14.0-16.0 sec   173 MBytes   726 Mbits/sec
[  3] 16.0-18.0 sec   171 MBytes   718 Mbits/sec
[  3] 18.0-20.0 sec   169 MBytes   710 Mbits/sec
[  3] 20.0-22.0 sec   168 MBytes   705 Mbits/sec
[  3] 22.0-24.0 sec   166 MBytes   698 Mbits/sec
[  3] 24.0-26.0 sec   165 MBytes   690 Mbits/sec
[  3] 26.0-28.0 sec   163 MBytes   682 Mbits/sec
[  3]  0.0-30.0 sec  2.47 GBytes   708 Mbits/sec
[  3] Sent 1803573 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   888 MBytes   248 Mbits/sec   0.071 ms 1170188/1803572 (65%)
[  3]  0.0-30.0 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf3 (10.0.8.3:5001) | BW: 248 Mbits/sec | Loss: 65% | Jitter: 0.071 ms


### Test: nf1 → nf3 (10.0.8.3:5001)

```
------------------------------------------------------------
Client connecting to 10.0.8.3, UDP port 5001
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 59730 connected with 10.0.8.3 port 5001
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   167 MBytes   699 Mbits/sec
[  3]  2.0- 4.0 sec   165 MBytes   691 Mbits/sec
[  3]  4.0- 6.0 sec   163 MBytes   686 Mbits/sec
[  3]  6.0- 8.0 sec   176 MBytes   740 Mbits/sec
[  3]  8.0-10.0 sec   178 MBytes   748 Mbits/sec
[  3] 10.0-12.0 sec   178 MBytes   748 Mbits/sec
[  3] 12.0-14.0 sec   165 MBytes   691 Mbits/sec
[  3] 14.0-16.0 sec   164 MBytes   686 Mbits/sec
[  3] 16.0-18.0 sec   163 MBytes   683 Mbits/sec
[  3] 18.0-20.0 sec   171 MBytes   716 Mbits/sec
[  3] 20.0-22.0 sec   172 MBytes   722 Mbits/sec
[  3] 22.0-24.0 sec   163 MBytes   685 Mbits/sec
[  3] 24.0-26.0 sec   175 MBytes   734 Mbits/sec
[  3] 26.0-28.0 sec   178 MBytes   747 Mbits/sec
[  3] 28.0-30.0 sec   178 MBytes   748 Mbits/sec
[  3]  0.0-30.0 sec  2.50 GBytes   715 Mbits/sec
[  3] Sent 1821086 datagrams
[  3] Server Report:
[  3]  0.0-29.6 sec   561 MBytes   159 Mbits/sec   0.484 ms 1421137/1821084 (78%)
```
**Completed:** nf1 → nf3 (10.0.8.3:5001) | BW: 159 Mbits/sec | Loss: 78% | Jitter: 0.484 ms


### Test: nf3 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 35547 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   166 MBytes   698 Mbits/sec
[  3]  2.0- 4.0 sec   169 MBytes   707 Mbits/sec
[  3]  4.0- 6.0 sec   167 MBytes   701 Mbits/sec
[  3]  6.0- 8.0 sec   165 MBytes   693 Mbits/sec
[  3]  8.0-10.0 sec   163 MBytes   684 Mbits/sec
[  3] 10.0-12.0 sec   174 MBytes   732 Mbits/sec
[  3] 12.0-14.0 sec   175 MBytes   734 Mbits/sec
[  3] 14.0-16.0 sec   173 MBytes   725 Mbits/sec
[  3] 16.0-18.0 sec   171 MBytes   719 Mbits/sec
[  3] 18.0-20.0 sec   169 MBytes   710 Mbits/sec
[  3] 20.0-22.0 sec   168 MBytes   703 Mbits/sec
[  3] 22.0-24.0 sec   166 MBytes   696 Mbits/sec
[  3] 24.0-26.0 sec   164 MBytes   687 Mbits/sec
[  3] 26.0-28.0 sec   171 MBytes   715 Mbits/sec
[  3] 28.0-30.0 sec   175 MBytes   735 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   709 Mbits/sec
[  3] Sent 1806870 datagrams
[  3] Server Report:
[  3]  0.0-29.8 sec  1.19 GBytes   344 Mbits/sec   0.232 ms 934571/1806869 (52%)
[  3]  0.0-29.8 sec  90 datagrams received out-of-order
```
**Completed:** nf3 → nf4 (10.0.9.3:5002) | BW: 344 Mbits/sec | Loss: 52% | Jitter: 0.232 ms


### Test: nf0 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 48795 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   171 MBytes   719 Mbits/sec
[  3]  2.0- 4.0 sec   170 MBytes   714 Mbits/sec
[  3]  4.0- 6.0 sec   168 MBytes   706 Mbits/sec
[  3]  6.0- 8.0 sec   166 MBytes   697 Mbits/sec
[  3]  8.0-10.0 sec   165 MBytes   690 Mbits/sec
[  3] 10.0-12.0 sec   164 MBytes   686 Mbits/sec
[  3] 12.0-14.0 sec   176 MBytes   737 Mbits/sec
[  3] 14.0-16.0 sec   174 MBytes   731 Mbits/sec
[  3] 16.0-18.0 sec   173 MBytes   724 Mbits/sec
[  3] 18.0-20.0 sec   171 MBytes   717 Mbits/sec
[  3] 20.0-22.0 sec   169 MBytes   711 Mbits/sec
[  3] 22.0-24.0 sec   168 MBytes   703 Mbits/sec
[  3] 24.0-26.0 sec   166 MBytes   695 Mbits/sec
[  3] 26.0-28.0 sec   165 MBytes   690 Mbits/sec
[  3] 28.0-30.0 sec   168 MBytes   703 Mbits/sec
[  3]  0.0-30.0 sec  2.47 GBytes   708 Mbits/sec
[  3] Sent 1804334 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec   541 MBytes   150 Mbits/sec  12.591 ms 1418192/1804332 (79%)
```
**Completed:** nf0 → nf4 (10.0.9.3:5002) | BW: 150 Mbits/sec | Loss: 79% | Jitter: 12.591 ms


### Test: nf1 → nf4 (10.0.9.3:5002)

```
------------------------------------------------------------
Client connecting to 10.0.9.3, UDP port 5002
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 49707 connected with 10.0.9.3 port 5002
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   170 MBytes   714 Mbits/sec
[  3]  2.0- 4.0 sec   172 MBytes   720 Mbits/sec
[  3]  4.0- 6.0 sec   169 MBytes   708 Mbits/sec
[  3]  6.0- 8.0 sec   166 MBytes   697 Mbits/sec
[  3]  8.0-10.0 sec   165 MBytes   691 Mbits/sec
[  3] 10.0-12.0 sec   163 MBytes   682 Mbits/sec
[  3] 12.0-14.0 sec   175 MBytes   734 Mbits/sec
[  3] 14.0-16.0 sec   174 MBytes   731 Mbits/sec
[  3] 16.0-18.0 sec   173 MBytes   725 Mbits/sec
[  3] 18.0-20.0 sec   171 MBytes   719 Mbits/sec
[  3] 20.0-22.0 sec   170 MBytes   713 Mbits/sec
[  3] 22.0-24.0 sec   168 MBytes   704 Mbits/sec
[  3] 24.0-26.0 sec   167 MBytes   699 Mbits/sec
[  3] 26.0-28.0 sec   165 MBytes   693 Mbits/sec
[  3]  0.0-30.0 sec  2.47 GBytes   707 Mbits/sec
[  3] Sent 1802130 datagrams
[  3] Server Report:
[  3]  0.0-30.3 sec  1.19 GBytes   337 Mbits/sec  15.014 ms 934123/1802111 (52%)
[  3]  0.0-30.3 sec  4 datagrams received out-of-order
```
**Completed:** nf1 → nf4 (10.0.9.3:5002) | BW: 337 Mbits/sec | Loss: 52% | Jitter: 15.014 ms


### Test: nf3 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 46588 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   166 MBytes   695 Mbits/sec
[  3]  2.0- 4.0 sec   165 MBytes   692 Mbits/sec
[  3]  4.0- 6.0 sec   165 MBytes   693 Mbits/sec
[  3]  6.0- 8.0 sec   175 MBytes   735 Mbits/sec
[  3]  8.0-10.0 sec   173 MBytes   727 Mbits/sec
[  3] 10.0-12.0 sec   172 MBytes   721 Mbits/sec
[  3] 12.0-14.0 sec   170 MBytes   714 Mbits/sec
[  3] 14.0-16.0 sec   169 MBytes   707 Mbits/sec
[  3] 16.0-18.0 sec   167 MBytes   698 Mbits/sec
[  3] 18.0-20.0 sec   165 MBytes   694 Mbits/sec
[  3] 20.0-22.0 sec   163 MBytes   683 Mbits/sec
[  3] 22.0-24.0 sec   174 MBytes   729 Mbits/sec
[  3] 24.0-26.0 sec   173 MBytes   727 Mbits/sec
[  3] 26.0-28.0 sec   172 MBytes   722 Mbits/sec
[  3] 28.0-30.0 sec   170 MBytes   714 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   710 Mbits/sec
[  3] Sent 1809093 datagrams
[  3] Server Report:
[  3]  0.0-29.8 sec   554 MBytes   156 Mbits/sec  14.495 ms 1413914/1809032 (78%)
[  3]  0.0-29.8 sec  104 datagrams received out-of-order
```
**Completed:** nf3 → nf0 (10.0.10.3:5003) | BW: 156 Mbits/sec | Loss: 78% | Jitter: 14.495 ms


### Test: nf4 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 57977 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   170 MBytes   712 Mbits/sec
[  3]  2.0- 4.0 sec   167 MBytes   701 Mbits/sec
[  3]  4.0- 6.0 sec   165 MBytes   690 Mbits/sec
[  3]  6.0- 8.0 sec   173 MBytes   726 Mbits/sec
[  3]  8.0-10.0 sec   173 MBytes   727 Mbits/sec
[  3] 10.0-12.0 sec   171 MBytes   716 Mbits/sec
[  3] 12.0-14.0 sec   168 MBytes   703 Mbits/sec
[  3] 14.0-16.0 sec   165 MBytes   691 Mbits/sec
[  3] 16.0-18.0 sec   170 MBytes   712 Mbits/sec
[  3] 18.0-20.0 sec   174 MBytes   732 Mbits/sec
[  3] 20.0-22.0 sec   171 MBytes   719 Mbits/sec
[  3] 22.0-24.0 sec   169 MBytes   707 Mbits/sec
[  3] 24.0-26.0 sec   165 MBytes   694 Mbits/sec
[  3] 26.0-28.0 sec   167 MBytes   699 Mbits/sec
[  3] 28.0-30.0 sec   175 MBytes   735 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   711 Mbits/sec
[  3] Sent 1811050 datagrams
[  3] Server Report:
[  3]  0.0-29.7 sec   902 MBytes   255 Mbits/sec   0.102 ms 1167710/1811049 (64%)
[  3]  0.0-29.7 sec  1 datagrams received out-of-order
```
**Completed:** nf4 → nf0 (10.0.10.3:5003) | BW: 255 Mbits/sec | Loss: 64% | Jitter: 0.102 ms


### Test: nf1 → nf0 (10.0.10.3:5003)

```
------------------------------------------------------------
Client connecting to 10.0.10.3, UDP port 5003
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.11.3 port 39852 connected with 10.0.10.3 port 5003
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   176 MBytes   736 Mbits/sec
[  3]  2.0- 4.0 sec   174 MBytes   730 Mbits/sec
[  3]  4.0- 6.0 sec   173 MBytes   724 Mbits/sec
[  3]  6.0- 8.0 sec   171 MBytes   715 Mbits/sec
[  3]  8.0-10.0 sec   169 MBytes   708 Mbits/sec
[  3] 10.0-12.0 sec   167 MBytes   702 Mbits/sec
[  3] 12.0-14.0 sec   166 MBytes   698 Mbits/sec
[  3] 14.0-16.0 sec   164 MBytes   689 Mbits/sec
[  3] 16.0-18.0 sec   165 MBytes   692 Mbits/sec
[  3] 18.0-20.0 sec   175 MBytes   735 Mbits/sec
[  3] 20.0-22.0 sec   174 MBytes   731 Mbits/sec
[  3] 22.0-24.0 sec   173 MBytes   724 Mbits/sec
[  3] 24.0-26.0 sec   171 MBytes   716 Mbits/sec
[  3] 26.0-28.0 sec   170 MBytes   711 Mbits/sec
[  3] 28.0-30.0 sec   168 MBytes   704 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   714 Mbits/sec
[  3] Sent 1819890 datagrams
[  3] Server Report:
[  3]  0.0-29.8 sec  1.27 GBytes   366 Mbits/sec   0.161 ms 891624/1819889 (49%)
[  3]  0.0-29.8 sec  3 datagrams received out-of-order
```
**Completed:** nf1 → nf0 (10.0.10.3:5003) | BW: 366 Mbits/sec | Loss: 49% | Jitter: 0.161 ms


### Test: nf3 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.8.3 port 52015 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   164 MBytes   687 Mbits/sec
[  3]  2.0- 4.0 sec   171 MBytes   719 Mbits/sec
[  3]  4.0- 6.0 sec   169 MBytes   707 Mbits/sec
[  3]  6.0- 8.0 sec   168 MBytes   703 Mbits/sec
[  3]  8.0-10.0 sec   166 MBytes   696 Mbits/sec
[  3] 10.0-12.0 sec   165 MBytes   690 Mbits/sec
[  3] 12.0-14.0 sec   163 MBytes   683 Mbits/sec
[  3] 14.0-16.0 sec   162 MBytes   680 Mbits/sec
[  3] 16.0-18.0 sec   161 MBytes   673 Mbits/sec
[  3] 18.0-20.0 sec   169 MBytes   710 Mbits/sec
[  3] 20.0-22.0 sec   169 MBytes   710 Mbits/sec
[  3] 22.0-24.0 sec   168 MBytes   704 Mbits/sec
[  3] 24.0-26.0 sec   166 MBytes   695 Mbits/sec
[  3] 26.0-28.0 sec   165 MBytes   693 Mbits/sec
[  3] 28.0-30.0 sec   164 MBytes   687 Mbits/sec
[  3]  0.0-30.0 sec  2.43 GBytes   696 Mbits/sec
[  3] Sent 1772773 datagrams
[  3] Server Report:
[  3]  0.0-30.2 sec   246 MBytes  68.2 Mbits/sec  10.356 ms 1597034/1772415 (90%)
[  3]  0.0-30.2 sec  3 datagrams received out-of-order
```
**Completed:** nf3 → nf1 (10.0.11.3:5004) | BW: 68.2 Mbits/sec | Loss: 90% | Jitter: 10.356 ms


### Test: nf4 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.9.3 port 40378 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   173 MBytes   725 Mbits/sec
[  3]  2.0- 4.0 sec   170 MBytes   714 Mbits/sec
[  3]  4.0- 6.0 sec   167 MBytes   702 Mbits/sec
[  3]  6.0- 8.0 sec   165 MBytes   690 Mbits/sec
[  3]  8.0-10.0 sec   170 MBytes   714 Mbits/sec
[  3] 10.0-12.0 sec   174 MBytes   728 Mbits/sec
[  3] 12.0-14.0 sec   171 MBytes   715 Mbits/sec
[  3] 14.0-16.0 sec   168 MBytes   705 Mbits/sec
[  3] 16.0-18.0 sec   166 MBytes   696 Mbits/sec
[  3] 18.0-20.0 sec   167 MBytes   699 Mbits/sec
[  3] 20.0-22.0 sec   174 MBytes   731 Mbits/sec
[  3] 22.0-24.0 sec   172 MBytes   722 Mbits/sec
[  3] 24.0-26.0 sec   169 MBytes   709 Mbits/sec
[  3] 26.0-28.0 sec   166 MBytes   697 Mbits/sec
[  3] 28.0-30.0 sec   164 MBytes   686 Mbits/sec
[  3]  0.0-30.0 sec  2.48 GBytes   709 Mbits/sec
[  3] Sent 1806428 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   501 MBytes   140 Mbits/sec   0.291 ms 1449322/1806427 (80%)
[  3]  0.0-30.0 sec  4 datagrams received out-of-order
```
**Completed:** nf4 → nf1 (10.0.11.3:5004) | BW: 140 Mbits/sec | Loss: 80% | Jitter: 0.291 ms


### Test: nf0 → nf1 (10.0.11.3:5004)

```
------------------------------------------------------------
Client connecting to 10.0.11.3, UDP port 5004
Sending 1472 byte datagrams
UDP buffer size: 2.00 MByte (default)
------------------------------------------------------------
[  3] local 10.0.10.3 port 38246 connected with 10.0.11.3 port 5004
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0- 2.0 sec   175 MBytes   732 Mbits/sec
[  3]  2.0- 4.0 sec   174 MBytes   731 Mbits/sec
[  3]  4.0- 6.0 sec   173 MBytes   725 Mbits/sec
[  3]  6.0- 8.0 sec   171 MBytes   718 Mbits/sec
[  3]  8.0-10.0 sec   169 MBytes   710 Mbits/sec
[  3] 10.0-12.0 sec   168 MBytes   703 Mbits/sec
[  3] 12.0-14.0 sec   166 MBytes   695 Mbits/sec
[  3] 14.0-16.0 sec   164 MBytes   690 Mbits/sec
[  3] 16.0-18.0 sec   166 MBytes   695 Mbits/sec
[  3] 18.0-20.0 sec   175 MBytes   736 Mbits/sec
[  3] 20.0-22.0 sec   174 MBytes   729 Mbits/sec
[  3] 22.0-24.0 sec   172 MBytes   721 Mbits/sec
[  3] 24.0-26.0 sec   171 MBytes   716 Mbits/sec
[  3] 26.0-28.0 sec   169 MBytes   708 Mbits/sec
[  3]  0.0-30.0 sec  2.49 GBytes   714 Mbits/sec
[  3] Sent 1819289 datagrams
[  3] Server Report:
[  3]  0.0-30.0 sec   918 MBytes   256 Mbits/sec   0.042 ms 1164160/1819288 (64%)
[  3]  0.0-30.0 sec  1 datagrams received out-of-order
```
**Completed:** nf0 → nf1 (10.0.11.3:5004) | BW: 256 Mbits/sec | Loss: 64% | Jitter: 0.042 ms


---

## Summary Statistics

### Phase 1: Latency & Cross-Port Connectivity

* Bitfile type: **router**
* Adjacent port pairs: 4 OK, 0 FAIL
* Cross-port pairs:    8 OK, 0 FAIL
* Average RTT (adjacent): .658 ms
* Average RTT (cross-port): .536 ms

### Phases 2-3: Aggregate Throughput Tests
* Total flows tested: 6
* Average per-flow bandwidth: 127.50 Mbits/sec
* **Total aggregate bandwidth: 765 Mbits/sec**

### Phase 4: Individual Flow Baseline Tests
* Total flows tested: 12
* **Peak individual flow bandwidth: 366 Mbits/sec**
* **Min individual flow bandwidth: 68.2 Mbits/sec**
* **Average individual flow bandwidth: 227.26 Mbits/sec**

