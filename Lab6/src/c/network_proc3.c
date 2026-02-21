// ───────────────────────────────────────────────────────────────────
//  FIREWALL PORT FILTER (C Version)
// ───────────────────────────────────────────────────────────────────

#define PACKET_BASE_ADDR  0x400
#define BLOCKED_PORT      23    // Telnet (often blocked for security)
#define STATUS_PENDING    0x00
#define STATUS_ACCEPT     0x01
#define STATUS_REJECT     0x02

typedef struct {
    unsigned int status;       // Offset 0x00 (Result of the filter)
    unsigned int protocol;     // Offset 0x04 (e.g., 6 for TCP)
    unsigned int src_port;     // Offset 0x08
    unsigned int dest_port;    // Offset 0x0C (Port to check)
} TCPHeader;

void main() {
    // 1. Setup Pointer to Packet Buffer
    volatile TCPHeader* pkt = (TCPHeader*)PACKET_BASE_ADDR;

    // 2. Inspect Destination Port
    // We strictly forbid traffic on the BLOCKED_PORT.
    if (pkt->dest_port == BLOCKED_PORT) {
        // 3a. Block the Packet
        pkt->status = STATUS_REJECT;
    } else {
        // 3b. Allow the Packet
        pkt->status = STATUS_ACCEPT;
    }

    return;
}