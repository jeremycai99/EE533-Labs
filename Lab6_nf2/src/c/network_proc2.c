// ───────────────────────────────────────────────────────────────────
//  ROUTER TTL PROCESSOR (C Version)
// ───────────────────────────────────────────────────────────────────

#define PACKET_BASE_ADDR  0x300
#define ACTION_DROP       0x00
#define ACTION_FORWARD    0x01

typedef struct {
    unsigned int action;       // Offset 0x00 (0=Drop, 1=Forward)
    unsigned int ttl;          // Offset 0x04 (Time To Live)
    unsigned int src_ip;       // Offset 0x08
    unsigned int dest_ip;      // Offset 0x0C
} IPPacket;

void main() {
    // 1. Setup Pointer to Packet Buffer
    volatile IPPacket* pkt = (IPPacket*)PACKET_BASE_ADDR;

    // 2. Check TTL (Time To Live)
    // If TTL is 1 or less, the packet dies here.
    if (pkt->ttl <= 1) {
        pkt->action = ACTION_DROP;
        pkt->ttl = 0; // Zero it out for clarity
        return;
    }

    // 3. Decrement TTL
    // Standard routing behavior: decrease TTL by 1 at every hop.
    pkt->ttl = pkt->ttl - 1;

    // 4. Set Action to Forward
    // The packet is still valid, send it to the next hop.
    pkt->action = ACTION_FORWARD;

    return;
}