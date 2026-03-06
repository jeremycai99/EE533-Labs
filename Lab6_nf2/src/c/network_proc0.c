// ───────────────────────────────────────────────────────────────────
//  PACKET PROCESSOR (C Version)
// ───────────────────────────────────────────────────────────────────

#define PACKET_BASE_ADDR  0x100
#define PACKET_TYPE_VALID 0xAA

// Define the structure of our network packet
typedef struct {
    unsigned int header_len;   // Offset 0x00
    unsigned int type;         // Offset 0x04
    unsigned int src_ip;       // Offset 0x08
    unsigned int dest_ip;      // Offset 0x0C
    unsigned int checksum;     // Offset 0x10
    unsigned int payload[4];   // Offset 0x14 - 0x20
} Packet;

void main() {
    // 1. Setup Pointer to Packet Buffer
    // In a real embedded system, this points to a specific memory address.
    volatile Packet* pkt = (Packet*)PACKET_BASE_ADDR;

    // 2. Read & Validate Packet Type
    if (pkt->type != PACKET_TYPE_VALID) {
        // 8. Invalid Packet Handler
        // Write -1 (0xFFFFFFFF) to checksum to indicate error
        pkt->checksum = 0xFFFFFFFF;
        return;
    }

    // 4. NAT: Swap Source and Destination IPs
    unsigned int temp_ip = pkt->src_ip;
    pkt->src_ip = pkt->dest_ip;
    pkt->dest_ip = temp_ip;

    // 5. Prepare for Checksum Loop
    unsigned int sum = 0;
    int i;

    // 6. Checksum Loop (Sum the payload)
    for (i = 0; i < 4; i++) {
        sum += pkt->payload[i];
    }

    // 7. Write Checksum back to Header
    pkt->checksum = sum;

    // 9. Finish
    return;
}