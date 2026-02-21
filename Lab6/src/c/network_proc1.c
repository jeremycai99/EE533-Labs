// ───────────────────────────────────────────────────────────────────
//  PACKET ENCRYPTION MODULE (C Version)
// ───────────────────────────────────────────────────────────────────

#define PACKET_BASE_ADDR  0x200
#define STATUS_PLAIN      0x00
#define STATUS_SECURE     0x01
#define ENCRYPTION_KEY    0xDEADBEEF  // 32-bit Cipher Key

// Define the structure of our data packet
typedef struct {
    unsigned int status;       // Offset 0x00 (0=Plain, 1=Secure)
    unsigned int packet_id;    // Offset 0x04
    unsigned int payload[4];   // Offset 0x08 - 0x14 (Data to encrypt)
} CryptoPacket;

void main() {
    // 1. Setup Pointer to Packet Buffer
    // Pointing to memory address 0x200
    volatile CryptoPacket* pkt = (CryptoPacket*)PACKET_BASE_ADDR;

    // 2. Check Status
    // If the packet is not "Plain" (0), it might already be encrypted.
    // In that case, we do nothing and exit.
    if (pkt->status != STATUS_PLAIN) {
        return;
    }

    // 3. Encrypt Payload Loop
    // We iterate through the 4 words of payload and apply XOR (^)
    // with our secret key.
    int i;
    for (i = 0; i < 4; i++) {
        pkt->payload[i] = pkt->payload[i] ^ ENCRYPTION_KEY;
    }

    // 4. Update Status Flag
    // Mark the packet as "Secure" so downstream systems know it is encrypted.
    pkt->status = STATUS_SECURE;

    // 5. Finish
    return;
}