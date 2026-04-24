/*
 * net_model_send.c
 *
 * Raw Ethernet sender for model_test-emitted SoC command frames.
 *
 * This is intended for the old NetFPGA Fedora environment where Python may be
 * too old.  It still needs permission to open AF_PACKET/SOCK_RAW sockets:
 * run as root, or ask an admin/TA to install the compiled binary setuid root.
 */

#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define ETHERTYPE_SOC 0x88B5
#define MAX_LINE 65536

typedef unsigned long long u64;

typedef struct {
    int index;
    int expect;
    int cpu;
    int rb_words;
    int n_words;
    u64 *words;
} frame_t;

typedef struct {
    frame_t *v;
    int n;
    int cap;
} frame_vec_t;

static const char *CLASS_NAMES[4] = {
    "DoS", "NORMAL", "c_ci_na_1", "c_se_na_1"
};

static void die(const char *msg)
{
    perror(msg);
    exit(1);
}

static double now_sec(void)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void sleep_sec(double sec)
{
    if (sec <= 0.0)
        return;
    usleep((unsigned int)(sec * 1000000.0));
}

static void put_le16(FILE *fh, unsigned int v)
{
    fputc((int)(v & 0xFF), fh);
    fputc((int)((v >> 8) & 0xFF), fh);
}

static void put_le32(FILE *fh, unsigned int v)
{
    fputc((int)(v & 0xFF), fh);
    fputc((int)((v >> 8) & 0xFF), fh);
    fputc((int)((v >> 16) & 0xFF), fh);
    fputc((int)((v >> 24) & 0xFF), fh);
}

static void write_pcap_header(FILE *fh)
{
    put_le32(fh, 0xA1B2C3D4U); /* native-endian pcap magic */
    put_le16(fh, 2);
    put_le16(fh, 4);
    put_le32(fh, 0);
    put_le32(fh, 0);
    put_le32(fh, 65535);
    put_le32(fh, 1); /* LINKTYPE_ETHERNET */
}

static void write_pcap_packet(FILE *fh, const unsigned char *pkt, int len, double ts)
{
    unsigned int sec;
    unsigned int usec;

    if (ts < 0.0)
        ts = 0.0;
    sec = (unsigned int)ts;
    usec = (unsigned int)((ts - (double)sec) * 1000000.0);

    put_le32(fh, sec);
    put_le32(fh, usec);
    put_le32(fh, (unsigned int)len);
    put_le32(fh, (unsigned int)len);
    fwrite(pkt, 1, (size_t)len, fh);
}

static int parse_mac(const char *s, unsigned char mac[6])
{
    unsigned int b0, b1, b2, b3, b4, b5;
    int n;

    n = sscanf(s, "%x:%x:%x:%x:%x:%x", &b0, &b1, &b2, &b3, &b4, &b5);
    if (n != 6)
        return -1;
    if (b0 > 255 || b1 > 255 || b2 > 255 || b3 > 255 || b4 > 255 || b5 > 255)
        return -1;
    mac[0] = (unsigned char)b0;
    mac[1] = (unsigned char)b1;
    mac[2] = (unsigned char)b2;
    mac[3] = (unsigned char)b3;
    mac[4] = (unsigned char)b4;
    mac[5] = (unsigned char)b5;
    return 0;
}

static int get_iface_index(int sock, const char *iface)
{
    struct ifreq ifr;

    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0)
        die("SIOCGIFINDEX");
    return ifr.ifr_ifindex;
}

static int get_iface_mac(int sock, const char *iface, unsigned char mac[6])
{
    struct ifreq ifr;

    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    if (ioctl(sock, SIOCGIFHWADDR, &ifr) < 0)
        return -1;
    memcpy(mac, ifr.ifr_hwaddr.sa_data, 6);
    return 0;
}

static void frame_vec_push(frame_vec_t *fv, frame_t *f)
{
    frame_t *nv;
    int new_cap;

    if (fv->n == fv->cap) {
        new_cap = fv->cap ? fv->cap * 2 : 16;
        nv = (frame_t *)realloc(fv->v, (size_t)new_cap * sizeof(frame_t));
        if (!nv) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        fv->v = nv;
        fv->cap = new_cap;
    }
    fv->v[fv->n++] = *f;
}

static void free_frames(frame_vec_t *fv)
{
    int i;
    for (i = 0; i < fv->n; i++)
        free(fv->v[i].words);
    free(fv->v);
    fv->v = 0;
    fv->n = 0;
    fv->cap = 0;
}

static void make_smoke(frame_vec_t *fv)
{
    static const u64 smoke_words[] = {
        0x1000000300000000ULL,
        0xE5901000E3A00000ULL,
        0xE0813002E5902004ULL,
        0xEAFFFFFEE5803008ULL,
        0x2000000200000000ULL,
        0x000000140000000AULL,
        0x0000000000000000ULL,
        0x3000000000000000ULL,
        0x4000000200000000ULL,
        0x5000000000000000ULL
    };
    frame_t f;
    int i;

    memset(&f, 0, sizeof(f));
    f.index = 0;
    f.expect = 1;
    f.cpu = 1;
    f.rb_words = 2;
    f.n_words = (int)(sizeof(smoke_words) / sizeof(smoke_words[0]));
    f.words = (u64 *)malloc((size_t)f.n_words * sizeof(u64));
    if (!f.words) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (i = 0; i < f.n_words; i++)
        f.words[i] = smoke_words[i];
    frame_vec_push(fv, &f);
}

static void parse_frame_file(const char *path, frame_vec_t *fv)
{
    FILE *fh;
    char line[MAX_LINE];
    int lineno;

    fh = fopen(path, "r");
    if (!fh)
        die(path);

    lineno = 0;
    while (fgets(line, sizeof(line), fh)) {
        char *tok;
        char *save;
        frame_t f;
        int declared;
        int word_cap;

        lineno++;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        memset(&f, 0, sizeof(f));
        declared = -1;
        word_cap = 0;

        tok = strtok_r(line, " \t\r\n", &save);
        if (!tok)
            continue;
        if (strcmp(tok, "FRAME") != 0) {
            fprintf(stderr, "%s:%d: expected FRAME line\n", path, lineno);
            exit(1);
        }
        tok = strtok_r(0, " \t\r\n", &save);
        if (!tok) {
            fprintf(stderr, "%s:%d: missing frame index\n", path, lineno);
            exit(1);
        }
        f.index = atoi(tok);

        while ((tok = strtok_r(0, " \t\r\n", &save)) != 0) {
            char *eq;
            if (strlen(tok) == 16 && strchr(tok, '=') == 0) {
                break;
            }
            eq = strchr(tok, '=');
            if (!eq) {
                fprintf(stderr, "%s:%d: bad metadata token %s\n", path, lineno, tok);
                exit(1);
            }
            *eq = 0;
            if (strcmp(tok, "expect") == 0)
                f.expect = atoi(eq + 1);
            else if (strcmp(tok, "cpu") == 0)
                f.cpu = atoi(eq + 1);
            else if (strcmp(tok, "rb") == 0)
                f.rb_words = atoi(eq + 1);
            else if (strcmp(tok, "words") == 0)
                declared = atoi(eq + 1);
        }

        while (tok) {
            if (f.n_words == word_cap) {
                u64 *nw;
                word_cap = word_cap ? word_cap * 2 : 64;
                nw = (u64 *)realloc(f.words, (size_t)word_cap * sizeof(u64));
                if (!nw) {
                    fprintf(stderr, "out of memory\n");
                    exit(1);
                }
                f.words = nw;
            }
            f.words[f.n_words++] = strtoull(tok, 0, 16);
            tok = strtok_r(0, " \t\r\n", &save);
        }

        if (declared >= 0 && declared != f.n_words) {
            fprintf(stderr, "%s:%d: metadata says words=%d, got %d\n",
                    path, lineno, declared, f.n_words);
            exit(1);
        }
        frame_vec_push(fv, &f);
    }
    fclose(fh);
}

static unsigned char *build_packet(frame_t *f,
                                   const unsigned char dst[6],
                                   const unsigned char src[6],
                                   unsigned short ethertype,
                                   int *out_len)
{
    int len;
    int pos;
    int i;
    unsigned char *pkt;

    len = 14 + 2 + f->n_words * 8;
    if (len < 64)
        len = 64;
    pkt = (unsigned char *)calloc((size_t)len, 1);
    if (!pkt) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    memcpy(pkt + 0, dst, 6);
    memcpy(pkt + 6, src, 6);
    pkt[12] = (unsigned char)((ethertype >> 8) & 0xFF);
    pkt[13] = (unsigned char)(ethertype & 0xFF);
    pos = 16; /* Ethernet header (14) + pkt_proc pad (2) */

    for (i = 0; i < f->n_words; i++) {
        u64 w = f->words[i];
        pkt[pos + 0] = (unsigned char)((w >> 56) & 0xFF);
        pkt[pos + 1] = (unsigned char)((w >> 48) & 0xFF);
        pkt[pos + 2] = (unsigned char)((w >> 40) & 0xFF);
        pkt[pos + 3] = (unsigned char)((w >> 32) & 0xFF);
        pkt[pos + 4] = (unsigned char)((w >> 24) & 0xFF);
        pkt[pos + 5] = (unsigned char)((w >> 16) & 0xFF);
        pkt[pos + 6] = (unsigned char)((w >> 8) & 0xFF);
        pkt[pos + 7] = (unsigned char)(w & 0xFF);
        pos += 8;
    }
    *out_len = len;
    return pkt;
}

static int parse_rx_words(const unsigned char *pkt, int len, int rb_words, u64 *words, int max_words)
{
    int avail;
    int count;
    int i;
    int pos;

    if (len < 16)
        return 0;
    avail = (len - 16) / 8;
    count = avail;
    if (rb_words > 0 && rb_words < count)
        count = rb_words;
    if (count > max_words)
        count = max_words;

    pos = 16;
    for (i = 0; i < count; i++) {
        u64 w = 0;
        int j;
        for (j = 0; j < 8; j++)
            w = (w << 8) | pkt[pos + j];
        words[i] = w;
        pos += 8;
    }
    return count;
}

static int wait_for_response(int sock, double timeout_s, int rb_words,
                             unsigned short ethertype, u64 *words, int max_words)
{
    double deadline;
    unsigned char buf[65536];

    deadline = now_sec() + timeout_s;
    while (now_sec() < deadline) {
        double rem;
        struct timeval tv;
        fd_set rfds;
        int rc;
        ssize_t n;
        struct sockaddr_ll from;
        socklen_t fromlen;

        rem = deadline - now_sec();
        if (rem < 0.0)
            rem = 0.0;
        tv.tv_sec = (long)rem;
        tv.tv_usec = (long)((rem - (double)tv.tv_sec) * 1000000.0);

        FD_ZERO(&rfds);
        FD_SET(sock, &rfds);
        rc = select(sock + 1, &rfds, 0, 0, &tv);
        if (rc <= 0)
            break;

        fromlen = sizeof(from);
        n = recvfrom(sock, buf, sizeof(buf), 0, (struct sockaddr *)&from, &fromlen);
        if (n <= 0)
            continue;
        if (from.sll_pkttype == PACKET_OUTGOING)
            continue;
        if (n < 16)
            continue;
        if ((((unsigned int)buf[12] << 8) | buf[13]) != ethertype)
            continue;
        rc = parse_rx_words(buf, (int)n, rb_words, words, max_words);
        if (rc > 0)
            return rc;
    }
    return 0;
}

static float bf16_to_float(unsigned int bits)
{
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)(bits & 0xFFFF) << 16;
    return v.f;
}

static int argmax4(float f[4])
{
    int best;
    int i;
    best = 0;
    for (i = 1; i < 4; i++) {
        if (f[i] > f[best])
            best = i;
    }
    return best;
}

static void decode_ann(u64 *words, int n_words, int *expected, int expected_n)
{
    int sample;
    int correct;

    if (n_words < 4) {
        printf("\nANN decode skipped: no response with at least 4 readback words was captured\n");
        return;
    }

    printf("\nANN decode:\n");
    correct = 0;
    for (sample = 0; sample < 4; sample++) {
        u64 w = words[sample];
        unsigned int hi = (unsigned int)((w >> 32) & 0xFFFFFFFFULL);
        unsigned int lo = (unsigned int)(w & 0xFFFFFFFFULL);
        unsigned int hx[4];
        float fl[4];
        int pred;

        hx[0] = lo & 0xFFFF;
        hx[1] = (lo >> 16) & 0xFFFF;
        hx[2] = hi & 0xFFFF;
        hx[3] = (hi >> 16) & 0xFFFF;
        fl[0] = bf16_to_float(hx[0]);
        fl[1] = bf16_to_float(hx[1]);
        fl[2] = bf16_to_float(hx[2]);
        fl[3] = bf16_to_float(hx[3]);
        pred = argmax4(fl);

        printf("  sample %d: pred=%d:%-10s logits=%04X/%+7.2f %04X/%+7.2f %04X/%+7.2f %04X/%+7.2f",
               sample, pred, CLASS_NAMES[pred],
               hx[0], fl[0], hx[1], fl[1], hx[2], fl[2], hx[3], fl[3]);
        if (expected && sample < expected_n) {
            if (pred == expected[sample]) {
                correct++;
                printf(" expected=%d:%s PASS", expected[sample], CLASS_NAMES[expected[sample]]);
            } else {
                printf(" expected=%d:%s FAIL", expected[sample], CLASS_NAMES[expected[sample]]);
            }
        }
        printf("\n");
    }
    if (expected)
        printf("  accuracy: %d/%d\n", correct, expected_n < 4 ? expected_n : 4);
}

static int parse_expected(const char *s, int *expected, int max_expected)
{
    char tmp[256];
    char *tok;
    char *save;
    int n;

    if (!s)
        return 0;
    strncpy(tmp, s, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = 0;

    n = 0;
    tok = strtok_r(tmp, ",", &save);
    while (tok && n < max_expected) {
        expected[n++] = atoi(tok);
        tok = strtok_r(0, ",", &save);
    }
    return n;
}

static void usage(const char *prog)
{
    printf("Usage: %s (--frames FILE | --smoke cpu-add) --iface IFACE [options]\n", prog);
    printf("Options:\n");
    printf("  --dst-mac MAC            default ff:ff:ff:ff:ff:ff\n");
    printf("  --src-mac MAC            default interface MAC\n");
    printf("  --ethertype HEX          default 0x88B5\n");
    printf("  --gap SEC                default 0.02\n");
    printf("  --cpu-gap SEC            default 0.25\n");
    printf("  --timeout SEC            default 10\n");
    printf("  --rx-out FILE            save RX words\n");
    printf("  --pcap-out FILE          write Ethernet frames to pcap, no raw socket\n");
    printf("  --decode-ann             decode first 4-word ANN response\n");
    printf("  --expected-classes LIST  example 0,1,2,3\n");
    printf("  --dry-run                parse only\n");
}

int main(int argc, char **argv)
{
    const char *iface = 0;
    const char *frames_path = 0;
    const char *smoke = 0;
    const char *dst_mac_s = "ff:ff:ff:ff:ff:ff";
    const char *src_mac_s = 0;
    const char *rx_out_path = 0;
    const char *pcap_out_path = 0;
    const char *expected_s = 0;
    unsigned short ethertype = ETHERTYPE_SOC;
    double gap = 0.02;
    double cpu_gap = 0.25;
    double timeout_s = 10.0;
    int dry_run = 0;
    int decode = 0;
    int expected[4];
    int expected_n;
    frame_vec_t frames;
    unsigned char dst_mac[6];
    unsigned char src_mac[6];
    int i;
    int sock;
    int ifindex;
    int expect_count;
    int total_bytes;
    u64 last_rx[64];
    int last_rx_n;
    FILE *rx_out;

    static struct option long_opts[] = {
        {"iface", required_argument, 0, 1},
        {"frames", required_argument, 0, 2},
        {"smoke", required_argument, 0, 3},
        {"dst-mac", required_argument, 0, 4},
        {"src-mac", required_argument, 0, 5},
        {"ethertype", required_argument, 0, 6},
        {"gap", required_argument, 0, 7},
        {"cpu-gap", required_argument, 0, 8},
        {"timeout", required_argument, 0, 9},
        {"rx-out", required_argument, 0, 10},
        {"decode-ann", no_argument, 0, 11},
        {"expected-classes", required_argument, 0, 12},
        {"dry-run", no_argument, 0, 13},
        {"pcap-out", required_argument, 0, 14},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    memset(&frames, 0, sizeof(frames));
    expected_n = 0;
    last_rx_n = 0;
    rx_out = 0;

    while (1) {
        int c = getopt_long(argc, argv, "h", long_opts, 0);
        if (c == -1)
            break;
        switch (c) {
        case 1: iface = optarg; break;
        case 2: frames_path = optarg; break;
        case 3: smoke = optarg; break;
        case 4: dst_mac_s = optarg; break;
        case 5: src_mac_s = optarg; break;
        case 6: ethertype = (unsigned short)strtoul(optarg, 0, 0); break;
        case 7: gap = atof(optarg); break;
        case 8: cpu_gap = atof(optarg); break;
        case 9: timeout_s = atof(optarg); break;
        case 10: rx_out_path = optarg; break;
        case 11: decode = 1; break;
        case 12: expected_s = optarg; break;
        case 13: dry_run = 1; break;
        case 14: pcap_out_path = optarg; break;
        case 'h':
        default:
            usage(argv[0]);
            return c == 'h' ? 0 : 1;
        }
    }

    if ((frames_path ? 1 : 0) == (smoke ? 1 : 0)) {
        usage(argv[0]);
        fprintf(stderr, "provide exactly one of --frames or --smoke cpu-add\n");
        return 1;
    }
    if (smoke && strcmp(smoke, "cpu-add") != 0) {
        fprintf(stderr, "--smoke only supports cpu-add\n");
        return 1;
    }
    if (!dry_run && !pcap_out_path && !iface) {
        usage(argv[0]);
        fprintf(stderr, "--iface is required unless --dry-run or --pcap-out is used\n");
        return 1;
    }

    if (parse_mac(dst_mac_s, dst_mac) < 0) {
        fprintf(stderr, "bad --dst-mac\n");
        return 1;
    }

    if (frames_path)
        parse_frame_file(frames_path, &frames);
    else
        make_smoke(&frames);

    sock = -1;
    if (!dry_run && !pcap_out_path) {
        sock = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
        if (sock < 0)
            die("socket(AF_PACKET, SOCK_RAW)");
        ifindex = get_iface_index(sock, iface);
        if (src_mac_s) {
            if (parse_mac(src_mac_s, src_mac) < 0) {
                fprintf(stderr, "bad --src-mac\n");
                return 1;
            }
        } else if (get_iface_mac(sock, iface, src_mac) < 0) {
            die("SIOCGIFHWADDR");
        }
    } else {
        if (src_mac_s) {
            if (parse_mac(src_mac_s, src_mac) < 0) {
                fprintf(stderr, "bad --src-mac\n");
                return 1;
            }
        } else if (pcap_out_path && iface) {
            int tmp_sock = socket(AF_INET, SOCK_DGRAM, 0);
            if (tmp_sock >= 0 && get_iface_mac(tmp_sock, iface, src_mac) == 0) {
                close(tmp_sock);
            } else {
                if (tmp_sock >= 0)
                    close(tmp_sock);
                parse_mac("02:00:00:00:00:01", src_mac);
            }
        } else {
            parse_mac("02:00:00:00:00:01", src_mac);
        }
        ifindex = 0;
    }

    total_bytes = 0;
    expect_count = 0;
    for (i = 0; i < frames.n; i++) {
        int len = 14 + 2 + frames.v[i].n_words * 8;
        if (len < 64)
            len = 64;
        total_bytes += len;
        if (frames.v[i].expect)
            expect_count++;
    }
    printf("frames=%d expect_responses=%d bytes=%d dst=%s ethertype=0x%04X\n",
           frames.n, expect_count, total_bytes, dst_mac_s, ethertype);

    if (dry_run) {
        int show = frames.n < 5 ? frames.n : 5;
        for (i = 0; i < show; i++) {
            printf("  frame %d: words=%d expect=%d cpu=%d rb=%d\n",
                   frames.v[i].index, frames.v[i].n_words, frames.v[i].expect,
                   frames.v[i].cpu, frames.v[i].rb_words);
        }
        if (frames.n > 5)
            printf("  ... %d more frames\n", frames.n - 5);
        free_frames(&frames);
        if (sock >= 0)
            close(sock);
        return 0;
    }

    if (pcap_out_path) {
        FILE *pcap;
        double ts;

        pcap = fopen(pcap_out_path, "wb");
        if (!pcap)
            die(pcap_out_path);
        write_pcap_header(pcap);
        ts = 0.0;
        for (i = 0; i < frames.n; i++) {
            frame_t *f = &frames.v[i];
            unsigned char *pkt;
            int pkt_len;

            pkt = build_packet(f, dst_mac, src_mac, ethertype, &pkt_len);
            write_pcap_packet(pcap, pkt, pkt_len, ts);
            free(pkt);
            ts += f->cpu ? cpu_gap : gap;
        }
        fclose(pcap);
        printf("wrote pcap: %s (%d frames)\n", pcap_out_path, frames.n);
        free_frames(&frames);
        if (sock >= 0)
            close(sock);
        return 0;
    }

    if (rx_out_path) {
        rx_out = fopen(rx_out_path, "w");
        if (!rx_out)
            die(rx_out_path);
    }

    for (i = 0; i < frames.n; i++) {
        frame_t *f = &frames.v[i];
        unsigned char *pkt;
        int pkt_len;
        ssize_t sent;
        struct sockaddr_ll addr;

        pkt = build_packet(f, dst_mac, src_mac, ethertype, &pkt_len);
        memset(&addr, 0, sizeof(addr));
        addr.sll_family = AF_PACKET;
        addr.sll_ifindex = ifindex;
        addr.sll_halen = ETH_ALEN;
        memcpy(addr.sll_addr, dst_mac, 6);

        sent = sendto(sock, pkt, pkt_len, 0, (struct sockaddr *)&addr, sizeof(addr));
        free(pkt);
        if (sent < 0)
            die("sendto");
        printf("[TX frame=%04d] bytes=%d words=%d expect=%d\n",
               f->index, pkt_len, f->n_words, f->expect);

        if (f->expect) {
            int n;
            n = wait_for_response(sock, timeout_s, f->rb_words, ethertype, last_rx, 64);
            if (n <= 0) {
                printf("[RX frame=%04d] TIMEOUT after %.1fs\n", f->index, timeout_s);
                if (rx_out)
                    fclose(rx_out);
                free_frames(&frames);
                close(sock);
                return 2;
            }
            last_rx_n = n;
            printf("[RX frame=%04d]", f->index);
            if (rx_out)
                fprintf(rx_out, "RX %d", f->index);
            for (n = 0; n < last_rx_n; n++) {
                printf(" %016llX", (unsigned long long)last_rx[n]);
                if (rx_out)
                    fprintf(rx_out, " %016llX", (unsigned long long)last_rx[n]);
            }
            printf("\n");
            if (rx_out)
                fprintf(rx_out, "\n");
        } else {
            sleep_sec(f->cpu ? cpu_gap : gap);
        }
    }

    if (smoke && last_rx_n >= 2) {
        if (last_rx[0] == 0x000000140000000AULL &&
            last_rx[1] == 0x000000000000001EULL)
            printf("smoke cpu-add: PASS\n");
        else
            printf("smoke cpu-add: FAIL\n");
    }

    expected_n = parse_expected(expected_s, expected, 4);
    if (decode)
        decode_ann(last_rx, last_rx_n, expected_n ? expected : 0, expected_n);

    if (rx_out)
        fclose(rx_out);
    free_frames(&frames);
    close(sock);
    return 0;
}
