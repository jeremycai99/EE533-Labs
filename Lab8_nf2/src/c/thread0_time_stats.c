/*
 * thread0_time_stats.c — Time-Domain Statistics (8 features)
 * Target: ARMv4T + MUL/MLA, no FPU, no stdlib
 * Compile: arm-none-eabi-gcc -march=armv4t -O2 -ffreestanding -nostdlib -c
 *
 * Features: RMS, peak_abs, P2P, crest, MAV, zero_crossings, variance, skewness
 * All INT16 arithmetic with MUL.
 */

#define N       256
#define N_SHIFT 8

volatile short *const samples  = (volatile short *)0x0000;
volatile short *const feat_out = (volatile short *)0x0200;

static inline int abs16(int x) { return (x < 0) ? -x : x; }

static int isqrt(unsigned int x)
{
    unsigned int r = 0, bit = 1u << 30;
    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= r + bit) {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return (int)r;
}

static int idiv(int a, int b)
{
    if (b == 0) return 0;
    int neg = 0, q = 0, i;
    if (a < 0) { a = -a; neg ^= 1; }
    if (b < 0) { b = -b; neg ^= 1; }
    for (i = 15; i >= 0; i--) {
        if (a >= (b << i)) {
            a -= (b << i);
            q += (1 << i);
        }
    }
    return neg ? -q : q;
}

void thread0_time_stats(void)
{
    int sum     = 0;
    int sum_abs = 0;
    int sum_sq  = 0;
    int sum_cu  = 0;
    int peak_pos = -32768;
    int peak_neg =  32767;
    int peak_abs = 0;
    int zc = 0, prev_sign = 0;
    int i;

    for (i = 0; i < N; i++) {
        int x  = samples[i];
        int ax = abs16(x);

        sum     += x;
        sum_abs += ax;
        sum_sq  += (x * x) >> N_SHIFT;
        sum_cu  += ax * ((x * x) >> 16);

        if (x  > peak_pos) peak_pos = x;
        if (x  < peak_neg) peak_neg = x;
        if (ax > peak_abs) peak_abs = ax;

        int s = (x >= 0) ? 1 : 0;
        if (i > 0 && s != prev_sign) zc++;
        prev_sign = s;
    }

    int rms   = isqrt((unsigned int)sum_sq);
    int mav   = sum_abs >> N_SHIFT;
    int p2p   = peak_pos - peak_neg;
    int crest = (mav > 0) ? idiv(peak_abs, mav) : 0;
    int mean  = sum >> N_SHIFT;
    int var   = sum_sq - (mean * mean);

    feat_out[0] = (short)rms;
    feat_out[1] = (short)peak_abs;
    feat_out[2] = (short)p2p;
    feat_out[3] = (short)crest;
    feat_out[4] = (short)mav;
    feat_out[5] = (short)zc;
    feat_out[6] = (short)(var >> 4);
    feat_out[7] = (short)(sum_cu >> N_SHIFT);
}
