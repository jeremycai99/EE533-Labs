/*
 * thread1_spectral.c — Spectral Features (8 features)
 * Target: ARMv4T + MUL/MLA, no FPU, no stdlib
 * Compile: arm-none-eabi-gcc -march=armv4t -O2 -ffreestanding -nostdlib -c
 *
 * Features: centroid, spread, 4 band energies, flatness, rolloff
 * Walsh-like spectral decomposition, MUL for weighted stats.
 *
 * NOTE: All half-periods are powers of 2, so we use >> (shift)
 * instead of / (division) to avoid __aeabi_idiv dependency.
 */

#define N       256
#define N_SHIFT 8

volatile short *const samples  = (volatile short *)0x0000;
volatile short *const feat_out = (volatile short *)0x0200;

static inline int abs16(int x) { return (x < 0) ? -x : x; }

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

void thread1_spectral(void)
{
    /*
     * Walsh bins k=2, k=8, k=32, k=128
     * Half-periods:  64,  16,   4,   1
     * Shift amounts:  6,   4,   2,   0   (log2 of half-period)
     *
     * phase = i >> shift  replaces  phase = i / h
     * This compiles to a single LSR — no division, no __aeabi_idiv.
     */
    static const int hp_shift[4] = {6, 4, 2, 0};
    int band_e[4];
    int b, i;

    for (b = 0; b < 4; b++) {
        int acc_p = 0, acc_n = 0;
        int sh = hp_shift[b];
        for (i = 0; i < N; i++) {
            int ax = abs16((int)samples[i]);
            if ((i >> sh) & 1) acc_n += ax;
            else               acc_p += ax;
        }
        band_e[b] = abs16(acc_p - acc_n) >> 4;
    }

    /* Spectral centroid = Σ(k·E[k]) / Σ(E[k])  (Q4) */
    int num = 0, den = 0;
    for (b = 0; b < 4; b++) {
        num += b * band_e[b];
        den += band_e[b];
    }
    int centroid = (den > 0) ? idiv(num << 4, den) : 0;

    /* Spectral spread = Σ((k−μ)²·E[k]) / Σ(E[k]) */
    int spread_num = 0;
    for (b = 0; b < 4; b++) {
        int d = (b << 4) - centroid;
        spread_num += d * d * band_e[b] >> 8;
    }
    int spread = (den > 0) ? idiv(spread_num, den) : 0;

    /* Rolloff: first bin accumulating ≥ 87.5% of total */
    int thresh = (den * 7) >> 3;
    int cumul = 0, rolloff = 3;
    for (b = 0; b < 4; b++) {
        cumul += band_e[b];
        if (cumul >= thresh) { rolloff = b; break; }
    }

    /* Flatness proxy: min/max band energy (Q8) */
    int emin = band_e[0], emax = band_e[0];
    for (b = 1; b < 4; b++) {
        if (band_e[b] < emin) emin = band_e[b];
        if (band_e[b] > emax) emax = band_e[b];
    }
    int flat = (emax > 0) ? idiv(emin << 8, emax) : 0;

    feat_out[8]  = (short)centroid;
    feat_out[9]  = (short)spread;
    feat_out[10] = (short)band_e[0];
    feat_out[11] = (short)band_e[1];
    feat_out[12] = (short)band_e[2];
    feat_out[13] = (short)band_e[3];
    feat_out[14] = (short)flat;
    feat_out[15] = (short)rolloff;
}