/*
 * thread2_wavelet.c — 4-Level Haar Wavelet Decomposition (8 features)
 * Target: ARMv4T + MUL/MLA, no FPU, no stdlib
 * Compile: arm-none-eabi-gcc -march=armv4t -O2 -ffreestanding -nostdlib -c
 *
 * Features: 4 band energies + 4 band peaks
 * Purely ADD/SUB — MUL not needed for this thread.
 */

#define N 256

volatile short *const samples  = (volatile short *)0x0000;
volatile short *const feat_out = (volatile short *)0x0200;

/* Scratch buffer for in-place wavelet transform.
 * Located in CPU DMEM scratch region — adjust address as needed. */
static short wbuf[N];

static inline int abs16(int x) { return (x < 0) ? -x : x; }

void thread2_wavelet(void)
{
    int level, i, len = N;

    /* Copy samples into working buffer */
    for (i = 0; i < N; i++)
        wbuf[i] = samples[i];

    for (level = 0; level < 4; level++) {
        int half = len >> 1;
        int energy = 0, peak = 0;

        for (i = 0; i < half; i++) {
            int a = wbuf[i << 1];
            int b = wbuf[(i << 1) + 1];

            int approx = (a + b) >> 1;
            int detail = (a - b) >> 1;

            int ad = abs16(detail);
            energy += ad;
            if (ad > peak) peak = ad;

            wbuf[i] = (short)approx;  /* in-place for next level */
        }

        feat_out[16 + level]     = (short)(energy >> 4);  /* band energy */
        feat_out[16 + 4 + level] = (short)peak;           /* band peak   */
        len = half;
    }
}
