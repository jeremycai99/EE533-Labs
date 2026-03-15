/*
 * thread3_envelope.c — Envelope Analysis (8 features)
 * Target: ARMv4T + MUL/MLA, no FPU, no stdlib
 * Compile: arm-none-eabi-gcc -march=armv4t -O2 -ffreestanding -nostdlib -c
 *
 * Features: env_mean, env_peak, env_crest, env_var, 4 segment energies
 * Envelope: env[n] ≈ |x[n]| + |x[n]−x[n−2]|/2
 * MUL for env² (variance computation).
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

void thread3_envelope(void)
{
    int env_sum    = 0;
    int env_sum_sq = 0;
    int env_peak   = 0;
    int prev_env   = 0;
    int prev2      = 0;
    int env_zc     = 0;
    int seg_e[4]   = {0, 0, 0, 0};
    int i;

    for (i = 0; i < N; i++) {
        int x   = samples[i];
        int ax  = abs16(x);
        int diff = (i >= 2) ? abs16(x - samples[i - 2]) >> 1 : 0;
        int env  = ax + diff;

        env_sum    += env;
        env_sum_sq += (env * env) >> N_SHIFT;

        if (env > env_peak) env_peak = env;

        /* Direction reversal counting on envelope */
        if (i > 1) {
            int d1 = env - prev_env;
            int d0 = prev_env - prev2;
            if ((d1 ^ d0) < 0) env_zc++;
        }

        seg_e[i >> 6] += env;
        prev2    = prev_env;
        prev_env = env;
    }

    int env_mean  = env_sum >> N_SHIFT;
    int env_crest = (env_mean > 0) ? idiv(env_peak, env_mean) : 0;
    int env_var   = env_sum_sq - (env_mean * env_mean);

    feat_out[24] = (short)env_mean;
    feat_out[25] = (short)env_peak;
    feat_out[26] = (short)env_crest;
    feat_out[27] = (short)(env_var >> 4);
    feat_out[28] = (short)(seg_e[0] >> 6);
    feat_out[29] = (short)(seg_e[1] >> 6);
    feat_out[30] = (short)(seg_e[2] >> 6);
    feat_out[31] = (short)(seg_e[3] >> 6);
}
