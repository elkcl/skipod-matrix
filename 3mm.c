#define _GNU_SOURCE
/* Include benchmark-specific header. */
#include "3mm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>
#include <sched.h>

#define MIN(_A, _B) (((_A) < (_B)) ? _A : _B)
#define MAX(_A, _B) (((_A) > (_B)) ? _A : _B)
#define ROUNDDOWN(_A, _N) ((_A) / (_N) * (_N))
#define ROUNDUP(_A, _N) (((_A) + (_N) - 1) / (_N) * (_N))

#ifdef VERIFY
#define Q(_v) #_v
#define STR(_v) Q(_v)
#define PASTER3(x,y,z) x ## y ## z
#define CONCAT3(x,y,z) PASTER3(x,y,z)

#define EPS 0.01f

#ifndef COMPUTE_DUMPS
extern int CONCAT3(_binary_MATRIX, NI, _dump_start);
float (*ans_mat)[NL] = (float (*)[NL]) &CONCAT3(_binary_MATRIX, NI, _dump_start);
#endif
#endif

double bench_t_start, bench_t_end;

static double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0) {
        printf("Error return from gettimeofday: %d", stat);
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
    bench_t_start = rtclock();
}

void bench_timer_stop() {
    bench_t_end = rtclock();
}

void bench_timer_print() {
    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static void init_array_orig(int ni, int nj, int nk, int nl, int nm, float A[restrict ni][nk], float B[restrict nk][nj],
                            float C[restrict nj][nm], float D[restrict nm][nl]) {
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nk; j++) {
            A[i][j] = (float) ((i * j + 1) % ni) / (5 * ni);
        }
    }
    for (int i = 0; i < nk; i++) {
        for (int j = 0; j < nj; j++) {
            B[i][j] = (float) ((i * (j + 1) + 2) % nj) / (5 * nj);
        }
    }
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nm; j++) {
            C[i][j] = (float) (i * (j + 3) % nl) / (5 * nl);
        }
    }
    for (int i = 0; i < nm; i++) {
        for (int j = 0; j < nl; j++) {
            D[i][j] = (float) ((i * (j + 2) + 2) % nk) / (5 * nk);
        }
    }
}

static void init_array(int ni, int nj, int nk, int nl, int nm, float A[restrict ni][nk], float B[restrict nk][nj],
                       float C[restrict nj][nm], float D[restrict nm][nl]) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nk; j++) {
            A[i][j] = (float) ((i * j + 1) % ni) / (5 * ni);
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nk; i++) {
        for (int j = 0; j < nj; j++) {
            B[i][j] = (float) ((i * (j + 1) + 2) % nj) / (5 * nj);
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nm; j++) {
            C[i][j] = (float) (i * (j + 3) % nl) / (5 * nl);
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nm; i++) {
        for (int j = 0; j < nl; j++) {
            D[i][j] = (float) ((i * (j + 2) + 2) % nk) / (5 * nk);
        }
    }
}

#ifdef LAPTOP
enum Caches { L1 = 48 * 1024 * 3 / 4, L2 = 1280 * 1024 * 3 / 4, L3 = 12 * 1024 * 1024 * 3 / 4 };
// update 6x16 submatrix C[x:x+6][y:y+16]
// using A[x:x+6][l:r] and B[l:r][y:y+16]
typedef float vec __attribute__((vector_size(32)));
void microkernel(int x, int y, int l, int r, int nk, int nj, const float A[restrict][nk], const vec B[restrict],
                 vec C[restrict]) {
    vec t00, t01, t10, t11, t20, t21, t30, t31, t40, t41, t50, t51;

    t00 = C[((x + 0) * nj + y) / 8 + 0];
    t01 = C[((x + 0) * nj + y) / 8 + 1];

    t10 = C[((x + 1) * nj + y) / 8 + 0];
    t11 = C[((x + 1) * nj + y) / 8 + 1];

    t20 = C[((x + 2) * nj + y) / 8 + 0];
    t21 = C[((x + 2) * nj + y) / 8 + 1];

    t30 = C[((x + 3) * nj + y) / 8 + 0];
    t31 = C[((x + 3) * nj + y) / 8 + 1];

    t40 = C[((x + 4) * nj + y) / 8 + 0];
    t41 = C[((x + 4) * nj + y) / 8 + 1];

    t50 = C[((x + 5) * nj + y) / 8 + 0];
    t51 = C[((x + 5) * nj + y) / 8 + 1];

    for (int k = l; k < r; k++) {
        vec a0 = (vec) {} + A[x + 0][k];
        t00 += a0 * B[(k * nj + y) / 8];
        t01 += a0 * B[(k * nj + y) / 8 + 1];

        vec a1 = (vec) {} + A[x + 1][k];
        t10 += a1 * B[(k * nj + y) / 8];
        t11 += a1 * B[(k * nj + y) / 8 + 1];

        vec a2 = (vec) {} + A[x + 2][k];
        t20 += a2 * B[(k * nj + y) / 8];
        t21 += a2 * B[(k * nj + y) / 8 + 1];

        vec a3 = (vec) {} + A[x + 3][k];
        t30 += a3 * B[(k * nj + y) / 8];
        t31 += a3 * B[(k * nj + y) / 8 + 1];

        vec a4 = (vec) {} + A[x + 4][k];
        t40 += a4 * B[(k * nj + y) / 8];
        t41 += a4 * B[(k * nj + y) / 8 + 1];

        vec a5 = (vec) {} + A[x + 5][k];
        t50 += a5 * B[(k * nj + y) / 8];
        t51 += a5 * B[(k * nj + y) / 8 + 1];
    }

    C[((x + 0) * nj + y) / 8 + 0] = t00;
    C[((x + 0) * nj + y) / 8 + 1] = t01;

    C[((x + 1) * nj + y) / 8 + 0] = t10;
    C[((x + 1) * nj + y) / 8 + 1] = t11;

    C[((x + 2) * nj + y) / 8 + 0] = t20;
    C[((x + 2) * nj + y) / 8 + 1] = t21;

    C[((x + 3) * nj + y) / 8 + 0] = t30;
    C[((x + 3) * nj + y) / 8 + 1] = t31;

    C[((x + 4) * nj + y) / 8 + 0] = t40;
    C[((x + 4) * nj + y) / 8 + 1] = t41;

    C[((x + 5) * nj + y) / 8 + 0] = t50;
    C[((x + 5) * nj + y) / 8 + 1] = t51;
}

void matmul(int ni_, int nk_, int nj_, const float A_[restrict ni_][nk_], const float B_[restrict nk_][nj_],
            float C_[restrict ni_][nj_], int ni, int nk, int nj, float A[restrict ni][nk], float B[restrict nk][nj],
            float C[restrict ni][nj]) {
    /* int ni = (ni_ + 5) / 6 * 6; */
    /* int nk = nk_; */
    /* int nj = (nj_ + 15) / 16 * 16; */
    /**/
    /* float (*A)[nk] = aligned_alloc(32, ni * sizeof(*A)); */
    /* memset(A, 0, ni * sizeof(*A)); */
    /* float (*B)[nj] = aligned_alloc(32, nk * sizeof(*B)); */
    /* memset(B, 0, nk * sizeof(*B)); */
    /* float (*C)[nj] = aligned_alloc(32, ni * sizeof(*C)); */
    /* memset(C, 0, ni * sizeof(*C)); */

    memset(A, 0, ni * sizeof(*A));
    memset(B, 0, nk * sizeof(*B));
    memset(C, 0, ni * sizeof(*C));
    for (int i = 0; i < ni_; ++i) {
        memcpy(&A[i], &A_[i], nk_ * sizeof(*A_[i]));
        memcpy(&C[i], &C_[i], nj_ * sizeof(*C_[i]));
    }
    for (int k = 0; k < nk_; ++k) {
        memcpy(&B[k], &B_[k], nj_ * sizeof(*B_[k]));
    }

    const int s3 = MAX(ROUNDDOWN(L3 / nk, 16), 1);
    const int s2 = MAX(ROUNDDOWN(L2 / nk, 6), 1);
    const int s1 = MAX(L1 / s3, 1);
    /* const int s3 = 64; */
    /* const int s2 = 120; */
    /* const int s1 = 240; */
    omp_set_num_threads(omp_get_max_threads() / 2);
    #pragma omp parallel for collapse(2) proc_bind(spread)
    for (int i3 = 0; i3 < nj; i3 += s3) {
        // now we are working with b[:][i3:i3+s3]
        for (int i2 = 0; i2 < ni; i2 += s2) {
            /* printf("External thread %3d is running on CPU %3d\n", omp_get_thread_num(), sched_getcpu()); */
            // now we are working with a[i2:i2+s2][:]
            for (int i1 = 0; i1 < nk; i1 += s1) {
                // now we are working with b[i1:i1+s1][i3:i3+s3]
                // this equates to updating c[i2:i2+s2][i3:i3+s3]
                // with [l:r] = [i1:i1+s1]
                #pragma omp parallel for collapse(2) proc_bind(close) num_threads(2)
                for (int x = i2; x < MIN(i2 + s2, ni); x += 6) {
                    for (int y = i3; y < MIN(i3 + s3, nj); y += 16) {
                        /* printf("Internal thread %3d is running on CPU %3d\n", omp_get_thread_num(), sched_getcpu()); */
                        microkernel(x, y, i1, MIN(i1 + s1, nk), nk, nj, A, (vec *) B, (vec *) C);
                    }
                }
            }
        }
    }

    /* for (int i = 0; i < ni; i += 6) { */
    /*     for (int j = 0; j < nj; j += 16) { */
    /*         microkernel(i, j, 0, nk, nk, nj, A, (vec *) B, (vec *) C); */
    /*     } */
    /* } */

    for (int i = 0; i < ni_; ++i) {
        memcpy(&C_[i], &C[i], nj_ * sizeof(*C_[i]));
    }

    // for (int i = 0; i < n; i++)
    //     memcpy(&_c[i * n], &c[i * ny], 4 * n);
    /* free(A); */
    /* free(B); */
    /* free(C); */
}
#else
#ifdef POLUS
enum Caches { L1 = 64 * 1024 * 3 / 4, L2 = 512 * 1024 * 3 / 4, L3 = 8192 * 1024 * 3 / 4 };
// update 8x16 submatrix C[x:x+8][y:y+16]
// using A[x:x+8][l:r] and B[l:r][y:y+16]
typedef float vec __attribute__((vector_size(16)));
void microkernel(int x, int y, int l, int r, int nk, int nj, const float A[restrict][nk], const vec B[restrict],
                 vec C[restrict]) {
    vec t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33, t40, t41, t42, t43, t50, t51,
        t52, t53, t60, t61, t62, t63, t70, t71, t72, t73;

    t00 = C[((x + 0) * nj + y) / 4 + 0];
    t01 = C[((x + 0) * nj + y) / 4 + 1];
    t02 = C[((x + 0) * nj + y) / 4 + 2];
    t03 = C[((x + 0) * nj + y) / 4 + 3];

    t10 = C[((x + 1) * nj + y) / 4 + 0];
    t11 = C[((x + 1) * nj + y) / 4 + 1];
    t12 = C[((x + 1) * nj + y) / 4 + 2];
    t13 = C[((x + 1) * nj + y) / 4 + 3];

    t20 = C[((x + 2) * nj + y) / 4 + 0];
    t21 = C[((x + 2) * nj + y) / 4 + 1];
    t22 = C[((x + 2) * nj + y) / 4 + 2];
    t23 = C[((x + 2) * nj + y) / 4 + 3];

    t30 = C[((x + 3) * nj + y) / 4 + 0];
    t31 = C[((x + 3) * nj + y) / 4 + 1];
    t32 = C[((x + 3) * nj + y) / 4 + 2];
    t33 = C[((x + 3) * nj + y) / 4 + 3];

    t40 = C[((x + 4) * nj + y) / 4 + 0];
    t41 = C[((x + 4) * nj + y) / 4 + 1];
    t42 = C[((x + 4) * nj + y) / 4 + 2];
    t43 = C[((x + 4) * nj + y) / 4 + 3];

    t50 = C[((x + 5) * nj + y) / 4 + 0];
    t51 = C[((x + 5) * nj + y) / 4 + 1];
    t52 = C[((x + 5) * nj + y) / 4 + 2];
    t53 = C[((x + 5) * nj + y) / 4 + 3];

    t60 = C[((x + 6) * nj + y) / 4 + 0];
    t61 = C[((x + 6) * nj + y) / 4 + 1];
    t62 = C[((x + 6) * nj + y) / 4 + 2];
    t63 = C[((x + 6) * nj + y) / 4 + 3];

    t70 = C[((x + 7) * nj + y) / 4 + 0];
    t71 = C[((x + 7) * nj + y) / 4 + 1];
    t72 = C[((x + 7) * nj + y) / 4 + 2];
    t73 = C[((x + 7) * nj + y) / 4 + 3];

    for (int k = l; k < r; k++) {
        vec a0 = (vec) {} + A[x + 0][k];
        t00 += a0 * B[(k * nj + y) / 4];
        t01 += a0 * B[(k * nj + y) / 4 + 1];
        t02 += a0 * B[(k * nj + y) / 4 + 2];
        t03 += a0 * B[(k * nj + y) / 4 + 3];

        vec a1 = (vec) {} + A[x + 1][k];
        t10 += a1 * B[(k * nj + y) / 4];
        t11 += a1 * B[(k * nj + y) / 4 + 1];
        t12 += a1 * B[(k * nj + y) / 4 + 2];
        t13 += a1 * B[(k * nj + y) / 4 + 3];

        vec a2 = (vec) {} + A[x + 2][k];
        t20 += a2 * B[(k * nj + y) / 4];
        t21 += a2 * B[(k * nj + y) / 4 + 1];
        t22 += a2 * B[(k * nj + y) / 4 + 2];
        t23 += a2 * B[(k * nj + y) / 4 + 3];

        vec a3 = (vec) {} + A[x + 3][k];
        t30 += a3 * B[(k * nj + y) / 4];
        t31 += a3 * B[(k * nj + y) / 4 + 1];
        t32 += a3 * B[(k * nj + y) / 4 + 2];
        t33 += a3 * B[(k * nj + y) / 4 + 3];

        vec a4 = (vec) {} + A[x + 4][k];
        t40 += a4 * B[(k * nj + y) / 4];
        t41 += a4 * B[(k * nj + y) / 4 + 1];
        t42 += a4 * B[(k * nj + y) / 4 + 2];
        t43 += a4 * B[(k * nj + y) / 4 + 3];

        vec a5 = (vec) {} + A[x + 5][k];
        t50 += a5 * B[(k * nj + y) / 4];
        t51 += a5 * B[(k * nj + y) / 4 + 1];
        t52 += a5 * B[(k * nj + y) / 4 + 2];
        t53 += a5 * B[(k * nj + y) / 4 + 3];

        vec a6 = (vec) {} + A[x + 6][k];
        t60 += a6 * B[(k * nj + y) / 4];
        t61 += a6 * B[(k * nj + y) / 4 + 1];
        t62 += a6 * B[(k * nj + y) / 4 + 2];
        t63 += a6 * B[(k * nj + y) / 4 + 3];

        vec a7 = (vec) {} + A[x + 7][k];
        t70 += a7 * B[(k * nj + y) / 4];
        t71 += a7 * B[(k * nj + y) / 4 + 1];
        t72 += a7 * B[(k * nj + y) / 4 + 2];
        t73 += a7 * B[(k * nj + y) / 4 + 3];
    }

    C[((x + 0) * nj + y) / 4 + 0] = t00;
    C[((x + 0) * nj + y) / 4 + 1] = t01;
    C[((x + 0) * nj + y) / 4 + 2] = t02;
    C[((x + 0) * nj + y) / 4 + 3] = t03;

    C[((x + 1) * nj + y) / 4 + 0] = t10;
    C[((x + 1) * nj + y) / 4 + 1] = t11;
    C[((x + 1) * nj + y) / 4 + 2] = t12;
    C[((x + 1) * nj + y) / 4 + 3] = t13;

    C[((x + 2) * nj + y) / 4 + 0] = t20;
    C[((x + 2) * nj + y) / 4 + 1] = t21;
    C[((x + 2) * nj + y) / 4 + 2] = t22;
    C[((x + 2) * nj + y) / 4 + 3] = t23;

    C[((x + 3) * nj + y) / 4 + 0] = t30;
    C[((x + 3) * nj + y) / 4 + 1] = t31;
    C[((x + 3) * nj + y) / 4 + 2] = t32;
    C[((x + 3) * nj + y) / 4 + 3] = t33;

    C[((x + 4) * nj + y) / 4 + 0] = t40;
    C[((x + 4) * nj + y) / 4 + 1] = t41;
    C[((x + 4) * nj + y) / 4 + 2] = t42;
    C[((x + 4) * nj + y) / 4 + 3] = t43;

    C[((x + 5) * nj + y) / 4 + 0] = t50;
    C[((x + 5) * nj + y) / 4 + 1] = t51;
    C[((x + 5) * nj + y) / 4 + 2] = t52;
    C[((x + 5) * nj + y) / 4 + 3] = t53;

    C[((x + 6) * nj + y) / 4 + 0] = t60;
    C[((x + 6) * nj + y) / 4 + 1] = t61;
    C[((x + 6) * nj + y) / 4 + 2] = t62;
    C[((x + 6) * nj + y) / 4 + 3] = t63;

    C[((x + 7) * nj + y) / 4 + 0] = t70;
    C[((x + 7) * nj + y) / 4 + 1] = t71;
    C[((x + 7) * nj + y) / 4 + 2] = t72;
    C[((x + 7) * nj + y) / 4 + 3] = t73;
}

void matmul(int ni_, int nk_, int nj_, const float A_[restrict ni_][nk_], const float B_[restrict nk_][nj_],
            float C_[restrict ni_][nj_], int ni, int nk, int nj, float A[restrict ni][nk], float B[restrict nk][nj],
            float C[restrict ni][nj]) {
    /* int ni = (ni_ + 5) / 6 * 6; */
    /* int nk = nk_; */
    /* int nj = (nj_ + 15) / 16 * 16; */
    /**/
    /* float (*A)[nk] = aligned_alloc(32, ni * sizeof(*A)); */
    /* memset(A, 0, ni * sizeof(*A)); */
    /* float (*B)[nj] = aligned_alloc(32, nk * sizeof(*B)); */
    /* memset(B, 0, nk * sizeof(*B)); */
    /* float (*C)[nj] = aligned_alloc(32, ni * sizeof(*C)); */
    /* memset(C, 0, ni * sizeof(*C)); */

    memset(A, 0, ni * sizeof(*A));
    memset(B, 0, nk * sizeof(*B));
    memset(C, 0, ni * sizeof(*C));
    for (int i = 0; i < ni_; ++i) {
        memcpy(&A[i], &A_[i], nk_ * sizeof(*A_[i]));
        memcpy(&C[i], &C_[i], nj_ * sizeof(*C_[i]));
    }
    for (int k = 0; k < nk_; ++k) {
        memcpy(&B[k], &B_[k], nj_ * sizeof(*B_[k]));
    }

    const int s3 = ROUNDDOWN(L3 / nk, 16);
    const int s2 = ROUNDDOWN(L2 / nk, 8);
    const int s1 = L1 / s3;
    /* const int s3 = 64; */
    /* const int s2 = 120; */
    /* const int s1 = 240; */

    omp_set_num_threads(ROUNDUP(omp_get_max_threads(), 8) / 8);
    #pragma omp parallel for collapse(2) proc_bind(spread)
    for (int i3 = 0; i3 < nj; i3 += s3) {
        // now we are working with b[:][i3:i3+s3]
        for (int i2 = 0; i2 < ni; i2 += s2) {
            // now we are working with a[i2:i2+s2][:]
            for (int i1 = 0; i1 < nk; i1 += s1) {
                // now we are working with b[i1:i1+s1][i3:i3+s3]
                // this equates to updating c[i2:i2+s2][i3:i3+s3]
                // with [l:r] = [i1:i1+s1]
                #pragma omp parallel for collapse(2) proc_bind(close) num_threads(8)
                for (int x = i2; x < MIN(i2 + s2, ni); x += 8) {
                    for (int y = i3; y < MIN(i3 + s3, nj); y += 16) {
                        microkernel(x, y, i1, MIN(i1 + s1, nk), nk, nj, A, (vec *) B, (vec *) C);
                    }
                }
            }
        }
    }

    /* for (int i = 0; i < ni; i += 6) { */
    /*     for (int j = 0; j < nj; j += 16) { */
    /*         microkernel(i, j, 0, nk, nk, nj, A, (vec *) B, (vec *) C); */
    /*     } */
    /* } */

    for (int i = 0; i < ni_; ++i) {
        memcpy(&C_[i], &C[i], nj_ * sizeof(*C_[i]));
    }

    // for (int i = 0; i < n; i++)
    //     memcpy(&_c[i * n], &c[i * ny], 4 * n);
    /* free(A); */
    /* free(B); */
    /* free(C); */
}
#endif
#endif

static void print_array(int ni, int nl, const float G[ni][nl]) {
    int i, j;

    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "G");
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            if ((i * ni + j) % 20 == 0) {
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "%0.2f ", G[i][j]);
        }
    }
    fprintf(stderr, "\nend   dump: %s\n", "G");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

#if defined(LAPTOP) || defined(POLUS)
static void kernel_3mm(int ni_, int nj_, int nk_, int nl_, int nm_, float E_[restrict ni_][nj_],
                       const float A_[restrict ni_][nk_], const float B_[restrict nk_][nj_],
                       float F_[restrict nj_][nl_], const float C_[restrict nj_][nm_],
                       const float D_[restrict nm_][nl_], float G_[restrict ni_][nl_], int ni, int nj, int nk, int nl,
                       int nm, float E[restrict ni_][nj_], float A[restrict ni_][nk_], float B[restrict nk_][nj_],
                       float F[restrict nj_][nl_], float C[restrict nj_][nm_], float D[restrict nm_][nl_],
                       float G[restrict ni_][nl_]) {
    matmul(ni_, nk_, nj_, A_, B_, E_, ni, nk, nj, A, B, E);
    matmul(nj_, nm_, nl_, C_, D_, F_, nj, nm, nl, C, D, F);
    matmul(ni_, nj_, nl_, E_, F_, G_, ni, nj, nl, E, F, G);
}
#else
#define kernel_3mm(_ni, _nj, _nk, _nl, _nm, _E, _A, _B, _F, _C, _D, _G) \
    kernel_3mm_orig((_ni), (_nj), (_nk), (_nl), (_nm), (_E), (_A), (_B), (_F), (_C), (_D), (_G))
#endif
static void kernel_3mm_orig(int ni, int nj, int nk, int nl, int nm, float E[restrict ni][nj],
                            const float A[restrict ni][nk], const float B[restrict nk][nj], float F[restrict nj][nl],
                            const float C[restrict nj][nm], const float D[restrict nm][nl], float G[restrict ni][nl]) {
    int i, j, k;

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            E[i][j] = 0.0f;
            for (k = 0; k < nk; ++k) {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            F[i][j] = 0.0f;
            for (k = 0; k < nm; ++k) {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            G[i][j] = 0.0f;
            for (k = 0; k < nj; ++k) {
                G[i][j] += E[i][k] * F[k][j];
            }
        }
    }
}
#ifdef VERIFY
bool verify(bool dump, int ni, int nj, int nk, int nl, int nm, const float E[restrict ni][nj],
            const float A[restrict ni][nk], const float B[restrict nk][nj], const float F[restrict nj][nl],
            const float C[restrict nj][nm], const float D[restrict nm][nl], const float G[restrict ni][nl]) {
    #ifdef COMPUTE_DUMPS
    float(*A_)[nk] = malloc(ni * sizeof(*A_));
    float(*B_)[nj] = malloc(nk * sizeof(*B_));
    float(*E_)[nj] = malloc(ni * sizeof(*E_));
    float(*C_)[nm] = malloc(nj * sizeof(*C_));
    float(*D_)[nl] = malloc(nm * sizeof(*D_));
    float(*F_)[nl] = malloc(nj * sizeof(*F_));
    float(*G_)[nl] = malloc(ni * sizeof(*G_));
    init_array_orig(ni, nj, nk, nl, nm, A_, B_, C_, D_);
    kernel_3mm_orig(ni, nj, nk, nl, nm, E_, A_, B_, F_, C_, D_, G_);
    if (dump) {
        int fd = creat("MATRIX" STR(NI), 0660);
        write(fd, G_, ni * sizeof(*G_));
        close(fd);
    }
    for (int i = 0; i < ni; ++i) {
        for (int l = 0; l < nl; ++l) {
            if (fabsf(G[i][l] - G_[i][l]) >= EPS) {
                if (dump) {
                    puts("CORRECT DUMP:");
                    /* print_array(ni, nl, G_); */
                    puts("WRONG DUMP:");
                    /* print_array(ni, nl, G); */
                }
                free((void *) E_);
                free((void *) A_);
                free((void *) B_);
                free((void *) F_);
                free((void *) C_);
                free((void *) D_);
                free((void *) G_);
                return false;
            }
        }
    }
    free((void *) E_);
    free((void *) A_);
    free((void *) B_);
    free((void *) F_);
    free((void *) C_);
    free((void *) D_);
    free((void *) G_);
    return true;
#else
    for (int i = 0; i < ni; ++i) {
        for (int l = 0; l < nl; ++l) {
            if (fabsf(G[i][l] - ans_mat[i][l]) >= EPS) {
                if (dump) {
                    puts("CORRECT DUMP:");
                    print_array(ni, nl, ans_mat);
                    puts("WRONG DUMP:");
                    print_array(ni, nl, G);
                }
                return false;
            }
        }
    }
    return true;
#endif
}
#endif

int main(int argc, char **argv) {
    const int ni_ = NI;
    const int nj_ = NJ;
    const int nk_ = NK;
    const int nl_ = NL;
    const int nm_ = NM;

    float(*E_)[ni_][nj_];
    E_ = (float(*)[ni_][nj_]) malloc((ni_) * (nj_) * sizeof(float));
    float(*A_)[ni_][nk_];
    A_ = (float(*)[ni_][nk_]) malloc((ni_) * (nk_) * sizeof(float));
    float(*B_)[nk_][nj_];
    B_ = (float(*)[nk_][nj_]) malloc((nk_) * (nj_) * sizeof(float));
    float(*F_)[nj_][nl_];
    F_ = (float(*)[nj_][nl_]) malloc((nj_) * (nl_) * sizeof(float));
    float(*C_)[nj_][nm_];
    C_ = (float(*)[nj_][nm_]) malloc((nj_) * (nm_) * sizeof(float));
    float(*D_)[nm_][nl_];
    D_ = (float(*)[nm_][nl_]) malloc((nm_) * (nl_) * sizeof(float));
    float(*G_)[ni_][nl_];
    G_ = (float(*)[ni_][nl_]) malloc((ni_) * (nl_) * sizeof(float));

#ifdef LAPTOP
    const int ni = ROUNDUP(ni_, 6);
    const int nj = ROUNDUP(nj_, 3 * 16);
    const int nk = nk_;
    const int nl = ROUNDUP(nl_, 16);
    const int nm = nm_;

    float(*A)[nk] = aligned_alloc(32, ni * sizeof(*A));
    float(*B)[nj] = aligned_alloc(32, nk * sizeof(*B));
    float(*E)[nj] = aligned_alloc(32, ni * sizeof(*E));
    float(*C)[nm] = aligned_alloc(32, nj * sizeof(*C));
    float(*D)[nl] = aligned_alloc(32, nm * sizeof(*D));
    float(*F)[nl] = aligned_alloc(32, nj * sizeof(*F));
    float(*G)[nl] = aligned_alloc(32, ni * sizeof(*G));
#else
#ifdef POLUS
    const int ni = ROUNDUP(ni_, 8);
    const int nj = ROUNDUP(nj_, 16);
    const int nk = nk_;
    const int nl = ROUNDUP(nl_, 16);
    const int nm = nm_;

    float(*A)[nk] = aligned_alloc(16, ni * sizeof(*A));
    float(*B)[nj] = aligned_alloc(16, nk * sizeof(*B));
    float(*E)[nj] = aligned_alloc(16, ni * sizeof(*E));
    float(*C)[nm] = aligned_alloc(16, nj * sizeof(*C));
    float(*D)[nl] = aligned_alloc(16, nm * sizeof(*D));
    float(*F)[nl] = aligned_alloc(16, nj * sizeof(*F));
    float(*G)[nl] = aligned_alloc(16, ni * sizeof(*G));
#endif
#endif

    init_array(ni_, nj_, nk_, nl_, nm_, *A_, *B_, *C_, *D_);

    bench_timer_start();

#if defined(LAPTOP) || defined(POLUS)
    kernel_3mm(ni_, nj_, nk_, nl_, nm_, *E_, *A_, *B_, *F_, *C_, *D_, *G_, ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
#else
    kernel_3mm(ni_, nj_, nk_, nl_, nm_, *E_, *A_, *B_, *F_, *C_, *D_, *G_);
#endif

    bench_timer_stop();
    bench_timer_print();
    #ifdef VERIFY
    bool dump = false;
    if ((argc > 1 && !strcmp(argv[1], "dump")) || (argc > 2 && !strcmp(argv[2], "dump"))) {
        dump = true;
        /* print_array(ni_, nl_, *G_); */
    }
    if ((argc > 1 && !strcmp(argv[1], "verify")) || (argc > 2 && !strcmp(argv[2], "verify"))) {
        if (verify(dump, ni_, nj_, nk_, nl_, nm_, *E_, *A_, *B_, *F_, *C_, *D_, *G_)) {
            puts("PASSED!");
        } else {
            puts("FAILED!");
        }
    }
    #endif


    free((void *) E_);
    free((void *) A_);
    free((void *) B_);
    free((void *) F_);
    free((void *) C_);
    free((void *) D_);
    free((void *) G_);

#if defined(LAPTOP) || defined(POLUS)
    free((void *) E);
    free((void *) A);
    free((void *) B);
    free((void *) F);
    free((void *) C);
    free((void *) D);
    free((void *) G);
#endif

    return 0;
}
