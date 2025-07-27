#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Inf 0x7fffffff

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// --- CUDA Device Functions ---
__device__ uint8_t ParityCheck_device(uint8_t *c, uint8_t *H, uint64_t n, uint64_t s) {
    for (size_t j = 0; j < s; j++){
        uint8_t syndrome = 0;
        for (size_t i = 0; i < n; i++)
            syndrome ^= (c[i] * H[j*n + i]);
        if (syndrome == 1) return 0;
    }
    return 1;
}

__device__ void HardDec_device(uint8_t *c, double *llr, uint64_t n) {
    for (size_t i = 0; i < n; i++)
        c[i] = (llr[i] > 0.0) ? 0 : 1;
}

__device__ int parity_device(uint8_t array[], uint64_t n) {
    int sum = 0;
    for (uint64_t i = 0; i < n; i++)
        sum += array[i];
    return sum % 2;
}

__device__ double prob_parity_device(int parity_cHD, double *absL, uint64_t n) {
    double prob_even = 1.0;
    for (uint64_t i = 0; i < n; i++) {
        prob_even *= (1.0 - 2.0 * exp(-absL[i]) / (1.0 + exp(-absL[i])));
    }
    prob_even = 0.5 * (1.0 + prob_even);
    return (parity_cHD == 0) ? prob_even : 1.0 - prob_even;
}

__device__ void AddTEP_device(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n) {
    for (size_t i = 0; i < n; i++)
        c[perm[i]] = cHD[perm[i]] ^ TEP[i];
}

__device__ double JacLog_device(double x) {
    if (x > 30) return x;
    if (x < -30) return 0.0;
    return log(1.0 + exp(x));
}

__device__ void QuickSort_device(double *a, size_t *perm, uint64_t n) {
    // Simple bubble sort for device (avoid recursion)
    for (uint64_t i = 0; i < n-1; i++) {
        for (uint64_t j = 0; j < n-i-1; j++) {
            if (a[j] > a[j+1]) {
                double temp = a[j];
                a[j] = a[j+1];
                a[j+1] = temp;

                size_t temp_perm = perm[j];
                perm[j] = perm[j+1];
                perm[j+1] = temp_perm;
            }
        }
    }
}

__device__ double getPM_HD_device(double *absL, uint64_t n) {
    double pm = 0;
    for(size_t i = 0; i < n; i++)
        pm += JacLog_device(-absL[i]);
    return pm;
}

__device__ double getPM_device(uint8_t *TEP, double *absL, double PM_HD, uint64_t n) {
    double pm = PM_HD;
    for(size_t i = 0; i < n; i++) {
        if (TEP[i] == 1)
            pm += (JacLog_device(absL[i]) - JacLog_device(-absL[i]));
    }
    return pm;
}

__device__ double getLConf_device(double *pNL, double P_notGuess, uint64_t cur_L, double *score, uint64_t s, uint8_t even) {
    double P_pos = 0.0;
    for(size_t i = 0; i <= cur_L; i++)
        P_pos += exp(-score[4*i+1]);
    if(even==1) s--;
    double P_neg = pow(2.0, -(double)s) * P_notGuess;
    pNL[0] = P_neg;
    return (P_pos + P_neg > 1e-9) ? (P_pos / (P_pos + P_neg)) : 1.0;
}

__device__ int32_t findMax_device(int32_t a, int32_t b) {
    return !(b > a) ? a : b;
}

__device__ void mountain_build_device(int32_t *u, int32_t k, int32_t w, int32_t W1, int32_t n1) {
    for(size_t i = k + 1; i < w; i++)
        u[i] = u[k];
    uint64_t W2 = W1;
    for(size_t i = 0; i < w; i++)
        W2 -= u[i];
    uint64_t q = (uint64_t)floor((double)W2 / (double)(n1 - u[k]));
    uint64_t r = W2 - q*(n1 - u[k]);
    if (q != 0) {
        for(size_t i = w-q; i < w; i++)
            u[i] = n1;
    }
    if (w > q)
        u[w-q-1] = u[w-q-1] + r;
}

__device__ void getAPP_device(uint64_t cur_L, double *score, double *APP) {
    if (cur_L == 0) return;
    double P_pos = 0.0;
    for(size_t i=0; i<cur_L; i++)
        P_pos += exp(-score[4*i+1]);
    if (P_pos < 1e-30) return;
    double den = score[4*(cur_L-1)+3] / P_pos;
    for(size_t i=0; i<cur_L; i++)
        APP[i] = exp(-score[4*i+1]) * den;
}

// Complete SOGRAND device implementation
__device__ void sogrand_main_logic_device(double* chat_list, double* s_list, double* T_val,
                                         double* curL_val, double* pNL_val, double* APP_list,
                                         double* llr, uint8_t* H_flat, int n, int s, int IC,
                                         uint64_t L, uint64_t Tmax, double thres, int even) {

    // Allocate local arrays (n=15 max)
    size_t perm[15];
    uint8_t cHD[15];
    uint8_t TEP[15];
    uint8_t c[15];
    double absL[15];
    int32_t u[15];
    int32_t d[15];
    int32_t D[15];

    for(size_t i = 0; i < n; i++) perm[i] = i;
    for(size_t i = 0; i < 4*L; i++) s_list[i] = 0;
    for(size_t i = 0; i < L; i++) APP_list[i] = 0;

    uint64_t cur_L = 0;
    HardDec_device(cHD, llr, n);
    uint8_t parity_cHD = parity_device(cHD, n);
    pNL_val[0] = 0.0;

    if (Tmax == 0) Tmax = Inf;

    for (size_t i = 0; i < n; i++) {
        TEP[i] = 0;
        absL[i] = fabs(llr[i]);
    }

    double P_notGuess = 1.0;
    if (even == 1) P_notGuess = prob_parity_device(parity_cHD, absL, n);

    double PM_HD = getPM_HD_device(absL, n);
    QuickSort_device(absL, perm, n);

    if (IC < 0) {
        if (round((double)n/2) > 1) {
            double beta = (absL[(uint64_t)round((double)n/2) - 1] - absL[0]) / (round((double)n/2) - 1);
            IC = (beta > 1e-9) ? findMax_device((int32_t)round(absL[0]/beta - 1), 0) : 0;
        } else {
            IC = 0;
        }
    }

    AddTEP_device(c, cHD, TEP, perm, n);
    T_val[0] = 1;
    if (parity_cHD == 0 || even == 0)
        P_notGuess -= exp(-getPM_device(TEP, absL, PM_HD, n));

    if (ParityCheck_device(c, H_flat, n, s) == 1) {
        double pm = getPM_device(TEP, absL, PM_HD, n);
        APP_list[0] = pm;
        for(size_t i=0; i<n; i++) chat_list[i] = c[i];
        s_list[0] = pm;
        s_list[1] = pm;
        s_list[2] = T_val[0];
        s_list[3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
        cur_L++;
        if (even == 1) P_notGuess -= exp(-pm);
        if ((s_list[3] > thres) || (cur_L == L)) {
            getAPP_device(cur_L, s_list, APP_list);
            curL_val[0] = cur_L;
            return;
        }
    }

    int32_t wt = IC + 1;
    while ((cur_L < L) && (T_val[0] < Tmax)) {
        int32_t w = 1;
        double temp_sqrt = pow(1+2*((double)n+(double)IC), 2.0) - 8*wt;
        if (temp_sqrt >= 0) {
            w = findMax_device(1, (int32_t)ceil((1+2*((double)n+(double)IC) - sqrt(temp_sqrt))/2.0));
        } else {
            wt++;
            continue;
        }

        if (even == 1 && (w % 2 != parity_cHD)) w++;

        while (w <= n) {
            int32_t W = wt - IC*w;
            if (W < w*(w+1)/2) break;

            int32_t W1 = W - w*(w+1)/2;
            int32_t n1 = n - w;
            for (size_t i = 0; i < w; i++) u[i] = 0;

            mountain_build_device(u, 0, w, W1, n1);

            int mountain_iter_guard = 0;
            do {
                for (size_t i = 0; i < n; i++) TEP[i] = 0;
                for (size_t i = 0; i < w; i++) TEP[i+u[i]] = 1;
                AddTEP_device(c, cHD, TEP, perm, n);
                T_val[0]++;
                if (parity_cHD == 0 || even == 0)
                    P_notGuess -= exp(-getPM_device(TEP, absL, PM_HD, n));

                if (ParityCheck_device(c, H_flat, n, s) == 1) {
                    double pm = getPM_device(TEP, absL, PM_HD, n);
                    APP_list[cur_L] = pm;
                    for(size_t i=0; i<n; i++) chat_list[cur_L*n + i] = c[i];
                    s_list[4*cur_L] = pm;
                    s_list[4*cur_L+1] = pm;
                    s_list[4*cur_L+2] = T_val[0];
                    s_list[4*cur_L+3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
                    cur_L++;
                    if (even == 1) P_notGuess -= exp(-pm);
                    if ((s_list[4*(cur_L-1)+3] > thres) || (cur_L == L)) {
                        getAPP_device(cur_L, s_list, APP_list);
                        curL_val[0] = cur_L;
                        return;
                    }
                }

                for (size_t i = 0; i < w - 1; i++) d[i] = u[i+1] - u[i];
                d[w-1] = 0;
                D[w-1] = d[w-1];
                for (int i = w - 2; i >= 0; i--) D[i] = D[i+1] + d[i];

                if (D[0] < 2) break;

                int32_t k_mt = 0;
                for (int i = w-1; i > 0; i--) {
                    if (D[i] >= 2) { k_mt = i; break; }
                }
                u[k_mt]++;
                mountain_build_device(u, k_mt, w, W1, n1);

            } while (++mountain_iter_guard < 100000);

            w++;
            if (even == 1 && (w % 2 != parity_cHD)) w++;
        }
        wt++;
    }

    curL_val[0] = cur_L;
    getAPP_device(cur_L, s_list, APP_list);
}

// SOGRAND_bitSO device function
__device__ void SOGRAND_bitSO_device(double* L_APP, double* L_E, int* N_guess, double* llr,
                                    uint8_t* H_matrix, int n, int k, int L, uint64_t Tmax,
                                    double thres, int even) {

    // Allocate local memory for SOGRAND (fixed sizes for n=15, L=3)
    double chat_list[45]; // n * L = 15 * 3
    double s_list[12];    // 4 * L = 4 * 3
    double T_val, curL_val, pNL_val;
    double APP_list[3];   // L = 3

    sogrand_main_logic_device(chat_list, s_list, &T_val, &curL_val, &pNL_val, APP_list,
                             llr, H_matrix, n, n - k, -1, L, Tmax, thres, even);

    int curL = (int)curL_val;
    if (curL == 0) {
        for(int i=0; i<n; ++i) L_APP[i] = llr[i];
        for(int i=0; i<n; ++i) L_E[i] = 0;
    } else {
        double PM[3]; // L = 3
        for(int i=0; i<curL; ++i) PM[i] = s_list[4*i + 1];
        double p_notinlist = fmax(pNL_val, 1e-9);

        double pp1[15], pp0[15]; // n = 15
        for(int i=0; i<n; ++i) {
            pp1[i] = 1.0 / (1.0 + exp(llr[i]));
            pp0[i] = 1.0 - pp1[i];
            pp1[i] = fmax(pp1[i], 1e-9); pp1[i] = fmin(pp1[i], 1.0 - 1e-9);
            pp0[i] = fmax(pp0[i], 1e-9); pp0[i] = fmin(pp0[i], 1.0 - 1e-9);
        }

        double p[3]; // L = 3
        for(int i=0; i<curL; ++i) p[i] = exp(-PM[i]);

        double p1[15], p0[15]; // n = 15
        for(int i=0; i<n; ++i) {
            p1[i] = 0.0;
            p0[i] = 0.0;
        }

        for(int i=0; i<n; ++i) {
            for(int j=0; j<curL; ++j) {
                p1[i] += chat_list[i + j*n] * p[j];
                p0[i] += (1.0 - chat_list[i + j*n]) * p[j];
            }
        }

        for(int i=0; i<n; ++i) {
            p0[i] += p_notinlist * pp0[i];
            p1[i] += p_notinlist * pp1[i];
        }

        for(int i=0; i<n; ++i) {
            L_APP[i] = log(fmax(p0[i], 1e-30)) - log(fmax(p1[i], 1e-30));
            L_E[i] = L_APP[i] - llr[i];
        }
    }

    if (N_guess) *N_guess = (int)T_val;
}

// --- CUDA Kernels for Parallel Decoding ---
__global__ void sogrand_columns_kernel(double* input_data, double* L_APP_data, double* L_E_data,
                                      uint8_t* H_flat, int n, int k, int L, uint64_t Tmax,
                                      double thres, int even, int* NG_total) {
    int slice = blockIdx.x;
    int col = threadIdx.x;

    if (slice >= n || col >= n) return;

    // Each thread processes one column in one slice
    double vec_in[15], vec_L_APP[15], vec_L_E_vec[15]; // n=15
    int N_guess = 0;

    // Extract column data - FIXED INDEX CALCULATION
    for(int row = 0; row < n; row++) {
        int idx = slice * n * n + row * n + col;
        vec_in[row] = input_data[idx];
    }

    // Call SOGRAND
    SOGRAND_bitSO_device(vec_L_APP, vec_L_E_vec, &N_guess, vec_in, H_flat, n, k, L, Tmax, thres, even);

    // Store results back - FIXED INDEX CALCULATION
    for(int row = 0; row < n; row++) {
        int idx = slice * n * n + row * n + col;
        L_APP_data[idx] = vec_L_APP[row];
        L_E_data[idx] = vec_L_E_vec[row];
    }

    // Atomic add for NG count (thread-safe)
    atomicAdd(NG_total, N_guess);
}

__global__ void sogrand_rows_kernel(double* input_data, double* L_APP_data, double* L_E_data,
                                   uint8_t* H_flat, int n, int k, int L, uint64_t Tmax,
                                   double thres, int even, int* NG_total) {
    int slice = blockIdx.x;
    int row = threadIdx.x;

    if (slice >= n || row >= n) return;

    double vec_in[15], vec_L_APP[15], vec_L_E_vec[15];
    int N_guess = 0;

    // Extract row data - FIXED INDEX CALCULATION
    for(int col = 0; col < n; col++) {
        int idx = slice * n * n + row * n + col;
        vec_in[col] = input_data[idx];
    }

    SOGRAND_bitSO_device(vec_L_APP, vec_L_E_vec, &N_guess, vec_in, H_flat, n, k, L, Tmax, thres, even);

    // Store results back - FIXED INDEX CALCULATION
    for(int col = 0; col < n; col++) {
        int idx = slice * n * n + row * n + col;
        L_APP_data[idx] = vec_L_APP[col];
        L_E_data[idx] = vec_L_E_vec[col];
    }

    atomicAdd(NG_total, N_guess);
}

__global__ void sogrand_slices_kernel(double* input_data, double* L_APP_data, double* L_E_data,
                                     uint8_t* H_flat, int n, int k, int L, uint64_t Tmax,
                                     double thres, int even, int* NG_total) {
    int col = blockIdx.x;
    int row = threadIdx.x;

    if (col >= n || row >= n) return;

    double vec_in[15], vec_L_APP[15], vec_L_E_vec[15];
    int N_guess = 0;

    // Extract slice data - FIXED INDEX CALCULATION
    for(int slice_idx = 0; slice_idx < n; slice_idx++) {
        int idx = slice_idx * n * n + row * n + col;
        vec_in[slice_idx] = input_data[idx];
    }

    SOGRAND_bitSO_device(vec_L_APP, vec_L_E_vec, &N_guess, vec_in, H_flat, n, k, L, Tmax, thres, even);

    // Store results back - FIXED INDEX CALCULATION
    for(int slice_idx = 0; slice_idx < n; slice_idx++) {
        int idx = slice_idx * n * n + row * n + col;
        L_APP_data[idx] = vec_L_APP[slice_idx];
        L_E_data[idx] = vec_L_E_vec[slice_idx];
    }

    atomicAdd(NG_total, N_guess);
}

// Early termination kernel
__global__ void early_termination_kernel(double* L_APP_data, int* G_flat, int n, int k, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return; // Only one thread does this computation

    // Allocate shared memory for tensors
    extern __shared__ int shared_mem[];
    int* c_HD_data = shared_mem;
    int* c_test_data = &shared_mem[n*n*n];

    // Hard decision
    for(int i = 0; i < n*n*n; i++) {
        c_HD_data[i] = (L_APP_data[i] > 0) ? 0 : 1;
    }

    // Initialize c_test with systematic part
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = 0; col < k; col++) {
                int src_idx = slice * n * n + row * n + col;
                int dst_idx = slice * n * n + row * n + col;
                c_test_data[dst_idx] = c_HD_data[src_idx];
            }
        }
    }

    // Encode rows
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = k; col < n; col++) {
                int parity_val = 0;
                for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
                    int msg_idx = slice * n * n + row * n + msg_bit_idx;
                    parity_val += c_test_data[msg_idx] * G_flat[msg_bit_idx * n + col];
                }
                int col_idx = slice * n * n + row * n + col;
                c_test_data[col_idx] = parity_val % 2;
            }
        }
    }

    // Encode columns
    for (int slice = 0; slice < k; slice++) {
        for (int col = 0; col < n; col++) {
            for (int row = k; row < n; row++) {
                int parity_val = 0;
                for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
                    int msg_idx = slice * n * n + msg_bit_idx * n + col;
                    parity_val += c_test_data[msg_idx] * G_flat[msg_bit_idx * n + row];
                }
                int row_idx = slice * n * n + row * n + col;
                c_test_data[row_idx] = parity_val % 2;
            }
        }
    }

    // Encode slices
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for (int slice = k; slice < n; slice++) {
                int parity_val = 0;
                for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
                    int msg_idx = msg_bit_idx * n * n + row * n + col;
                    parity_val += c_test_data[msg_idx] * G_flat[msg_bit_idx * n + slice];
                }
                int slice_idx = slice * n * n + row * n + col;
                c_test_data[slice_idx] = parity_val % 2;
            }
        }
    }

    // Check if c_test matches c_HD
    int match = 1;
    for(int i = 0; i < n*n*n; i++) {
        if (c_test_data[i] != c_HD_data[i]) {
            match = 0;
            break;
        }
    }
    *result = match;
}

// --- Host Functions ---
int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));
    for (int i = 0; i < len; i++) poly[i] = (dec_val >> (len - 1 - i)) & 1;
    poly[len] = 1;
    return poly;
}

void getGH_sys_CRC(int n, int k, int** G, int** H) {
    const char* hex_poly = NULL;
    int r = n - k;

    if (r == 3) hex_poly = "0x5";
    else if (r == 4) hex_poly = "0x9";
    else if (r == 5 && k <= 10) hex_poly = "0x15";
    else if (r == 5 && k <= 26) hex_poly = "0x12";
    else if (r == 6 && k <= 25) hex_poly = "0x23";
    else if (r == 6 && k <= 57) hex_poly = "0x33";
    else {
        fprintf(stderr, "Error: (n, k) = (%d, %d) is not supported.\n", n, k);
        exit(1);
    }

    int poly_len;
    int* poly = koopman2matlab(hex_poly, &poly_len);

    int** P = (int**)malloc(k * sizeof(int*));
    for(int i=0; i<k; ++i) P[i] = (int*)malloc(r * sizeof(int));
    int* msg_poly = (int*)calloc(k + r, sizeof(int));

    for (int i = 0; i < k; i++) {
        memset(msg_poly, 0, (k + r) * sizeof(int));
        msg_poly[i] = 1;

        for (int j = 0; j < k; j++) {
            if (msg_poly[j] == 1) {
                for (int l = 0; l < poly_len; l++) {
                    msg_poly[j + l] ^= poly[l];
                }
            }
        }
        for (int j = 0; j < r; j++) P[i][j] = msg_poly[k + j];
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) G[i][j] = (i == j) ? 1 : 0;
        for (int j = 0; j < r; j++) G[i][k + j] = P[i][j];
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < k; j++) H[i][j] = P[j][i];
        for (int j = 0; j < r; j++) H[i][k + j] = (i == j) ? 1 : 0;
    }

    free(poly);
    free(msg_poly);
    for(int i=0; i<k; ++i) free(P[i]);
    free(P);
}

int early_termination_cuda(double* L_APP_data, int** G, int n, int k) {
    // Flatten G matrix for GPU
    int* G_flat;
    CUDA_CHECK(cudaMallocManaged(&G_flat, k * n * sizeof(int)));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            G_flat[i * n + j] = G[i][j];
        }
    }

    int* result;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));

    // Calculate shared memory size needed
    size_t shared_mem_size = 2 * n * n * n * sizeof(int);

    early_termination_kernel<<<1, 1, shared_mem_size>>>(L_APP_data, G_flat, n, k, result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int ret = *result;
    CUDA_CHECK(cudaFree(G_flat));
    CUDA_CHECK(cudaFree(result));
    return ret;
}

void hard_decision(double* llr, int* bits, int length) {
    int k = 10; // hardcoded for now
    int n = 15;
    int bit_idx = 0;

    int expected_length = k * k * k;
    if (length != expected_length) {
        fprintf(stderr, "Warning: hard_decision length mismatch\n");
    }

    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = 0; col < k; col++) {
                int tensor_idx = slice * n * n + row * n + col;
                bits[bit_idx++] = (llr[tensor_idx] > 0) ? 0 : 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_llr_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    const int n = 15;
    const int k = 10;
    const int codeword_block_size = n * n * n;
    const int message_block_size = k * k * k;

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));

    int** G = (int**)malloc(k * sizeof(int*));
    int** H = (int**)malloc((n-k) * sizeof(int*));
    for(int i = 0; i < k; i++) G[i] = (int*)malloc(n * sizeof(int));
    for(int i = 0; i < (n-k); i++) H[i] = (int*)malloc(n * sizeof(int));
    getGH_sys_CRC(n, k, G, H);

    // Flatten H matrix for GPU
    uint8_t* H_flat;
    CUDA_CHECK(cudaMallocManaged(&H_flat, (n-k) * n * sizeof(uint8_t)));
    for (int i = 0; i < (n-k); i++) {
        for (int j = 0; j < n; j++) {
            H_flat[i*n + j] = (uint8_t)H[i][j];
        }
    }

    // Check if code is even
    int even = 1;
    for (int i = 0; i < k; i++) {
        int row_sum = 0;
        for (int j = 0; j < n; j++) {
            row_sum += G[i][j];
        }
        if (row_sum % 2 != 0) {
            even = 0;
            break;
        }
    }

    const int L = 3;
    const int Imax = 30;
    const uint64_t Tmax = UINT64_MAX;
    const double p_ET = 1e-5;
    const double thres = 1.0 - p_ET;

    double alpha[100];
    for(int i = 0; i < 100; i++) alpha[i] = 0.7;

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("CUDA Decoding %s to %s using cubic code (n=%d, k=%d)...\n", input_filename, output_filename, n, k);

    double llr_buffer[codeword_block_size];
    int bit_buffer[message_block_size];
    unsigned char byte_out = 0;
    int bit_count_out = 0;

    int total_NG = 0;
    int total_NG_p = 0;
    double total_iterations = 0;
    int block_count = 0;

    // GPU memory for NG counting
    int* d_NG_total;
    CUDA_CHECK(cudaMallocManaged(&d_NG_total, sizeof(int)));

    while (fread(llr_buffer, sizeof(double), codeword_block_size, fin) == codeword_block_size) {
        block_count++;

        // Allocate unified memory tensors
        double* L_channel;
        double* L_APP;
        double* L_E;
        double* L_A;
        double* input;

        CUDA_CHECK(cudaMallocManaged(&L_channel, codeword_block_size * sizeof(double)));
        CUDA_CHECK(cudaMallocManaged(&L_APP, codeword_block_size * sizeof(double)));
        CUDA_CHECK(cudaMallocManaged(&L_E, codeword_block_size * sizeof(double)));
        CUDA_CHECK(cudaMallocManaged(&L_A, codeword_block_size * sizeof(double)));
        CUDA_CHECK(cudaMallocManaged(&input, codeword_block_size * sizeof(double)));

        memcpy(L_channel, llr_buffer, codeword_block_size * sizeof(double));
        CUDA_CHECK(cudaMemset(L_E, 0, codeword_block_size * sizeof(double)));

        double n_iter = 0;
        int NG = 0;
        int NG_p = 0;

        for (int iter = 1; iter <= Imax; iter++) {
            // Columns processing
            n_iter += 0.5;
            *d_NG_total = 0;

            // L_A = alpha * L_E
            for(int idx = 0; idx < codeword_block_size; idx++) {
                L_A[idx] = alpha[2*iter-2] * L_E[idx];
                input[idx] = L_channel[idx] + L_A[idx];
            }

            // Launch CUDA kernel for columns (avoid race conditions with proper grid/block sizes)
            dim3 grid_cols(n, 1, 1);
            dim3 block_cols(n, 1, 1);
            sogrand_columns_kernel<<<grid_cols, block_cols>>>(
                input, L_APP, L_E, H_flat, n, k, L, Tmax, thres, even, d_NG_total);
            CUDA_CHECK(cudaDeviceSynchronize());

            NG += *d_NG_total;

            if (early_termination_cuda(L_APP, G, n, k)) break;

            // Rows processing
            n_iter += 0.5;
            *d_NG_total = 0;

            // Update L_A and input
            for(int idx = 0; idx < codeword_block_size; idx++) {
                L_A[idx] = alpha[2*iter-1] * L_E[idx];
                input[idx] = L_channel[idx] + L_A[idx];
            }

            // Launch CUDA kernel for rows
            dim3 grid_rows(n, 1, 1);
            dim3 block_rows(n, 1, 1);
            sogrand_rows_kernel<<<grid_rows, block_rows>>>(
                input, L_APP, L_E, H_flat, n, k, L, Tmax, thres, even, d_NG_total);
            CUDA_CHECK(cudaDeviceSynchronize());

            NG += *d_NG_total;

            if (early_termination_cuda(L_APP, G, n, k)) break;

            // Slices processing
            n_iter += 0.5;
            *d_NG_total = 0;

            // Update input (L_A same as previous)
            for(int idx = 0; idx < codeword_block_size; idx++) {
                input[idx] = L_channel[idx] + L_A[idx];
            }

            // Launch CUDA kernel for slices
            dim3 grid_slices(n, 1, 1);
            dim3 block_slices(n, 1, 1);
            sogrand_slices_kernel<<<grid_slices, block_slices>>>(
                input, L_APP, L_E, H_flat, n, k, L, Tmax, thres, even, d_NG_total);
            CUDA_CHECK(cudaDeviceSynchronize());

            NG += *d_NG_total;

            if (early_termination_cuda(L_APP, G, n, k)) break;
        }

        total_NG += NG;
        total_iterations += n_iter;

        // Hard decode and output
        hard_decision(L_APP, bit_buffer, message_block_size);

        for (int i = 0; i < message_block_size; i++) {
            byte_out = (byte_out << 1) | bit_buffer[i];
            bit_count_out++;
            if (bit_count_out == 8) {
                fwrite(&byte_out, 1, 1, fout);
                byte_out = 0;
                bit_count_out = 0;
            }
        }

        // Cleanup tensors
        CUDA_CHECK(cudaFree(L_channel));
        CUDA_CHECK(cudaFree(L_APP));
        CUDA_CHECK(cudaFree(L_E));
        CUDA_CHECK(cudaFree(L_A));
        CUDA_CHECK(cudaFree(input));
    }

    if (bit_count_out > 0) {
        byte_out <<= (8 - bit_count_out);
        fwrite(&byte_out, 1, 1, fout);
    }

    printf("CUDA Decoding complete. %d block(s) decoded.\n", block_count);
    printf("Average iterations per block: %.2f\n", total_iterations / block_count);
    printf("Average NG per block: %.2f\n", (double)total_NG / block_count);
    printf("Average NG per info bit: %.2f\n", (double)total_NG / (block_count * k * k * k));

    // Cleanup
    fclose(fin);
    fclose(fout);
    for(int i = 0; i < k; i++) free(G[i]);
    for(int i = 0; i < (n-k); i++) free(H[i]);
    free(G);
    free(H);
    CUDA_CHECK(cudaFree(H_flat));
    CUDA_CHECK(cudaFree(d_NG_total));

    return 0;
}
