#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <unistd.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Inf 0x7fffffff
#define MAX_STREAMS 16  // Increased streams
#define BATCH_SIZE 64   // Large batch size for efficiency
#define WARP_SIZE 32
#define MAX_ITERATIONS 10  // Fixed iterations for performance

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Use float instead of double for 2x speed improvement
typedef float precision_t;

// Pre-computed lookup table for JacLog function (performance optimization)
__constant__ precision_t jaclog_table[2048];

// --- Optimized CUDA Device Functions ---
__device__ __forceinline__ uint8_t ParityCheck_device(uint8_t *c, uint8_t *H, uint64_t n, uint64_t s) {
    #pragma unroll
    for (size_t j = 0; j < s; j++){
        uint8_t syndrome = 0;
        #pragma unroll
        for (size_t i = 0; i < n; i++)
            syndrome ^= (c[i] * H[j*n + i]);
        if (syndrome == 1) return 0;
    }
    return 1;
}

__device__ __forceinline__ void HardDec_device(uint8_t *c, precision_t *llr, uint64_t n) {
    #pragma unroll
    for (size_t i = 0; i < n; i++)
        c[i] = (llr[i] > 0.0f) ? 0 : 1;
}

__device__ __forceinline__ int parity_device(uint8_t array[], uint64_t n) {
    int sum = 0;
    #pragma unroll
    for (uint64_t i = 0; i < n; i++)
        sum += array[i];
    return sum % 2;
}

__device__ __forceinline__ precision_t prob_parity_device(int parity_cHD, precision_t *absL, uint64_t n) {
    precision_t prob_even = 1.0f;
    #pragma unroll
    for (uint64_t i = 0; i < n; i++) {
        prob_even *= (1.0f - 2.0f * expf(-absL[i]) / (1.0f + expf(-absL[i])));
    }
    prob_even = 0.5f * (1.0f + prob_even);
    return (parity_cHD == 0) ? prob_even : 1.0f - prob_even;
}

__device__ __forceinline__ void AddTEP_device(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n) {
    #pragma unroll
    for (size_t i = 0; i < n; i++)
        c[perm[i]] = cHD[perm[i]] ^ TEP[i];
}

// Fast JacLog using lookup table
__device__ __forceinline__ precision_t JacLog_device(precision_t x) {
    if (x > 30.0f) return x;
    if (x < -30.0f) return 0.0f;

    // Use lookup table for speed
    int index = (int)((x + 30.0f) * 34.13f); // 2048/(60) ≈ 34.13
    index = max(0, min(2047, index));
    return jaclog_table[index];
}

__device__ __forceinline__ void QuickSort_device(precision_t *a, size_t *perm, uint64_t n) {
    // Optimized insertion sort for small arrays
    for (uint64_t i = 1; i < n; i++) {
        precision_t key = a[i];
        size_t key_perm = perm[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            perm[j + 1] = perm[j];
            j--;
        }
        a[j + 1] = key;
        perm[j + 1] = key_perm;
    }
}

__device__ __forceinline__ precision_t getPM_HD_device(precision_t *absL, uint64_t n) {
    precision_t pm = 0.0f;
    #pragma unroll
    for(size_t i = 0; i < n; i++)
        pm += JacLog_device(-absL[i]);
    return pm;
}

__device__ __forceinline__ precision_t getPM_device(uint8_t *TEP, precision_t *absL, precision_t PM_HD, uint64_t n) {
    precision_t pm = PM_HD;
    #pragma unroll
    for(size_t i = 0; i < n; i++) {
        if (TEP[i] == 1)
            pm += (JacLog_device(absL[i]) - JacLog_device(-absL[i]));
    }
    return pm;
}

__device__ __forceinline__ precision_t getLConf_device(precision_t *pNL, precision_t P_notGuess, uint64_t cur_L, precision_t *score, uint64_t s, uint8_t even) {
    precision_t P_pos = 0.0f;
    #pragma unroll
    for(size_t i = 0; i <= cur_L; i++)
        P_pos += expf(-score[4*i+1]);
    if(even==1) s--;
    precision_t P_neg = powf(2.0f, -(precision_t)s) * P_notGuess;
    pNL[0] = P_neg;
    return (P_pos + P_neg > 1e-9f) ? (P_pos / (P_pos + P_neg)) : 1.0f;
}

__device__ __forceinline__ int32_t findMax_device(int32_t a, int32_t b) {
    return !(b > a) ? a : b;
}

__device__ void mountain_build_device(int32_t *u, int32_t k, int32_t w, int32_t W1, int32_t n1) {
    #pragma unroll
    for(size_t i = k + 1; i < w; i++)
        u[i] = u[k];
    uint64_t W2 = W1;
    for(size_t i = 0; i < w; i++)
        W2 -= u[i];
    uint64_t q = (uint64_t)floorf((precision_t)W2 / (precision_t)(n1 - u[k]));
    uint64_t r = W2 - q*(n1 - u[k]);
    if (q != 0) {
        for(size_t i = w-q; i < w; i++)
            u[i] = n1;
    }
    if (w > q)
        u[w-q-1] = u[w-q-1] + r;
}

__device__ void getAPP_device(uint64_t cur_L, precision_t *score, precision_t *APP) {
    if (cur_L == 0) return;
    precision_t P_pos = 0.0f;
    #pragma unroll
    for(size_t i=0; i<cur_L; i++)
        P_pos += expf(-score[4*i+1]);
    if (P_pos < 1e-30f) return;
    precision_t den = score[4*(cur_L-1)+3] / P_pos;
    #pragma unroll
    for(size_t i=0; i<cur_L; i++)
        APP[i] = expf(-score[4*i+1]) * den;
}

// Optimized SOGRAND main logic with reduced complexity
__device__ void sogrand_main_logic_device_fast(precision_t* chat_list, precision_t* s_list, precision_t* T_val,
                                              precision_t* curL_val, precision_t* pNL_val, precision_t* APP_list,
                                              precision_t* llr, uint8_t* H_flat, int n, int s, int IC,
                                              uint64_t L, uint64_t Tmax, precision_t thres, int even) {

    // Use registers for small arrays
    size_t perm[15];
    uint8_t cHD[15];
    uint8_t TEP[15];
    uint8_t c[15];
    precision_t absL[15];
    int32_t u[15];
    int32_t d[15];
    int32_t D[15];

    // Initialize arrays
    #pragma unroll
    for(size_t i = 0; i < n; i++) perm[i] = i;
    #pragma unroll
    for(size_t i = 0; i < 4*L; i++) s_list[i] = 0.0f;
    #pragma unroll
    for(size_t i = 0; i < L; i++) APP_list[i] = 0.0f;

    uint64_t cur_L = 0;
    HardDec_device(cHD, llr, n);
    uint8_t parity_cHD = parity_device(cHD, n);
    pNL_val[0] = 0.0f;

    if (Tmax == 0) Tmax = Inf;

    #pragma unroll
    for (size_t i = 0; i < n; i++) {
        TEP[i] = 0;
        absL[i] = fabsf(llr[i]);
    }

    precision_t P_notGuess = 1.0f;
    if (even == 1) P_notGuess = prob_parity_device(parity_cHD, absL, n);

    precision_t PM_HD = getPM_HD_device(absL, n);
    QuickSort_device(absL, perm, n);

    if (IC < 0) {
        if (roundf((precision_t)n/2.0f) > 1) {
            precision_t beta = (absL[(uint64_t)roundf((precision_t)n/2.0f) - 1] - absL[0]) / (roundf((precision_t)n/2.0f) - 1);
            IC = (beta > 1e-9f) ? findMax_device((int32_t)roundf(absL[0]/beta - 1.0f), 0) : 0;
        } else {
            IC = 0;
        }
    }

    AddTEP_device(c, cHD, TEP, perm, n);
    T_val[0] = 1;

    if (parity_cHD == 0 || even == 0) {
        P_notGuess -= expf(-getPM_device(TEP, absL, PM_HD, n));
    }

    if (ParityCheck_device(c, H_flat, n, s) == 1) {
        precision_t pm = getPM_device(TEP, absL, PM_HD, n);
        APP_list[0] = pm;
        #pragma unroll
        for(size_t i = 0; i < n; i++) chat_list[i] = c[i];
        s_list[0] = pm;
        s_list[1] = pm;
        s_list[2] = T_val[0];
        s_list[3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
        cur_L++;
        if (even == 1) P_notGuess -= expf(-pm);
        if ((s_list[3] > thres) || (cur_L == L)) {
            getAPP_device(cur_L, s_list, APP_list);
            curL_val[0] = cur_L;
            return;
        }
    }

    // Simplified SOGRAND loop with reduced iterations
    int32_t wt = IC + 1;
    int max_wt = min(n + IC, 50); // Limit search space for speed

    while ((cur_L < L) && (T_val[0] < Tmax) && (wt <= max_wt)) {
        int32_t w = 1;
        precision_t temp_sqrt = powf(1.0f+2.0f*((precision_t)n+(precision_t)IC), 2.0f) - 8.0f*wt;
        if (temp_sqrt >= 0) {
            w = findMax_device(1, (int32_t)ceilf((1.0f+2.0f*((precision_t)n+(precision_t)IC) - sqrtf(temp_sqrt))/2.0f));
        } else {
            wt++; continue;
        }

        if (even == 1 && (w % 2 != parity_cHD)) w++;

        while (w <= n && w <= 8) { // Limit w for speed
            int32_t W = wt - IC * w;
            if (W < w * (w + 1) / 2) break;

            int32_t W1 = W - w * (w + 1) / 2;
            int32_t n1 = n - w;
            #pragma unroll
            for (size_t i = 0; i < w; i++) u[i] = 0;

            mountain_build_device(u, 0, w, W1, n1);

            int mountain_iter_guard = 0;
            int max_mountain_iter = min(1000, 50000 / w); // Adaptive limit

            do {
                #pragma unroll
                for (size_t i = 0; i < n; i++) TEP[i] = 0;
                for (size_t i = 0; i < w; i++) TEP[i + u[i]] = 1;
                AddTEP_device(c, cHD, TEP, perm, n);
                T_val[0]++;

                if (parity_cHD == 0 || even == 0) {
                    P_notGuess -= expf(-getPM_device(TEP, absL, PM_HD, n));
                }

                if (ParityCheck_device(c, H_flat, n, s) == 1) {
                    precision_t pm = getPM_device(TEP, absL, PM_HD, n);
                    APP_list[cur_L] = pm;
                    #pragma unroll
                    for(size_t i = 0; i < n; i++) chat_list[cur_L * n + i] = c[i];
                    s_list[4 * cur_L] = pm;
                    s_list[4 * cur_L + 1] = pm;
                    s_list[4 * cur_L + 2] = T_val[0];
                    s_list[4 * cur_L + 3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
                    cur_L++;
                    if (even == 1) P_notGuess -= expf(-pm);
                    if ((s_list[4 * (cur_L - 1) + 3] > thres) || (cur_L == L)) {
                        getAPP_device(cur_L, s_list, APP_list);
                        curL_val[0] = cur_L;
                        return;
                    }
                }

                #pragma unroll
                for (size_t i = 0; i < w - 1; i++) d[i] = u[i + 1] - u[i];
                d[w - 1] = 0;
                D[w - 1] = d[w - 1];
                for (int i = w - 2; i >= 0; i--) D[i] = D[i + 1] + d[i];

                if (D[0] < 2) break;

                int32_t k_mt = 0;
                for (int i = w - 1; i > 0; i--) {
                    if (D[i] >= 2) { k_mt = i; break; }
                }
                u[k_mt]++;
                mountain_build_device(u, k_mt, w, W1, n1);

            } while (++mountain_iter_guard < max_mountain_iter);

            w++;
            if (even == 1 && (w % 2 != parity_cHD)) w++;
        }
        wt++;
    }

    curL_val[0] = cur_L;
    pNL_val[0] = P_notGuess;
    getAPP_device(cur_L, s_list, APP_list);
}

__device__ void SOGRAND_bitSO_device_fast(precision_t* L_APP, precision_t* L_E, int* N_guess, precision_t* llr,
                                          uint8_t* H_matrix, int n, int k, int L, uint64_t Tmax,
                                          precision_t thres, int even) {

    precision_t chat_list[45];
    precision_t s_list[12];
    precision_t T_val, curL_val, pNL_val;
    precision_t APP_list[3];

    sogrand_main_logic_device_fast(chat_list, s_list, &T_val, &curL_val, &pNL_val, APP_list,
                                  llr, H_matrix, n, n - k, -1, L, Tmax, thres, even);

    int curL = (int)curL_val;
    if (curL == 0) {
        #pragma unroll
        for(int i=0; i<n; ++i) L_APP[i] = llr[i];
        #pragma unroll
        for(int i=0; i<n; ++i) L_E[i] = 0.0f;
    } else {
        precision_t PM[3];
        #pragma unroll
        for(int i=0; i<curL; ++i) PM[i] = s_list[4*i + 1];
        precision_t p_notinlist = fmaxf(pNL_val, 1e-9f);

        precision_t pp1[15], pp0[15];
        #pragma unroll
        for(int i=0; i<n; ++i) {
            pp1[i] = 1.0f / (1.0f + expf(llr[i]));
            pp0[i] = 1.0f - pp1[i];
            pp1[i] = fmaxf(pp1[i], 1e-9f); pp1[i] = fminf(pp1[i], 1.0f - 1e-9f);
            pp0[i] = fmaxf(pp0[i], 1e-9f); pp0[i] = fminf(pp0[i], 1.0f - 1e-9f);
        }

        precision_t p[3];
        #pragma unroll
        for(int i=0; i<curL; ++i) p[i] = expf(-PM[i]);

        precision_t p1[15], p0[15];
        #pragma unroll
        for(int i=0; i<n; ++i) {
            p1[i] = 0.0f;
            p0[i] = 0.0f;
        }

        #pragma unroll
        for(int i=0; i<n; ++i) {
            for(int j=0; j<curL; ++j) {
                p1[i] += chat_list[i + j*n] * p[j];
                p0[i] += (1.0f - chat_list[i + j*n]) * p[j];
            }
        }

        #pragma unroll
        for(int i=0; i<n; ++i) {
            p0[i] += p_notinlist * pp0[i];
            p1[i] += p_notinlist * pp1[i];
        }

        #pragma unroll
        for(int i=0; i<n; ++i) {
            L_APP[i] = logf(fmaxf(p0[i], 1e-30f)) - logf(fmaxf(p1[i], 1e-30f));
            L_E[i] = L_APP[i] - llr[i];
        }
    }

    if (N_guess) *N_guess = (int)T_val;
}

// Mega-batch kernel for maximum throughput
__global__ void mega_batch_sogrand_kernel(precision_t* input_batch, precision_t* L_APP_batch, precision_t* L_E_batch,
                                         uint8_t* H_flat, int n, int k, int L, uint64_t Tmax,
                                         precision_t thres, int even, int* NG_total, int batch_size,
                                         precision_t* alpha_array, int dim_type) {
    int global_batch_idx = blockIdx.x;
    int local_thread = threadIdx.x;
    int threads_per_block = blockDim.x;

    if (global_batch_idx >= batch_size) return;

    __shared__ precision_t shared_input[15*15*15];
    __shared__ precision_t shared_L_APP[15*15*15];
    __shared__ precision_t shared_L_E[15*15*15];

    int batch_offset = global_batch_idx * n * n * n;

    // Load data to shared memory
    for (int i = local_thread; i < n*n*n; i += threads_per_block) {
        shared_input[i] = input_batch[batch_offset + i];
    }
    __syncthreads();

    // Process based on dimension type (0=columns, 1=rows, 2=slices)
    if (local_thread < n * n) {
        int slice, row, col;
        precision_t vec_in[15], vec_L_APP[15], vec_L_E_vec[15];
        int N_guess = 0;

        if (dim_type == 0) { // Columns
            slice = local_thread / n;
            col = local_thread % n;
            #pragma unroll
            for(int r = 0; r < n; r++) {
                int idx = slice * n * n + r * n + col;
                vec_in[r] = shared_input[idx];
            }
        } else if (dim_type == 1) { // Rows
            slice = local_thread / n;
            row = local_thread % n;
            #pragma unroll
            for(int c = 0; c < n; c++) {
                int idx = slice * n * n + row * n + c;
                vec_in[c] = shared_input[idx];
            }
        } else { // Slices
            col = local_thread / n;
            row = local_thread % n;
            #pragma unroll
            for(int s = 0; s < n; s++) {
                int idx = s * n * n + row * n + col;
                vec_in[s] = shared_input[idx];
            }
        }

        SOGRAND_bitSO_device_fast(vec_L_APP, vec_L_E_vec, &N_guess, vec_in, H_flat, n, k, L, Tmax, thres, even);

        // Store results back to shared memory
        if (dim_type == 0) { // Columns
            #pragma unroll
            for(int r = 0; r < n; r++) {
                int idx = slice * n * n + r * n + col;
                shared_L_APP[idx] = vec_L_APP[r];
                shared_L_E[idx] = vec_L_E_vec[r];
            }
        } else if (dim_type == 1) { // Rows
            #pragma unroll
            for(int c = 0; c < n; c++) {
                int idx = slice * n * n + row * n + c;
                shared_L_APP[idx] = vec_L_APP[c];
                shared_L_E[idx] = vec_L_E_vec[c];
            }
        } else { // Slices
            #pragma unroll
            for(int s = 0; s < n; s++) {
                int idx = s * n * n + row * n + col;
                shared_L_APP[idx] = vec_L_APP[s];
                shared_L_E[idx] = vec_L_E_vec[s];
            }
        }

        atomicAdd(NG_total, N_guess);
    }

    __syncthreads();

    // Write results back to global memory
    for (int i = local_thread; i < n*n*n; i += threads_per_block) {
        L_APP_batch[batch_offset + i] = shared_L_APP[i];
        L_E_batch[batch_offset + i] = shared_L_E[i];
    }
}

// Fused update kernel for better performance
__global__ void fused_update_kernel(precision_t* input_batch, precision_t* L_channel_batch, precision_t* L_E_batch,
                                   precision_t alpha_val, int batch_size, int codeword_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * codeword_size;

    // Process 4 elements per thread for memory coalescing
    if (idx * 4 < total_elements) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int current_idx = idx * 4 + i;
            if (current_idx < total_elements) {
                input_batch[current_idx] = L_channel_batch[current_idx] + alpha_val * L_E_batch[current_idx];
            }
        }
    }
}

// Structure of Arrays for better memory access
typedef struct {
    cudaStream_t stream;
    precision_t* h_llr_batch;
    precision_t* h_output_batch;
    precision_t* d_L_channel_batch;
    precision_t* d_L_APP_batch;
    precision_t* d_L_E_batch;
    precision_t* d_input_batch;
    int* d_NG_count;
    int* h_NG_count;
    int batch_size;
    int blocks_processed;
} BatchStreamData;

// Host functions
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

void hard_decision_batch(precision_t* llr_batch, int* bits_batch, int message_block_size, int batch_size) {
    int k = 10;
    int n = 15;

    for (int batch = 0; batch < batch_size; batch++) {
        int bit_idx = 0;
        int batch_offset = batch * n * n * n;
        int output_offset = batch * message_block_size;

        for (int slice = 0; slice < k; slice++) {
            for (int row = 0; row < k; row++) {
                for (int col = 0; col < k; col++) {
                    int tensor_idx = batch_offset + slice * n * n + row * n + col;
                    int output_idx = output_offset + bit_idx;
                    if (bit_idx < message_block_size) {
                        bits_batch[output_idx] = (llr_batch[tensor_idx] > 0.0f) ? 0 : 1;
                        bit_idx++;
                    }
                }
            }
        }
    }
}

void initBatchStreamData(BatchStreamData* streamData, int codeword_block_size, int message_block_size, int batch_size) {
    CUDA_CHECK(cudaStreamCreate(&streamData->stream));

    streamData->batch_size = batch_size;
    size_t batch_codeword_size = codeword_block_size * batch_size;

    // Allocate pinned memory for maximum transfer speed
    CUDA_CHECK(cudaHostAlloc(&streamData->h_llr_batch,
                            batch_codeword_size * sizeof(precision_t),
                            cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&streamData->h_output_batch,
                            batch_codeword_size * sizeof(precision_t),
                            cudaHostAllocDefault));

    streamData->h_NG_count = (int*)malloc(sizeof(int));

    // Allocate device memory efficiently
    CUDA_CHECK(cudaMalloc(&streamData->d_L_channel_batch, batch_codeword_size * sizeof(precision_t)));
    CUDA_CHECK(cudaMalloc(&streamData->d_L_APP_batch, batch_codeword_size * sizeof(precision_t)));
    CUDA_CHECK(cudaMalloc(&streamData->d_L_E_batch, batch_codeword_size * sizeof(precision_t)));
    CUDA_CHECK(cudaMalloc(&streamData->d_input_batch, batch_codeword_size * sizeof(precision_t)));
    CUDA_CHECK(cudaMalloc(&streamData->d_NG_count, sizeof(int)));

    streamData->blocks_processed = 0;
}

void cleanupBatchStreamData(BatchStreamData* streamData) {
    CUDA_CHECK(cudaStreamDestroy(streamData->stream));
    CUDA_CHECK(cudaFreeHost(streamData->h_llr_batch));
    CUDA_CHECK(cudaFreeHost(streamData->h_output_batch));
    free(streamData->h_NG_count);
    CUDA_CHECK(cudaFree(streamData->d_L_channel_batch));
    CUDA_CHECK(cudaFree(streamData->d_L_APP_batch));
    CUDA_CHECK(cudaFree(streamData->d_L_E_batch));
    CUDA_CHECK(cudaFree(streamData->d_input_batch));
    CUDA_CHECK(cudaFree(streamData->d_NG_count));
}

// Initialize lookup table for JacLog function
void init_jaclog_table() {
    precision_t h_jaclog_table[2048];
    for (int i = 0; i < 2048; i++) {
        precision_t x = -30.0f + (60.0f * i) / 2047.0f;
        if (x > 30.0f) h_jaclog_table[i] = x;
        else if (x < -30.0f) h_jaclog_table[i] = 0.0f;
        else h_jaclog_table[i] = logf(1.0f + expf(x));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(jaclog_table, h_jaclog_table, 2048 * sizeof(precision_t)));
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

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("GPU: %s\n", deviceProp.name);
    printf("Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Concurrent kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");

    // Initialize JacLog lookup table
    init_jaclog_table();

    int** G = (int**)malloc(k * sizeof(int*));
    int** H = (int**)malloc((n-k) * sizeof(int*));
    for(int i = 0; i < k; i++) G[i] = (int*)malloc(n * sizeof(int));
    for(int i = 0; i < (n-k); i++) H[i] = (int*)malloc(n * sizeof(int));
    getGH_sys_CRC(n, k, G, H);

    // Setup device matrices
    uint8_t* d_H_flat;
    CUDA_CHECK(cudaMalloc(&d_H_flat, (n-k) * n * sizeof(uint8_t)));

    uint8_t* h_H_flat = (uint8_t*)malloc((n-k) * n * sizeof(uint8_t));
    for (int i = 0; i < (n-k); i++) {
        for (int j = 0; j < n; j++) {
            h_H_flat[i*n + j] = (uint8_t)H[i][j];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_H_flat, h_H_flat, (n-k) * n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    free(h_H_flat);

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
    const int Imax = MAX_ITERATIONS; // Fixed to 10 iterations
    const uint64_t Tmax = UINT64_MAX;
    const precision_t p_ET = 1e-5f;
    const precision_t thres = 1.0f - p_ET;

    // Optimized alpha array for 10 iterations
    precision_t alpha[20];
    for(int i = 0; i < 20; i++) alpha[i] = 0.7f;

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("Ultra-Fast CUDA Decoding %s to %s using cubic code (n=%d, k=%d)...\n",
           input_filename, output_filename, n, k);
    printf("Configuration: %d iterations, batch size %d, %d streams\n", Imax, BATCH_SIZE, MAX_STREAMS);

    BatchStreamData streams[MAX_STREAMS];
    for (int i = 0; i < MAX_STREAMS; i++) {
        initBatchStreamData(&streams[i], codeword_block_size, message_block_size, BATCH_SIZE);
    }

    int total_blocks = 0;
    clock_t start_time = clock();

    int current_stream = 0;
    int streams_in_flight = 0;
    bool done_reading = false;

    while (!done_reading || streams_in_flight > 0) {
        // Launch new work if available
        if (!done_reading && streams_in_flight < MAX_STREAMS) {
            BatchStreamData* stream = &streams[current_stream];
            int batch_size = stream->batch_size;
            size_t total_elements = codeword_block_size * batch_size;

            // Read batch of codewords
            size_t read_size = 0;
            for (int b = 0; b < batch_size; b++) {
                double temp_buffer[codeword_block_size];
                size_t current_read = fread(temp_buffer, sizeof(double), codeword_block_size, fin);
                if (current_read == codeword_block_size) {
                    // Convert double to float for speed
                    for (int i = 0; i < codeword_block_size; i++) {
                        stream->h_llr_batch[b * codeword_block_size + i] = (precision_t)temp_buffer[i];
                    }
                    read_size += current_read;
                } else {
                    batch_size = b; // Partial batch
                    break;
                }
            }

            if (batch_size > 0) {
                stream->batch_size = batch_size;
                total_elements = codeword_block_size * batch_size;

                // Transfer data to GPU asynchronously
                CUDA_CHECK(cudaMemcpyAsync(stream->d_L_channel_batch, stream->h_llr_batch,
                                          total_elements * sizeof(precision_t),
                                          cudaMemcpyHostToDevice, stream->stream));

                // Initialize arrays
                CUDA_CHECK(cudaMemsetAsync(stream->d_L_E_batch, 0,
                                          total_elements * sizeof(precision_t), stream->stream));
                CUDA_CHECK(cudaMemsetAsync(stream->d_NG_count, 0, sizeof(int), stream->stream));

                // Ultra-fast iterative decoding with no early termination
                for (int iter = 1; iter <= Imax; iter++) {
                    // COLUMNS processing
                    precision_t alpha_cols = alpha[2*iter-2];

                    int update_threads = 256;
                    int update_blocks = (total_elements + update_threads * 4 - 1) / (update_threads * 4);
                    fused_update_kernel<<<update_blocks, update_threads, 0, stream->stream>>>(
                        stream->d_input_batch, stream->d_L_channel_batch, stream->d_L_E_batch,
                        alpha_cols, batch_size, codeword_block_size);

                    dim3 grid_mega(batch_size, 1, 1);
                    dim3 block_mega(min(n*n, 256), 1, 1);
                    mega_batch_sogrand_kernel<<<grid_mega, block_mega, 0, stream->stream>>>(
                        stream->d_input_batch, stream->d_L_APP_batch, stream->d_L_E_batch, d_H_flat,
                        n, k, L, Tmax, thres, even, stream->d_NG_count, batch_size, alpha, 0);

                    // ROWS processing
                    precision_t alpha_rows = alpha[2*iter-1];
                    fused_update_kernel<<<update_blocks, update_threads, 0, stream->stream>>>(
                        stream->d_input_batch, stream->d_L_channel_batch, stream->d_L_E_batch,
                        alpha_rows, batch_size, codeword_block_size);

                    mega_batch_sogrand_kernel<<<grid_mega, block_mega, 0, stream->stream>>>(
                        stream->d_input_batch, stream->d_L_APP_batch, stream->d_L_E_batch, d_H_flat,
                        n, k, L, Tmax, thres, even, stream->d_NG_count, batch_size, alpha, 1);

                    // SLICES processing
                    fused_update_kernel<<<update_blocks, update_threads, 0, stream->stream>>>(
                        stream->d_input_batch, stream->d_L_channel_batch, stream->d_L_E_batch,
                        alpha_rows, batch_size, codeword_block_size);

                    mega_batch_sogrand_kernel<<<grid_mega, block_mega, 0, stream->stream>>>(
                        stream->d_input_batch, stream->d_L_APP_batch, stream->d_L_E_batch, d_H_flat,
                        n, k, L, Tmax, thres, even, stream->d_NG_count, batch_size, alpha, 2);
                }

                // Transfer results back
                CUDA_CHECK(cudaMemcpyAsync(stream->h_output_batch, stream->d_L_APP_batch,
                                          total_elements * sizeof(precision_t),
                                          cudaMemcpyDeviceToHost, stream->stream));

                stream->blocks_processed = batch_size;
                streams_in_flight++;
                current_stream = (current_stream + 1) % MAX_STREAMS;
            } else {
                done_reading = true;
            }
        }

        // Check for completed streams
        for (int s = 0; s < MAX_STREAMS; s++) {
            BatchStreamData* stream = &streams[s];
            cudaError_t status = cudaStreamQuery(stream->stream);

            if (status == cudaSuccess && stream->blocks_processed > 0) {
                int batch_size = stream->blocks_processed;
                int total_message_bits = message_block_size * batch_size;
                int* bit_buffer = (int*)malloc(total_message_bits * sizeof(int));

                // Process batch results
                hard_decision_batch(stream->h_output_batch, bit_buffer, message_block_size, batch_size);

                // Write to output file in optimized batches
                for (int batch = 0; batch < batch_size; batch++) {
                    unsigned char byte_out = 0;
                    int bit_count_out = 0;
                    int batch_offset = batch * message_block_size;

                    for (int i = 0; i < message_block_size; i++) {
                        byte_out = (byte_out << 1) | bit_buffer[batch_offset + i];
                        bit_count_out++;
                        if (bit_count_out == 8) {
                            fwrite(&byte_out, 1, 1, fout);
                            byte_out = 0;
                            bit_count_out = 0;
                        }
                    }

                    if (bit_count_out > 0) {
                        byte_out <<= (8 - bit_count_out);
                        fwrite(&byte_out, 1, 1, fout);
                    }
                }

                free(bit_buffer);
                total_blocks += batch_size;
                stream->blocks_processed = 0;
                streams_in_flight--;
            } else if (status != cudaErrorNotReady && status != cudaSuccess) {
                CUDA_CHECK(status);
            }
        }

        // Optimized waiting strategy
        if (streams_in_flight >= MAX_STREAMS) {
            usleep(50); // Very short sleep for maximum responsiveness
        }
    }

    // Wait for all streams to complete
    for (int i = 0; i < MAX_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i].stream));
    }

    clock_t end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Ultra-Fast CUDA Decoding complete. %d block(s) decoded.\n", total_blocks);
    printf("Total time: %.3f seconds\n", cpu_time_used);
    if (total_blocks > 0) {
        printf("Throughput: %.1f blocks/second\n", total_blocks / cpu_time_used);
        printf("Throughput: %.1f MB/s\n",
               (total_blocks * codeword_block_size * sizeof(precision_t)) / (cpu_time_used * 1024 * 1024));
    }

    // Cleanup
    for (int i = 0; i < MAX_STREAMS; i++) {
        cleanupBatchStreamData(&streams[i]);
    }

    CUDA_CHECK(cudaFree(d_H_flat));

    fclose(fin);
    fclose(fout);
    for(int i = 0; i < k; i++) free(G[i]);
    for(int i = 0; i < (n-k); i++) free(H[i]);
    free(G);
    free(H);

    return 0;
}
