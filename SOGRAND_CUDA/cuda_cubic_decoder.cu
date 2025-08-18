// sogrand_batched_streams.cu
// GPU batched SOGRAND (n=15, k=10) with streams, cuBLAS, and SIMD fallback.
// Keeps the exact decoding logic of your CPU reference (columns -> rows -> slices).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <stdbool.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Inf 0x7fffffff

// ===================== Tunables =====================
#ifndef MAX_STREAMS
#define MAX_STREAMS 4
#endif

#ifndef BATCH_PER_STREAM
#define BATCH_PER_STREAM 64   // try 64/128/256 depending on VRAM
#endif

#ifndef USE_CUBLAS_FOR_AXPY
#define USE_CUBLAS_FOR_AXPY 1 // 1 = cuBLAS daxpy, 0 = SIMD kernel (double2)
#endif
// ====================================================

#define CUDA_CHECK(call) \
 do { cudaError_t e = (call); if (e != cudaSuccess) { \
   fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
   exit(1);} } while(0)

#define CUBLAS_CHECK(call) \
 do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
   fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
   exit(1);} } while(0)


// --- Data Structures for 3D Tensor (CPU side; used for early_termination & hard_decision host ops) ---
typedef struct {
    double* data;
    int dim1, dim2, dim3;
} Tensor_d;

typedef struct {
    int* data;
    int dim1, dim2, dim3;
} Tensor_i;

Tensor_d create_tensor_d(int d1, int d2, int d3) {
    Tensor_d t; t.dim1=d1; t.dim2=d2; t.dim3=d3;
    t.data = (double*)calloc((size_t)d1*d2*d3, sizeof(double));
    return t;
}
Tensor_i create_tensor_i(int d1, int d2, int d3) {
    Tensor_i t; t.dim1=d1; t.dim2=d2; t.dim3=d3;
    t.data = (int*)calloc((size_t)d1*d2*d3, sizeof(int));
    return t;
}
void free_tensor_d(Tensor_d t) { free(t.data); }
void free_tensor_i(Tensor_i t) { free(t.data); }
int get_tensor_i(Tensor_i t, int i, int j, int k) { return t.data[k*t.dim1*t.dim2 + j*t.dim1 + i]; }
void set_tensor_i(Tensor_i t, int i, int j, int k, int val) { t.data[k*t.dim1*t.dim2 + j*t.dim1 + i] = val; }
double get_tensor_d(Tensor_d t, int i, int j, int k) { return t.data[k*t.dim1*t.dim2 + j*t.dim1 + i]; }
void set_tensor_d(Tensor_d t, int i, int j, int k, double val) { t.data[k*t.dim1*t.dim2 + j*t.dim1 + i] = val; }

// --- Host helpers for code construction ---
int** create_int_matrix(int rows, int cols) {
    int** m = (int**)malloc((size_t)rows * sizeof(int*));
    for (int i=0;i<rows;i++) m[i] = (int*)malloc((size_t)cols * sizeof(int));
    return m;
}
void free_int_matrix(int** m, int rows) {
    for (int i=0;i<rows;i++) free(m[i]);
    free(m);
}

int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? (int)floor(log2((double)dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));
    for (int i = 0; i < len; i++) poly[i] = (int)((dec_val >> (len - 1 - i)) & 1);
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
        fprintf(stderr, "Error: (n, k) = (%d, %d) not supported.\n", n, k);
        exit(1);
    }

    int poly_len;
    int* poly = koopman2matlab(hex_poly, &poly_len);

    int** P = (int**)malloc((size_t)k * sizeof(int*));
    for (int i=0;i<k;i++) P[i] = (int*)malloc((size_t)(n-k) * sizeof(int));
    int* msg_poly = (int*)calloc((size_t)(k + (n-k)), sizeof(int));

    for (int i = 0; i < k; i++) {
        memset(msg_poly, 0, (size_t)(k + (n-k)) * sizeof(int));
        msg_poly[i] = 1;
        for (int j = 0; j < k; j++) {
            if (msg_poly[j] == 1) {
                for (int l = 0; l < poly_len; l++) {
                    msg_poly[j + l] ^= poly[l];
                }
            }
        }
        for (int j = 0; j < (n-k); j++) P[i][j] = msg_poly[k + j];
    }

    for (int i=0;i<k;i++) {
        for (int j=0;j<k;j++) G[i][j] = (i==j)?1:0;
        for (int j=0;j<(n-k);j++) G[i][k+j] = P[i][j];
    }
    for (int i=0;i<(n-k);i++) {
        for (int j=0;j<k;j++) H[i][j] = P[j][i];
        for (int j=0;j<(n-k);j++) H[i][k+j] = (i==j)?1:0;
    }

    free(poly);
    free(msg_poly);
    for (int i=0;i<k;i++) free(P[i]);
    free(P);
}

// ===================== Device (decoder core) =====================
// All device functions mirror the CPU logic (n = 15, L <= 3).

__device__ uint8_t ParityCheck_device(uint8_t *c, uint8_t *H, uint64_t n, uint64_t s) {
    for (size_t j = 0; j < s; j++){
        uint8_t syndrome = 0;
        for (size_t i = 0; i < n; i++) syndrome ^= (c[i] * H[j*n + i]);
        if (syndrome == 1) return 0;
    }
    return 1;
}
__device__ void HardDec_device(uint8_t *c, double *llr, uint64_t n) {
    for (size_t i = 0; i < n; i++) c[i] = (llr[i] > 0.0) ? 0 : 1;
}
__device__ int parity_device(uint8_t array[], uint64_t n) {
    int sum = 0; for (uint64_t i = 0; i < n; i++) sum += array[i]; return sum % 2;
}
__device__ double prob_parity_device(int parity_cHD, double *absL, uint64_t n) {
    double p_e = 1.0;
    for (uint64_t i = 0; i < n; i++) p_e *= (1.0 - 2.0 * exp(-absL[i]) / (1.0 + exp(-absL[i])));
    p_e = 0.5 * (1.0 + p_e);
    return (parity_cHD == 0) ? p_e : 1.0 - p_e;
}
__device__ void AddTEP_device(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n) {
    for (size_t i = 0; i < n; i++) c[perm[i]] = cHD[perm[i]] ^ TEP[i];
}
__device__ double JacLog_device(double x) {
    if (x > 30) return x;
    if (x < -30) return 0.0;
    return log(1.0 + exp(x));
}
// stable insertion sort with permutation
__device__ void QuickSort_device(double *a, size_t *perm, uint64_t n) {
    if (n < 2) return;
    for (uint64_t i = 1; i < n; i++) {
        double key = a[i];
        size_t key_perm = perm[i];
        int j = (int)i - 1;
        while (j >= 0 && a[j] > key) {
            a[j+1] = a[j];
            perm[j+1] = perm[j];
            j--;
        }
        a[j+1] = key;
        perm[j+1] = key_perm;
    }
}
__device__ double getPM_HD_device(double *absL, uint64_t n) {
    double pm = 0; for (size_t i=0;i<n;i++) pm += JacLog_device(-absL[i]); return pm;
}
__device__ double getPM_device(uint8_t *TEP, double *absL, double PM_HD, uint64_t n) {
    double pm = PM_HD;
    for (size_t i=0;i<n;i++) if (TEP[i]==1) pm += (JacLog_device(absL[i]) - JacLog_device(-absL[i]));
    return pm;
}
__device__ double getLConf_device(double *pNL, double P_notGuess, uint64_t cur_L, double *score, uint64_t s, uint8_t even) {
    double P_pos = 0.0;
    for (size_t i=0;i<=cur_L;i++) P_pos += exp(-score[4*i+1]);
    if (even==1) s--;
    double P_neg = pow(2.0, -(double)s) * P_notGuess;
    pNL[0] = P_neg;
    return (P_pos + P_neg > 1e-9) ? (P_pos / (P_pos + P_neg)) : 1.0;
}
__device__ int32_t findMax_device(int32_t a, int32_t b) { return !(b > a) ? a : b; }
__device__ void mountain_build_device(int32_t *u, int32_t k, int32_t w, int32_t W1, int32_t n1){
    for(size_t i = k + 1; i < (size_t)w; i++) u[i] = u[k];
    uint64_t W2 = W1;
    for(size_t i = 0; i < (size_t)w; i++) W2 -= u[i];
    uint64_t q = (uint64_t)floor( (double)W2 / (double)(n1 - u[k]) );
    uint64_t r = W2 - q*(n1 - u[k]);
    if (q != 0) { for(size_t i = (size_t)w-q; i < (size_t)w; i++) u[i] = n1; }
    if ((size_t)w > q) u[w-q-1] = u[w-q-1] + (int32_t)r;
}
__device__ void getAPP_device(uint64_t cur_L, double *score, double *APP) {
    if (cur_L == 0) return;
    double P_pos = 0.0;
    for (size_t i=0;i<cur_L;i++) P_pos += exp(-score[4*i+1]);
    if (P_pos < 1e-30) return;
    double den = score[4*(cur_L-1)+3] / P_pos;
    for (size_t i=0;i<cur_L;i++) APP[i] = exp(-score[4*i+1]) * den;
}

// -- Device SOGRAND core (n=15, L<=3), mirrors CPU fixes exactly --
__device__ void sogrand_main_logic_device(double* chat_list, double* s_list, double* T_val,
                                         double* curL_val, double* pNL_val, double* APP_list,
                                         double* llr, uint8_t* H_flat, int n, int s, int IC,
                                         uint64_t L, uint64_t Tmax, double thres, int even) {

    size_t perm[15];
    uint8_t cHD[15], TEP[15], c[15];
    double absL[15];
    int32_t u[15], d[15], D[15];

    for (size_t i=0;i<(size_t)n;i++) perm[i] = i;
    for (size_t i=0;i<4*L;i++) s_list[i] = 0;
    for (size_t i=0;i<L;i++) APP_list[i] = 0;

    uint64_t cur_L = 0;
    HardDec_device(cHD, llr, n);
    uint8_t parity_cHD = parity_device(cHD, n);
    pNL_val[0] = 0.0;

    if (Tmax == 0) Tmax = Inf;

    for (size_t i=0;i<(size_t)n;i++){ TEP[i]=0; absL[i]=fabs(llr[i]); }

    double P_notGuess = 1.0;
    if (even == 1) P_notGuess = prob_parity_device(parity_cHD, absL, n);

    double PM_HD = getPM_HD_device(absL, n);
    QuickSort_device(absL, perm, n);

    if (IC < 0) {
        if (round((double)n/2) > 1) {
            double beta = (absL[(uint64_t)round((double)n/2)-1] - absL[0]) / (round((double)n/2)-1);
            IC = (beta > 1e-9) ? findMax_device((int32_t)round(absL[0]/beta - 1), 0) : 0;
        } else {
            IC = 0;
        }
    }

    AddTEP_device(c, cHD, TEP, perm, n);
    T_val[0] = 1;

    if (parity_cHD==0 || even==0) {
        P_notGuess -= exp(-getPM_device(TEP, absL, PM_HD, n));
    }

    if (ParityCheck_device(c, H_flat, n, s) == 1) {
        double pm = getPM_device(TEP, absL, PM_HD, n);
        APP_list[0] = pm;
        for (size_t i=0;i<(size_t)n;i++) chat_list[i] = c[i];
        s_list[0]=pm; s_list[1]=pm; s_list[2]=T_val[0];
        s_list[3]=getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
        cur_L++;
        if (even==1) P_notGuess -= exp(-pm);
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
        if (temp_sqrt >= 0) w = findMax_device(1, (int32_t)ceil((1+2*((double)n+(double)IC)-sqrt(temp_sqrt))/2.0));
        else { wt++; continue; }

        if (even==1 && (w%2 != parity_cHD)) w++;

        while (w <= n) {
            int32_t W = wt - IC*w;
            if (W < w*(w+1)/2) break;
            int32_t W1 = W - w*(w+1)/2;
            int32_t n1 = n - w;
            for (size_t i=0;i<(size_t)w;i++) u[i]=0;

            mountain_build_device(u,0,w,W1,n1);

            int guard = 0;
            do {
                for (size_t i=0;i<(size_t)n;i++) TEP[i]=0;
                for (size_t i=0;i<(size_t)w;i++) TEP[i+u[i]] = 1;
                AddTEP_device(c, cHD, TEP, perm, n);
                T_val[0]++;

                if (parity_cHD==0 || even==0) {
                    P_notGuess -= exp(-getPM_device(TEP, absL, PM_HD, n));
                }

                if (ParityCheck_device(c, H_flat, n, s) == 1) {
                    double pm = getPM_device(TEP, absL, PM_HD, n);
                    APP_list[cur_L] = pm;
                    for (size_t i=0;i<(size_t)n;i++) chat_list[cur_L*n + i] = c[i];
                    s_list[4*cur_L]   = pm;
                    s_list[4*cur_L+1] = pm;
                    s_list[4*cur_L+2] = T_val[0];
                    s_list[4*cur_L+3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
                    cur_L++;
                    if (even==1) P_notGuess -= exp(-pm);
                    if ((s_list[4*(cur_L-1)+3] > thres) || (cur_L == L)) {
                        getAPP_device(cur_L, s_list, APP_list);
                        curL_val[0] = cur_L;
                        return;
                    }
                }

                for (size_t i=0;i<(size_t)w-1;i++) d[i]=u[i+1]-u[i];
                d[w-1]=0; D[w-1]=d[w-1];
                for (int i=w-2;i>=0;i--) D[i]=D[i+1]+d[i];
                if (D[0] < 2) break;

                int32_t k_mt = 0;
                for (int i=w-1;i>0;i--) { if (D[i] >= 2) { k_mt=i; break; } }
                u[k_mt]++;
                mountain_build_device(u, k_mt, w, W1, n1);

            } while (++guard < 100000);

            w++;
            if (even==1 && (w%2 != parity_cHD)) w++;
        }
        wt++;
    }

    curL_val[0] = cur_L;
    pNL_val[0] = P_notGuess;
    getAPP_device(cur_L, s_list, APP_list);
}

__device__ void SOGRAND_bitSO_device(double* L_APP, double* L_E, int* N_guess, double* llr,
                                    uint8_t* H_matrix, int n, int k, int L, uint64_t Tmax,
                                    double thres, int even) {

    // local buffers sized for n<=15, L<=3
    double chat_list[15*3];
    double s_list[4*3];
    double T_val, curL_val, pNL_val;
    double APP_list[3];

    sogrand_main_logic_device(chat_list, s_list, &T_val, &curL_val, &pNL_val, APP_list,
                              llr, H_matrix, n, n-k, -1, L, Tmax, thres, even);

    int curL = (int)curL_val;
    if (curL == 0) {
        for (int i=0;i<n;i++) L_APP[i] = llr[i];
        for (int i=0;i<n;i++) L_E[i]   = 0.0;
    } else {
        double PM[3];
        for (int i=0;i<curL;i++) PM[i] = s_list[4*i + 1];
        double p_notinlist = fmax(pNL_val, 1e-9);

        double pp1[15], pp0[15];
        for (int i=0;i<n;i++) {
            pp1[i] = 1.0 / (1.0 + exp(llr[i]));
            pp0[i] = 1.0 - pp1[i];
            pp1[i] = fmin(fmax(pp1[i], 1e-9), 1.0 - 1e-9);
            pp0[i] = fmin(fmax(pp0[i], 1e-9), 1.0 - 1e-9);
        }

        double p[3];
        for (int i=0;i<curL;i++) p[i] = exp(-PM[i]);

        double p1[15]={0}, p0[15]={0};
        for (int i=0;i<n;i++) {
            for (int j=0;j<curL;j++) {
                p1[i] += chat_list[i + j*n] * p[j];
                p0[i] += (1.0 - chat_list[i + j*n]) * p[j];
            }
        }
        for (int i=0;i<n;i++) { p0[i] += p_notinlist * pp0[i]; p1[i] += p_notinlist * pp1[i]; }
        for (int i=0;i<n;i++) {
            L_APP[i] = log(fmax(p0[i], 1e-30)) - log(fmax(p1[i], 1e-30));
            L_E[i]   = L_APP[i] - llr[i];
        }
    }

    if (N_guess) *N_guess = (int)T_val;
}

// ===================== Batched Kernels (GPU) =====================

// SIMD(ish) vectorized daxpy: input = L_channel + alpha * L_E over entire batch
__global__ void update_input_kernel(double* __restrict__ input,
                                    const double* __restrict__ L_channel,
                                    const double* __restrict__ L_E,
                                    double alpha_val,
                                    int total_elems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid * 2; i < total_elems; i += stride * 2) {
        if (i + 1 < total_elems) {
            double2 c = *reinterpret_cast<const double2*>(&L_channel[i]);
            double2 e = *reinterpret_cast<const double2*>(&L_E[i]);
            double2 r; r.x = c.x + alpha_val * e.x; r.y = c.y + alpha_val * e.y;
            *reinterpret_cast<double2*>(&input[i]) = r;
        } else if (i < total_elems) {
            input[i] = L_channel[i] + alpha_val * L_E[i];
        }
    }
}

// COLUMNS
__global__ void sogrand_columns_kernel(const double* __restrict__ input_data,
                                       double* __restrict__ L_APP_data,
                                       double* __restrict__ L_E_data,
                                       const uint8_t* __restrict__ H_flat,
                                       int n, int k, int L, uint64_t Tmax,
                                       double thres, int even,
                                       int* __restrict__ NG_per_batch,
                                       int batch_blocks) {
    int slice = blockIdx.x;
    int col   = threadIdx.x;
    int b     = blockIdx.y;
    if (slice >= n || col >= n || b >= batch_blocks) return;

    const int cw_size = n*n*n;
    size_t base = (size_t)b * cw_size;

    double vec_in[15], vec_L_APP[15], vec_L_E_vec[15];
    int N_guess = 0;

    for (int row = 0; row < n; row++) {
        int idx = (int)(base + slice * n * n + row * n + col);
        vec_in[row] = input_data[idx];
    }

    SOGRAND_bitSO_device(vec_L_APP, vec_L_E_vec, &N_guess, vec_in,
                         (uint8_t*)H_flat, n, k, L, Tmax, thres, even);

    for (int row = 0; row < n; row++) {
        int idx = (int)(base + slice * n * n + row * n + col);
        L_APP_data[idx] = vec_L_APP[row];
        L_E_data[idx]   = vec_L_E_vec[row];
    }
    atomicAdd(&NG_per_batch[b], N_guess);
}

// ROWS
__global__ void sogrand_rows_kernel(const double* __restrict__ input_data,
                                    double* __restrict__ L_APP_data,
                                    double* __restrict__ L_E_data,
                                    const uint8_t* __restrict__ H_flat,
                                    int n, int k, int L, uint64_t Tmax,
                                    double thres, int even,
                                    int* __restrict__ NG_per_batch,
                                    int batch_blocks) {
    int slice = blockIdx.x;
    int row   = threadIdx.x;
    int b     = blockIdx.y;
    if (slice >= n || row >= n || b >= batch_blocks) return;

    const int cw_size = n*n*n;
    size_t base = (size_t)b * cw_size;

    double vec_in[15], vec_L_APP[15], vec_L_E_vec[15];
    int N_guess = 0;

    for (int col = 0; col < n; col++) {
        int idx = (int)(base + slice * n * n + row * n + col);
        vec_in[col] = input_data[idx];
    }

    SOGRAND_bitSO_device(vec_L_APP, vec_L_E_vec, &N_guess, vec_in,
                         (uint8_t*)H_flat, n, k, L, Tmax, thres, even);

    for (int col = 0; col < n; col++) {
        int idx = (int)(base + slice * n * n + row * n + col);
        L_APP_data[idx] = vec_L_APP[col];
        L_E_data[idx]   = vec_L_E_vec[col];
    }
    atomicAdd(&NG_per_batch[b], N_guess);
}

// SLICES
__global__ void sogrand_slices_kernel(const double* __restrict__ input_data,
                                      double* __restrict__ L_APP_data,
                                      double* __restrict__ L_E_data,
                                      const uint8_t* __restrict__ H_flat,
                                      int n, int k, int L, uint64_t Tmax,
                                      double thres, int even,
                                      int* __restrict__ NG_per_batch,
                                      int batch_blocks) {
    int col = blockIdx.x;
    int row = threadIdx.x;
    int b   = blockIdx.y;
    if (col >= n || row >= n || b >= batch_blocks) return;

    const int cw_size = n*n*n;
    size_t base = (size_t)b * cw_size;

    double vec_in[15], vec_L_APP[15], vec_L_E_vec[15];
    int N_guess = 0;

    for (int slice_idx = 0; slice_idx < n; slice_idx++) {
        int idx = (int)(base + slice_idx * n * n + row * n + col);
        vec_in[slice_idx] = input_data[idx];
    }

    SOGRAND_bitSO_device(vec_L_APP, vec_L_E_vec, &N_guess, vec_in,
                         (uint8_t*)H_flat, n, k, L, Tmax, thres, even);

    for (int slice_idx = 0; slice_idx < n; slice_idx++) {
        int idx = (int)(base + slice_idx * n * n + row * n + col);
        L_APP_data[idx] = vec_L_APP[slice_idx];
        L_E_data[idx]   = vec_L_E_vec[slice_idx];
    }
    atomicAdd(&NG_per_batch[b], N_guess);
}

// Per-codeword early termination
__global__ void early_termination_kernel(const double* __restrict__ L_APP_data,
                                         const int* __restrict__ G_flat,
                                         int n, int k,
                                         int batch_blocks,
                                         int* __restrict__ result_flags) {
    int b = blockIdx.x; // one block per codeword
    if (b >= batch_blocks) return;

    const int cw_size = n*n*n;
    size_t base = (size_t)b * cw_size;

    // Hard decision
    int c_HD[15*15*15];
    for (int i = 0; i < cw_size; i++) {
        c_HD[i] = (L_APP_data[base + i] > 0) ? 0 : 1;
    }

    int c_test[15*15*15];

    // copy systematic part
    for (int i=0;i<k;i++)
      for (int j=0;j<k;j++)
        for (int l=0;l<k;l++)
          c_test[i*n*n + j*n + l] = c_HD[i*n*n + j*n + l];

    // rows
    for (int slice=0; slice<k; slice++) {
        for (int row=0; row<k; row++) {
            for (int col=k; col<n; col++) {
                int parity_val=0;
                for (int m=0;m<k;m++) {
                    int msg_idx = slice*n*n + row*n + m;
                    parity_val += c_test[msg_idx] * G_flat[m*n + col];
                }
                c_test[slice*n*n + row*n + col] = parity_val & 1;
            }
        }
    }

    // cols
    for (int slice=0; slice<k; slice++) {
        for (int col=0; col<n; col++) {
            for (int row=k; row<n; row++) {
                int parity_val=0;
                for (int m=0;m<k;m++) {
                    int msg_idx = slice*n*n + m*n + col;
                    parity_val += c_test[msg_idx] * G_flat[m*n + row];
                }
                c_test[slice*n*n + row*n + col] = parity_val & 1;
            }
        }
    }

    // slices
    for (int row=0; row<n; row++) {
        for (int col=0; col<n; col++) {
            for (int slice=k; slice<n; slice++) {
                int parity_val=0;
                for (int m=0;m<k;m++) {
                    int msg_idx = m*n*n + row*n + col;
                    parity_val += c_test[msg_idx] * G_flat[m*n + slice];
                }
                c_test[slice*n*n + row*n + col] = parity_val & 1;
            }
        }
    }

    // compare
    int match = 1;
    for (int i=0;i<cw_size;i++) { if (c_test[i] != c_HD[i]) { match = 0; break; } }
    result_flags[b] = match;
}

// ===================== Host utilities =====================

void hard_decision(double* llr, int* bits, int length) {
    const int k = 10, n = 15;
    int bit_idx = 0;
    int expected = k*k*k;
    if (length != expected) {
        fprintf(stderr, "Warning: hard_decision length mismatch (got %d exp %d)\n", length, expected);
    }
    for (int slice=0;slice<k;slice++)
      for (int row=0;row<k;row++)
        for (int col=0;col<k;col++) {
            int tensor_idx = slice*n*n + row*n + col;
            bits[bit_idx++] = (llr[tensor_idx] > 0) ? 0 : 1;
        }
}

// Single-call: result = vec1 + alpha*vec2 over N
static inline void cublas_vector_add(cublasHandle_t handle, double* d_result,
                                     const double* d_vec1, const double* d_vec2,
                                     double alpha, int64_t N) {
    // d_result = d_vec1
    CUBLAS_CHECK(cublasDcopy(handle, (int)N, d_vec1, 1, d_result, 1));
    // d_result += alpha * d_vec2
    CUBLAS_CHECK(cublasDaxpy(handle, (int)N, &alpha, d_vec2, 1, d_result, 1));
}

// Stream buffer struct
typedef struct {
    cudaStream_t   stream;
    cublasHandle_t cublas_handle;

    double* h_llr_buffer;     // pinned: batch * cw_size doubles
    double* h_output_buffer;  // pinned: batch * cw_size doubles

    double* d_L_channel;
    double* d_L_APP;
    double* d_L_E;
    double* d_input;

    int*    d_NG_count;           // [batch]
    int*    h_NG_count;           // [batch]
    int*    d_early_term_result;  // [batch]
    int*    h_early_term_result;  // [batch]

    int     batch_blocks;         // filled in this batch
    int     blocks_processed;     // completion flag
} StreamData;

void initStreamData(StreamData* sd, int codeword_block_size, int batch_blocks) {
    CUDA_CHECK(cudaStreamCreate(&sd->stream));
    CUBLAS_CHECK(cublasCreate(&sd->cublas_handle));
    CUBLAS_CHECK(cublasSetStream(sd->cublas_handle, sd->stream));

    size_t elems = (size_t)batch_blocks * codeword_block_size;
    CUDA_CHECK(cudaHostAlloc(&sd->h_llr_buffer,    elems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&sd->h_output_buffer, elems * sizeof(double), cudaHostAllocDefault));

    sd->h_NG_count          = (int*)malloc(sizeof(int) * (size_t)batch_blocks);
    sd->h_early_term_result = (int*)malloc(sizeof(int) * (size_t)batch_blocks);

    CUDA_CHECK(cudaMalloc(&sd->d_L_channel, elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sd->d_L_APP,     elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sd->d_L_E,       elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sd->d_input,     elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sd->d_NG_count,  (size_t)batch_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sd->d_early_term_result, (size_t)batch_blocks * sizeof(int)));

    sd->batch_blocks = 0;
    sd->blocks_processed = 0;
}
void cleanupStreamData(StreamData* sd) {
    CUBLAS_CHECK(cublasDestroy(sd->cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(sd->stream));
    CUDA_CHECK(cudaFreeHost(sd->h_llr_buffer));
    CUDA_CHECK(cudaFreeHost(sd->h_output_buffer));
    free(sd->h_NG_count);
    free(sd->h_early_term_result);
    CUDA_CHECK(cudaFree(sd->d_L_channel));
    CUDA_CHECK(cudaFree(sd->d_L_APP));
    CUDA_CHECK(cudaFree(sd->d_L_E));
    CUDA_CHECK(cudaFree(sd->d_input));
    CUDA_CHECK(cudaFree(sd->d_NG_count));
    CUDA_CHECK(cudaFree(sd->d_early_term_result));
}

// ===================== Main =====================

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Usage: %s <input_llr_file> <output_file> [streams=%d] [batch=%d]\n",
                argv[0], MAX_STREAMS, BATCH_PER_STREAM);
        return 1;
    }
    const char* input_filename  = argv[1];
    const char* output_filename = argv[2];
    int streams_used  = (argc >= 4) ? atoi(argv[3]) : MAX_STREAMS;
    int batch_target  = (argc >= 5) ? atoi(argv[4]) : BATCH_PER_STREAM;
    if (streams_used <= 0 || streams_used > MAX_STREAMS) streams_used = MAX_STREAMS;
    if (batch_target <= 0) batch_target = BATCH_PER_STREAM;

    const int n = 15;
    const int k = 10;
    const int codeword_block_size = n * n * n;  // 3375 doubles
    const int message_block_size  = k * k * k;  // 1000 bits

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | GlobalMem: %.1f GB | UnifiedAddr: %d\n",
           prop.name, prop.multiProcessorCount, prop.totalGlobalMem/1073741824.0, prop.unifiedAddressing);

    // Build G, H on host
    int** G = create_int_matrix(k, n);
    int** H = create_int_matrix(n - k, n);
    getGH_sys_CRC(n, k, G, H);

    // Check even code
    int even = 1;
    for (int i=0;i<k;i++) {
        int row_sum=0; for (int j=0;j<n;j++) row_sum += G[i][j];
        if (row_sum & 1) { even = 0; break; }
    }

    const int L = 3;
    const int Imax = 30;
    const uint64_t Tmax = UINT64_MAX;
    const double p_ET = 1e-5;
    const double thres = 1.0 - p_ET;

    double alpha[100];
    for (int i=0;i<100;i++) alpha[i] = 0.7;

    // Flatten H and G to device
    uint8_t* d_H_flat;  int* d_G_flat;
    CUDA_CHECK(cudaMalloc(&d_H_flat, (size_t)(n-k)*n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_G_flat, (size_t)k*n * sizeof(int)));

    uint8_t* h_H_flat = (uint8_t*)malloc((size_t)(n-k)*n * sizeof(uint8_t));
    int*     h_G_flat = (int*)    malloc((size_t)k*n * sizeof(int));
    for (int i=0;i<(n-k);i++) for (int j=0;j<n;j++) h_H_flat[i*n + j] = (uint8_t)H[i][j];
    for (int i=0;i<k;i++)     for (int j=0;j<n;j++) h_G_flat[i*n + j] = G[i][j];
    CUDA_CHECK(cudaMemcpy(d_H_flat, h_H_flat, (size_t)(n-k)*n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G_flat, h_G_flat, (size_t)k*n * sizeof(int),        cudaMemcpyHostToDevice));
    free(h_H_flat); free(h_G_flat);

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }
    setvbuf(fin,  NULL, _IOFBF, 8*1024*1024);
    setvbuf(fout, NULL, _IOFBF, 8*1024*1024);

    printf("Decoding %s -> %s   (n=%d, k=%d, L=%d, Imax=%d) | streams=%d, batch=%d\n",
           input_filename, output_filename, n, k, L, Imax, streams_used, batch_target);

    StreamData streams[MAX_STREAMS];
    for (int i=0;i<streams_used;i++) initStreamData(&streams[i], codeword_block_size, batch_target);

    int total_blocks = 0;
    long long total_NG = 0;
    long long total_NG_p = 0;  // we track per-batch max inside kernels? (optional)
    double total_iterations = 0.0; // (optional if you later add per-iter counters)
    clock_t start_time = clock();

    int current_stream = 0;
    int batches_in_flight = 0;
    bool done_reading = false;

    while (!done_reading || batches_in_flight > 0) {
        // Push a new batch if capacity
        if (!done_reading && batches_in_flight < streams_used) {
            StreamData* S = &streams[current_stream];

            size_t want_elems = (size_t)batch_target * codeword_block_size;
            size_t got_elems  = fread(S->h_llr_buffer, sizeof(double), want_elems, fin);

            if (got_elems >= (size_t)codeword_block_size) {
                int blocks = (int)(got_elems / (size_t)codeword_block_size);
                S->batch_blocks = blocks;

                // H2D
                CUDA_CHECK(cudaMemcpyAsync(S->d_L_channel, S->h_llr_buffer,
                                           (size_t)blocks * codeword_block_size * sizeof(double),
                                           cudaMemcpyHostToDevice, S->stream));

                // zero buffers
                CUDA_CHECK(cudaMemsetAsync(S->d_L_E, 0,
                    (size_t)blocks * codeword_block_size * sizeof(double), S->stream));
                CUDA_CHECK(cudaMemsetAsync(S->d_NG_count, 0,
                    (size_t)blocks * sizeof(int), S->stream));
                CUDA_CHECK(cudaMemsetAsync(S->d_early_term_result, 0,
                    (size_t)blocks * sizeof(int), S->stream));

                dim3 grid_n_by_B(n, blocks, 1);
                dim3 block_n(n, 1, 1);

                for (int iter=1; iter<=Imax; iter++) {
                    // === COLUMNS — alpha[2*iter-2] ===
                    #if USE_CUBLAS_FOR_AXPY
                        cublas_vector_add(S->cublas_handle,
                                          S->d_input, S->d_L_channel, S->d_L_E,
                                          alpha[2*iter-2],
                                          (int64_t)blocks * codeword_block_size);
                    #else
                        {
                          int total = blocks * codeword_block_size;
                          int threads = 256;
                          int blocks_g = (total + (threads*2 - 1)) / (threads*2);
                          update_input_kernel<<<blocks_g, threads, 0, S->stream>>>(
                              S->d_input, S->d_L_channel, S->d_L_E,
                              alpha[2*iter-2], total);
                        }
                    #endif

                    sogrand_columns_kernel<<<grid_n_by_B, block_n, 0, S->stream>>>(
                        S->d_input, S->d_L_APP, S->d_L_E,
                        d_H_flat, n, k, L, Tmax, thres, even,
                        S->d_NG_count, blocks);

                    // Early termination check
                    early_termination_kernel<<<blocks, 1, 0, S->stream>>>(
                        S->d_L_APP, d_G_flat, n, k, blocks, S->d_early_term_result);

                    CUDA_CHECK(cudaMemcpyAsync(S->h_early_term_result, S->d_early_term_result,
                                               (size_t)blocks * sizeof(int),
                                               cudaMemcpyDeviceToHost, S->stream));
                    CUDA_CHECK(cudaStreamSynchronize(S->stream));
                    bool all_done = true;
                    for (int b=0;b<blocks;b++) if (S->h_early_term_result[b]==0) { all_done=false; break; }
                    if (all_done) break;

                    // === ROWS — alpha[2*iter-1] ===
                    #if USE_CUBLAS_FOR_AXPY
                        cublas_vector_add(S->cublas_handle,
                                          S->d_input, S->d_L_channel, S->d_L_E,
                                          alpha[2*iter-1],
                                          (int64_t)blocks * codeword_block_size);
                    #else
                        {
                          int total = blocks * codeword_block_size;
                          int threads = 256;
                          int blocks_g = (total + (threads*2 - 1)) / (threads*2);
                          update_input_kernel<<<blocks_g, threads, 0, S->stream>>>(
                              S->d_input, S->d_L_channel, S->d_L_E,
                              alpha[2*iter-1], total);
                        }
                    #endif

                    sogrand_rows_kernel<<<grid_n_by_B, block_n, 0, S->stream>>>(
                        S->d_input, S->d_L_APP, S->d_L_E,
                        d_H_flat, n, k, L, Tmax, thres, even,
                        S->d_NG_count, blocks);

                    early_termination_kernel<<<blocks, 1, 0, S->stream>>>(
                        S->d_L_APP, d_G_flat, n, k, blocks, S->d_early_term_result);

                    CUDA_CHECK(cudaMemcpyAsync(S->h_early_term_result, S->d_early_term_result,
                                               (size_t)blocks * sizeof(int),
                                               cudaMemcpyDeviceToHost, S->stream));
                    CUDA_CHECK(cudaStreamSynchronize(S->stream));
                    all_done = true;
                    for (int b=0;b<blocks;b++) if (S->h_early_term_result[b]==0) { all_done=false; break; }
                    if (all_done) break;

                    // === SLICES — reuse alpha[2*iter-1] (exactly like CPU) ===
                    #if USE_CUBLAS_FOR_AXPY
                        cublas_vector_add(S->cublas_handle,
                                          S->d_input, S->d_L_channel, S->d_L_E,
                                          alpha[2*iter-1],
                                          (int64_t)blocks * codeword_block_size);
                    #else
                        {
                          int total = blocks * codeword_block_size;
                          int threads = 256;
                          int blocks_g = (total + (threads*2 - 1)) / (threads*2);
                          update_input_kernel<<<blocks_g, threads, 0, S->stream>>>(
                              S->d_input, S->d_L_channel, S->d_L_E,
                              alpha[2*iter-1], total);
                        }
                    #endif

                    sogrand_slices_kernel<<<grid_n_by_B, block_n, 0, S->stream>>>(
                        S->d_input, S->d_L_APP, S->d_L_E,
                        d_H_flat, n, k, L, Tmax, thres, even,
                        S->d_NG_count, blocks);

                    early_termination_kernel<<<blocks, 1, 0, S->stream>>>(
                        S->d_L_APP, d_G_flat, n, k, blocks, S->d_early_term_result);

                    CUDA_CHECK(cudaMemcpyAsync(S->h_early_term_result, S->d_early_term_result,
                                               (size_t)blocks * sizeof(int),
                                               cudaMemcpyDeviceToHost, S->stream));
                    CUDA_CHECK(cudaStreamSynchronize(S->stream));
                    all_done = true;
                    for (int b=0;b<blocks;b++) if (S->h_early_term_result[b]==0) { all_done=false; break; }
                    if (all_done) break;
                }

                // D2H final L_APP for the whole batch
                CUDA_CHECK(cudaMemcpyAsync(S->h_output_buffer, S->d_L_APP,
                                           (size_t)blocks * codeword_block_size * sizeof(double),
                                           cudaMemcpyDeviceToHost, S->stream));

                // (Optional) D2H NG counters
                CUDA_CHECK(cudaMemcpyAsync(S->h_NG_count, S->d_NG_count,
                                           (size_t)blocks * sizeof(int),
                                           cudaMemcpyDeviceToHost, S->stream));

                S->blocks_processed = 1;
                batches_in_flight++;
                current_stream = (current_stream + 1) % streams_used;
            } else {
                done_reading = true;
            }
        }

        // Pull back completed batches and write out
        for (int s=0;s<streams_used;s++) {
            StreamData* S = &streams[s];
            if (S->blocks_processed == 0) continue;

            cudaError_t st = cudaStreamQuery(S->stream);
            if (st == cudaSuccess) {
                // Host-side hard decision and packing per codeword
                for (int b = 0; b < S->batch_blocks; b++) {
                    double* LAPP = S->h_output_buffer + (size_t)b * codeword_block_size;

                    int bit_buffer[message_block_size];
                    unsigned char byte_out = 0; int bit_count = 0;

                    hard_decision(LAPP, bit_buffer, message_block_size);
                    for (int i=0;i<message_block_size;i++) {
                        byte_out = (unsigned char)((byte_out << 1) | (bit_buffer[i] & 1));
                        bit_count++;
                        if (bit_count == 8) {
                            fwrite(&byte_out, 1, 1, fout);
                            byte_out = 0; bit_count = 0;
                        }
                    }
                    if (bit_count > 0) {
                        byte_out <<= (8 - bit_count);
                        fwrite(&byte_out, 1, 1, fout);
                    }
                    total_blocks++;
                }

                // Accumulate NG (sum of per-codeword counts)
                for (int b=0;b<S->batch_blocks;b++) total_NG += S->h_NG_count[b];

                S->blocks_processed = 0;
                batches_in_flight--;
            } else if (st != cudaErrorNotReady && st != cudaSuccess) {
                CUDA_CHECK(st);
            }
        }

        if (batches_in_flight >= streams_used) usleep(1000); // light backoff
    }

    // Drain
    for (int i=0;i<streams_used;i++) CUDA_CHECK(cudaStreamSynchronize(streams[i].stream));

    clock_t end_time = clock();
    double secs = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("CUDA Streaming Decoding complete. %d block(s).\n", total_blocks);
    if (secs > 0 && total_blocks > 0) {
        printf("Time: %.2f s | Throughput: %.2f blocks/s | Avg NG/block: %.2f\n",
               secs, total_blocks/secs, (double)total_NG / (double)total_blocks);
    }

    // Cleanup
    for (int i=0;i<streams_used;i++) cleanupStreamData(&streams[i]);
    CUDA_CHECK(cudaFree(d_H_flat));
    CUDA_CHECK(cudaFree(d_G_flat));
    fclose(fin); fclose(fout);
    free_int_matrix(G, k);
    free_int_matrix(H, n-k);
    return 0;
}
