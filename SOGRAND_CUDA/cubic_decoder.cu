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
#include <cooperative_groups.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Inf 0x7fffffff

// ===================== Enhanced Tunables =====================
#ifndef MAX_STREAMS
#define MAX_STREAMS 36
#endif

#define THREADS_PER_BLOCK 1024

#ifndef BATCH_PER_STREAM
#define BATCH_PER_STREAM 256
#endif

// ====================================================

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static const double EPSILON = 1e-40;

namespace cg = cooperative_groups;

// Constant memory for H and G matrices
__constant__ uint8_t d_H_const[8*16];
__constant__ int d_G_const[8*16];
__constant__ double d_alpha_const[100];

// Forward declaration of IDX function
__host__ __device__ __forceinline__ int IDX(int row, int col, int slice, int n) {
  return slice * n * n + row * n + col;
}

#define CUDA_CHECK(call) \
do { \
  cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1);} \
} while(0)

#define CUBLAS_CHECK(call) \
do { \
  cublasStatus_t status = (call); \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, status); \
    exit(1);} \
} while(0)

// Enhanced stream data structure
typedef struct {
  cudaStream_t stream;
  cudaEvent_t event;

  // Device memory - aligned for optimal access
  double* d_input;
  double* d_L_channel;
  double* d_L_APP;
  double* d_L_E;
  double* d_temp_buffer;
  int* d_convergence_flags;
  int* d_NG_count;
  double* d_iter_count;

  // Additional workspace for optimizations
  double* d_matrix_workspace;
  int* d_int_workspace;

  // Host memory - page-locked
  double* h_input_buffer;
  double* h_output_buffer;
  int* h_convergence_flags;
  int* h_NG_count;
  double* h_iter_count;

  int blocks_processed;
  size_t batch_id;
} StreamData;

__device__ __forceinline__ double4 load_double4_aligned(const double* addr) {
  return *reinterpret_cast<const double4*>(addr);
}

__device__ __forceinline__ void store_double4_aligned(double* addr, double4 val) {
  *reinterpret_cast<double4*>(addr) = val;
}

__device__ __forceinline__ int4 load_int4_aligned(const int* addr) {
  return *reinterpret_cast<const int4*>(addr);
}

__device__ __forceinline__ void store_int4_aligned(int* addr, int4 val) {
  *reinterpret_cast<int4*>(addr) = val;
}

// Vector math operations with FMA
__device__ __forceinline__ double4 fma_double4(double4 a, double alpha, double4 b) {
  return make_double4(
    fma(alpha, b.x, a.x),
    fma(alpha, b.y, a.y),
    fma(alpha, b.z, a.z),
    fma(alpha, b.w, a.w)
  );
}

// Vector helper functions
__device__ __forceinline__ double2 make_double2_safe(double x, double y) {
  return make_double2(x, y);
}

__device__ __forceinline__ double4 make_double4_safe(double x, double y, double z, double w) {
  return make_double4(x, y, z, w);
}

__device__ __forceinline__ int4 make_int4_safe(int x, int y, int z, int w) {
  return make_int4(x, y, z, w);
}

__device__ __forceinline__ uint4 make_uint4_safe(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
  return make_uint4(x, y, z, w);
}

__device__ __forceinline__ double2 add_double2_scaled(double2 a, double2 b, double alpha) {
  return make_double2(a.x + alpha * b.x, a.y + alpha * b.y);
}

__device__ __forceinline__ double4 add_double4_scaled(double4 a, double4 b, double alpha) {
  return make_double4(
    a.x + alpha * b.x,
    a.y + alpha * b.y,
    a.z + alpha * b.z,
    a.w + alpha * b.w
  );
}

__device__ __forceinline__ int4 hard_decision_double4(double4 llr) {
  return make_int4(
    (llr.x > 0.0) ? 0 : 1,
    (llr.y > 0.0) ? 0 : 1,
    (llr.z > 0.0) ? 0 : 1,
    (llr.w > 0.0) ? 0 : 1
  );
}

__device__ __forceinline__ uint4 xor_parity_uint4(uint4 c_vec, uint4 h_vec) {
  return make_uint4(
    c_vec.x * h_vec.x,
    c_vec.y * h_vec.y,
    c_vec.z * h_vec.z,
    c_vec.w * h_vec.w
  );
}

// ===================== Device Helper Functions (Vectorized) =====================
__device__ uint8_t ParityCheck_device_vectorized(uint8_t *c, const uint8_t *H, uint64_t n, uint64_t s) {
  for (size_t j = 0; j < s; j++) {
    uint8_t syndrome = 0;
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
      uint4 c_vec = make_uint4_safe(c[i], c[i+1], c[i+2], c[i+3]);
      uint4 h_vec = make_uint4_safe(H[j*n + i], H[j*n + i+1], H[j*n + i+2], H[j*n + i+3]);
      uint4 result = xor_parity_uint4(c_vec, h_vec);
      syndrome ^= result.x ^ result.y ^ result.z ^ result.w;
    }
    for (; i < n; i++) {
      syndrome ^= (c[i] * H[j*n + i]);
    }
    if (syndrome == 1) return 0;
  }
  return 1;
}

__device__ int32_t findMax_device(int32_t a, int32_t b) {
  return !(b > a) ? a : b;
}

__device__ void HardDec_device_vectorized(uint8_t *c, double *llr, uint64_t n) {
  size_t i = 0;
  for (; i + 3 < n; i += 4) {
    double4 llr_vec = make_double4_safe(llr[i], llr[i+1], llr[i+2], llr[i+3]);
    int4 hd_vec = hard_decision_double4(llr_vec);
    c[i] = hd_vec.x;
    c[i+1] = hd_vec.y;
    c[i+2] = hd_vec.z;
    c[i+3] = hd_vec.w;
  }
  for (; i < n; i++) c[i] = (llr[i] > 0.0) ? 0 : 1;
}

__device__ int parity_device_vectorized(uint8_t array[], uint64_t n) {
  int sum = 0;
  size_t i = 0;
  for (; i + 3 < n; i += 4) {
    int4 arr_vec = make_int4_safe(array[i], array[i+1], array[i+2], array[i+3]);
    sum += arr_vec.x + arr_vec.y + arr_vec.z + arr_vec.w;
  }
  for (; i < n; i++) sum += array[i];
  return sum % 2;
}

__device__ double prob_parity_device(int parity_cHD, double *absL, uint64_t n) {
  double p_e = 1.0;
  for (uint64_t i = 0; i < n; i++) {
    p_e *= (1.0 - 2.0 * exp(-absL[i]) / (1.0 + exp(-absL[i])));
  }
  p_e = 0.5 * (1.0 + p_e);
  return (parity_cHD == 0) ? p_e : 1.0 - p_e;
}

__device__ void AddTEP_device(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n) {
  for (size_t i = 0; i < n; i++) c[perm[i]] = cHD[perm[i]] ^ TEP[i];
}

__device__ double JacLog_device(double x) {
  if (x > 50) return x;
  if (x < -50) return 0.0;
  return log(1.0 + exp(x));
}

__device__ void QuickSort_device(double *a, size_t *perm, uint64_t n) {
  if (n < 2) return;
  double p = a[n / 2];
  uint64_t i = 0, j = n - 1;
  while (i <= j) {
    while (a[i] < p) i++;
    while (a[j] > p) j--;
    if (i <= j) {
      double t = a[i]; a[i] = a[j]; a[j] = t;
      size_t tt = perm[i]; perm[i] = perm[j]; perm[j] = tt;
      i++;
      if (j > 0) j--;
    }
  }
  if (j > 0) QuickSort_device(a, perm, j + 1);
  if (i < n) QuickSort_device(a + i, perm + i, n - i);
}

__device__ double getPM_HD_device(double *absL, uint64_t n) {
  double pm = 0;
  for(size_t i=0; i<n; i++) pm += JacLog_device(-absL[i]);
  return pm;
}

__device__ double getPM_device(uint8_t *TEP, double *absL, double PM_HD, uint64_t n) {
  double pm = PM_HD;
  for(size_t i = 0; i < n; i++) {
    if (TEP[i] == 1) pm += (JacLog_device(absL[i]) - JacLog_device(-absL[i]));
  }
  return pm;
}

__device__ double getLConf_device(double *pNL, double P_notGuess, uint64_t cur_L, double *score, uint64_t s, uint8_t even) {
  double P_pos = 0.0;
  for(size_t i = 0; i <= cur_L; i++) P_pos += exp(-score[4*i+1]);
  if(even==1) s--;
  double P_neg = pow(2.0, -(double)s) * P_notGuess;
  pNL[0] = P_neg;
  return (P_pos + P_neg > EPSILON) ? (P_pos / (P_pos + P_neg)) : 1.0;
}

__device__ void mountain_build_device(int32_t *u, int32_t k, int32_t w, int32_t W1, int32_t n1) {
  for(size_t i = k + 1; i < w; i++)
    u[i] = u[k];
  uint64_t W2 = W1;
  for(size_t i = 0; i < w; i++)
    W2 -= u[i];

  uint64_t q = (uint64_t)floor((double)W2 / (double)(n1 - u[k]));
  uint64_t r = W2 - q*(n1 - u[k]);
  if (q != 0){
    for(size_t i = w-q; i < w; i++)
      u[i] = n1;
  }
  if (w > q)
    u[w-q-1] = u[w-q-1] + r;
}

__device__ void getAPP_device(uint64_t cur_L, double *score, double *APP) {
  if (cur_L == 0) return;
  double P_positive = 0.0;
  for(size_t i=0; i<cur_L; i++) P_positive += exp(-score[4*i+1]);
  if (P_positive < EPSILON) return;
  double den = score[4*(cur_L-1)+3] / P_positive;
  for(size_t i=0; i<cur_L; i++) APP[i] = exp(-score[4*i+1]) * den;
}

// ===================== Complete SOGRAND Device Implementation =====================
__device__ void sogrand_main_logic_device(double* chat_list, double* s_list, double* T_val,
                                          double* curL_val, double* pNL_val, double* APP_list,
                                          double* llr, const uint8_t* H_flat, int n, int s,
                                          int IC, uint64_t L, uint64_t Tmax, double thres, int even) {
  size_t perm[16];
  uint8_t cHD[16];
  uint8_t TEP[16];
  uint8_t c[16];
  double absL[16];
  int32_t u[16], d[16], D[16];

  // Initialize arrays exactly as in C code
  for(size_t i = 0; i < n; i++) perm[i] = i;
  for(size_t i = 0; i < 4*L; i++) s_list[i] = 0;
  for(size_t i = 0; i < L; i++) APP_list[i] = 0.0;

  uint64_t cur_L = 0;
  HardDec_device_vectorized(cHD, llr, n);
  uint8_t parity_cHD = parity_device_vectorized(cHD, n);
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
    double beta = (absL[(uint64_t)round((double)n/2) - 1] - absL[0]) / (round((double)n/2) - 1);
    IC = findMax_device((int32_t)round(absL[0]/beta - 1), 0);
  }

  AddTEP_device(c, cHD, TEP, perm, n);
  T_val[0] = 1;
  if (parity_cHD == 0 || even == 0) P_notGuess -= exp(-getPM_device(TEP, absL, PM_HD, n));

  if (ParityCheck_device_vectorized(c, H_flat, n, s) == 1) {
    s_list[0] = 0;
    s_list[1] = getPM_device(TEP, absL, PM_HD, n);
    for(size_t i = 0; i < n; i++) chat_list[i] = c[i];
    s_list[2] = 1;
    s_list[3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
    cur_L++;
    curL_val[0] = (double)cur_L;
    if ((s_list[3] > thres) || (cur_L == L)) {
      getAPP_device(cur_L, s_list, APP_list);
      return;
    }
  }

  // Full enumeration loop - exactly following C code
  int32_t w = 0;
  int parity_w;
  int32_t W = 0;
  int32_t wt = IC + 1;
  int32_t wt_max = IC*n + n*(n+1)/2;
  int32_t W1;
  int32_t n1;
  int32_t k_mt;
  double temp = 1 + 2*((double)n + (double)IC);

  while ((cur_L < L) && (T_val[0] < Tmax) && (wt <= wt_max)) {
    w = findMax_device(1, (int32_t)ceil((temp - sqrt(pow((double)temp, 2.0) - 8*wt))/2));
    parity_w = w % 2;
    if (even == 1 && (w % 2 != parity_cHD)) w++;

    while (w <= n) {
      W = wt - IC*w;
      if (W < w*(w+1)/2) break;

      W1 = W - w*(w+1)/2;
      n1 = n - w;
      for (size_t i = 0; i < w; i++) u[i] = 0;

      mountain_build_device(u, 0, w, W1, n1);

      for (size_t i = 0; i < n; i++) TEP[i] = 0;
      for (size_t i = 0; i < w; i++) TEP[i + u[i]] = 1;
      AddTEP_device(c, cHD, TEP, perm, n);
      T_val[0]++;

      P_notGuess -= exp(-getPM_device(TEP, absL, PM_HD, n));

      if (ParityCheck_device_vectorized(c, H_flat, n, s) == 1) {
        s_list[4*cur_L] = wt;
        s_list[4*cur_L+1] = getPM_device(TEP, absL, PM_HD, n);
        for(size_t i = 0; i < n; i++) chat_list[cur_L*n + i] = c[i];
        s_list[4*cur_L+2] = T_val[0];
        s_list[4*cur_L+3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
        cur_L++;
        curL_val[0] = (double)cur_L;
        if ((s_list[4*(cur_L-1)+3] > thres) || (cur_L == L)) {
          getAPP_device(cur_L, s_list, APP_list);
          return;
        }
      }

      // Mountain descent
      for (size_t i = 0; i < w - 1; i++) d[i] = u[i+1] - u[i];
      d[w-1] = 0;
      D[w-1] = d[w-1];
      for (size_t i = 1; i < w; i++) D[w-i-1] = D[w-i] + d[w-i-1];

      while (D[0] >= 2) {
        k_mt = 0;
        for (size_t i = w-1; i > 0; i--) {
          if (D[i] >= 2) {
            k_mt = i;
            break;
          }
        }

        u[k_mt]++;
        mountain_build_device(u, k_mt, w, W1, n1);

        for (size_t i = 0; i < n; i++) TEP[i] = 0;
        for (size_t i = 0; i < w; i++) TEP[i + u[i]] = 1;

        AddTEP_device(c, cHD, TEP, perm, n);
        T_val[0]++;
        P_notGuess -= exp(-getPM_device(TEP, absL, PM_HD, n));

        if (ParityCheck_device_vectorized(c, H_flat, n, s) == 1) {
          s_list[4*cur_L] = wt;
          s_list[4*cur_L+1] = getPM_device(TEP, absL, PM_HD, n);
          for(size_t i = 0; i < n; i++) chat_list[cur_L*n + i] = c[i];
          s_list[4*cur_L+2] = T_val[0];
          s_list[4*cur_L+3] = getLConf_device(pNL_val, P_notGuess, cur_L, s_list, s, even);
          cur_L++;
          curL_val[0] = (double)cur_L;
          if ((s_list[4*(cur_L-1)+3] > thres) || (cur_L == L)) {
            getAPP_device(cur_L, s_list, APP_list);
            return;
          }
        }

        for (size_t i = 0; i < w - 1; i++) d[i] = u[i+1] - u[i];
        d[w-1] = 0;
        D[w-1] = d[w-1];
        for (size_t i = 1; i < w; i++) D[w-i-1] = D[w-i] + d[w-i-1];
      }

      w++;
      parity_w = w % 2;
      if (even == 1) {
        if (parity_w != parity_cHD) {
          w++;
        }
      }
    }
    wt++;
  }
}

__device__ int SOGRAND_bitSO_device_vectorized(double* L_APP, double* L_E, double* llr,
                                              const uint8_t* H_flat, int n, int k, int L,
                                              unsigned long Tmax, double thres, int even) {
  double chat_list[16*3];
  double s_list[4*3];
  double T_val, curL_val, pNL_val;
  double APP_list[3];

  sogrand_main_logic_device(chat_list, s_list, &T_val, &curL_val, &pNL_val,
                           APP_list, llr, H_flat, n, n-k, -1, L, Tmax, thres, even);

  int curL = (int)curL_val;
  if (curL == 0) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
      double4 llr_vec = make_double4_safe(llr[i], llr[i+1], llr[i+2], llr[i+3]);
      double4 zero_vec = make_double4_safe(0.0, 0.0, 0.0, 0.0);

      L_APP[i] = llr_vec.x; L_APP[i+1] = llr_vec.y; L_APP[i+2] = llr_vec.z; L_APP[i+3] = llr_vec.w;
      L_E[i] = zero_vec.x; L_E[i+1] = zero_vec.y; L_E[i+2] = zero_vec.z; L_E[i+3] = zero_vec.w;
    }
    for (; i < n; i++) {
      L_APP[i] = llr[i];
      L_E[i] = 0.0;
    }
  }
  else {
    double PM[3];
    for(int i = 0; i < curL; ++i) PM[i] = s_list[4*i + 1];
    double p_notinlist = fmax(pNL_val, EPSILON);

    double pp1[16], pp0[16];

    int i = 0;
    for (; i + 1 < n; i += 2) {
      double2 llr_vec = make_double2_safe(llr[i], llr[i+1]);
      double2 exp_vec = make_double2_safe(exp(llr_vec.x), exp(llr_vec.y));
      double2 pp1_vec = make_double2_safe(1.0 / (1.0 + exp_vec.x), 1.0 / (1.0 + exp_vec.y));
      double2 pp0_vec = make_double2_safe(1.0 - pp1_vec.x, 1.0 - pp1_vec.y);

      pp1_vec.x = fmax(fmin(pp1_vec.x, 1.0 - EPSILON), EPSILON);
      pp1_vec.y = fmax(fmin(pp1_vec.y, 1.0 - EPSILON), EPSILON);
      pp0_vec.x = fmax(fmin(pp0_vec.x, 1.0 - EPSILON), EPSILON);
      pp0_vec.y = fmax(fmin(pp0_vec.y, 1.0 - EPSILON), EPSILON);

      pp1[i] = pp1_vec.x; pp1[i+1] = pp1_vec.y;
      pp0[i] = pp0_vec.x; pp0[i+1] = pp0_vec.y;
    }

    for (; i < n; i++) {
      pp1[i] = 1.0 / (1.0 + exp(llr[i]));
      pp0[i] = 1.0 - pp1[i];
      pp1[i] = fmax(fmin(pp1[i], 1.0 - EPSILON), EPSILON);
      pp0[i] = fmax(fmin(pp0[i], 1.0 - EPSILON), EPSILON);
    }

    double p[3];
    for(int i = 0; i < curL; ++i) p[i] = exp(-PM[i]);

    double p1[16] = {0}, p0[16] = {0};

    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < curL; ++j) {
        p1[i] += chat_list[i + j*n] * p[j];
        p0[i] += (1.0 - chat_list[i + j*n]) * p[j];
      }
    }

    i = 0;
    for (; i + 1 < n; i += 2) {
      double2 p0_vec = make_double2_safe(p0[i], p0[i+1]);
      double2 p1_vec = make_double2_safe(p1[i], p1[i+1]);
      double2 pp0_vec = make_double2_safe(pp0[i], pp0[i+1]);
      double2 pp1_vec = make_double2_safe(pp1[i], pp1[i+1]);
      double2 llr_vec = make_double2_safe(llr[i], llr[i+1]);

      p0_vec.x += p_notinlist * pp0_vec.x;
      p0_vec.y += p_notinlist * pp0_vec.y;
      p1_vec.x += p_notinlist * pp1_vec.x;
      p1_vec.y += p_notinlist * pp1_vec.y;

      double2 L_APP_vec = make_double2_safe(
        log(fmax(p0_vec.x, EPSILON)) - log(fmax(p1_vec.x, EPSILON)),
        log(fmax(p0_vec.y, EPSILON)) - log(fmax(p1_vec.y, EPSILON))
      );

      double2 L_E_vec = make_double2_safe(
        L_APP_vec.x - llr_vec.x,
        L_APP_vec.y - llr_vec.y
      );

      L_APP[i] = L_APP_vec.x; L_APP[i+1] = L_APP_vec.y;
      L_E[i] = L_E_vec.x; L_E[i+1] = L_E_vec.y;
    }

    for (; i < n; i++) {
      p0[i] += p_notinlist * pp0[i];
      p1[i] += p_notinlist * pp1[i];
      L_APP[i] = log(fmax(p0[i], EPSILON)) - log(fmax(p1[i], EPSILON));
      L_E[i] = L_APP[i] - llr[i];
    }
  }

  return (int)T_val;
}

// ===================== FIXED GPU-based Convergence Check =====================
__device__ int check_convergence_single_block(const double* __restrict__ L_APP,
                                              int n, int k, int batch_idx, int cw_size) {
  const double* batch_L_APP = L_APP + (size_t)batch_idx * cw_size;

  uint8_t c_HD[16*16*16];
  uint8_t c_test[16*16*16];

  // Hard decision
  int i = 0;
  for (; i + 3 < cw_size; i += 4) {
    double4 llr_vec = make_double4_safe(batch_L_APP[i], batch_L_APP[i+1], batch_L_APP[i+2], batch_L_APP[i+3]);
    int4 hd_vec = hard_decision_double4(llr_vec);
    c_HD[i] = hd_vec.x;
    c_HD[i+1] = hd_vec.y;
    c_HD[i+2] = hd_vec.z;
    c_HD[i+3] = hd_vec.w;
  }
  for (; i < cw_size; i++) {
    c_HD[i] = (batch_L_APP[i] > 0.0) ? 0 : 1;
  }

  // Initialize c_test with zeros
  for (i = 0; i < cw_size; i++) {
    c_test[i] = 0;
  }

  // 2a. Copy systematic message part (k×k×k region)
  for (int slice = 0; slice < k; slice++) {
    for (int row = 0; row < k; row++) {
      for (int col = 0; col < k; col++) {
        c_test[IDX(row, col, slice, n)] = c_HD[IDX(row, col, slice, n)];
      }
    }
  }

  // 2b. Encode rows (parity for columns k through n-1)
  for (int slice = 0; slice < k; slice++) {
    for (int row = 0; row < k; row++) {
      for (int col = k; col < n; col++) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
          parity_val += c_test[IDX(row, msg_bit_idx, slice, n)] * d_G_const[msg_bit_idx * n + col];
        }
        c_test[IDX(row, col, slice, n)] = parity_val % 2;
      }
    }
  }

  // 2c. Encode columns (parity for rows k through n-1)
  for (int slice = 0; slice < k; slice++) {
    for (int col = 0; col < n; col++) {
      for (int row = k; row < n; row++) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
          parity_val += c_test[IDX(msg_bit_idx, col, slice, n)] * d_G_const[msg_bit_idx * n + row];
        }
        c_test[IDX(row, col, slice, n)] = parity_val % 2;
      }
    }
  }

  // 2d. Encode slices (parity for slices k through n-1)
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      for (int slice = k; slice < n; slice++) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
          parity_val += c_test[IDX(row, col, msg_bit_idx, n)] * d_G_const[msg_bit_idx * n + slice];
        }
        c_test[IDX(row, col, slice, n)] = parity_val % 2;
      }
    }
  }

  // 3. Check if re-encoded c_test matches c_HD
  for (i = 0; i < cw_size; i++) {
    if (c_test[i] != c_HD[i]) {
      return 0; // Not converged
    }
  }

  return 1; // Converged
}

// ===================== FIXED Unified GPU Iteration Kernel =====================
__global__ void unified_sogrand_iterations_kernel(
    const double* __restrict__ L_channel,
    double* __restrict__ L_APP,
    double* __restrict__ L_E,
    double* __restrict__ temp_buffer,
    int* __restrict__ convergence_flags,
    int* __restrict__ NG_count,
    double* __restrict__ iter_count_output,
    int n, int k, int L, uint64_t Tmax, double thres, int even,
    int max_iterations, int total_batches) {

    int batch_id = blockIdx.x;
    if (batch_id >= total_batches) return;

    int work_id = threadIdx.x;
    if (work_id >= n * n) return;

    const int cw_size = n * n * n;
    size_t base = (size_t)batch_id * cw_size;

    // Shared memory for coordination
    __shared__ int iteration_count;
    __shared__ int batch_converged;
    __shared__ int total_NG_count;
    __shared__ double phase_count;

    if (threadIdx.x == 0) {
        iteration_count = 1;
        batch_converged = 0;
        total_NG_count = 0;
        phase_count = 0.0;  // FIXED: Start from 0 like C code
    }
    __syncthreads();

    // Initialize L_APP with L_channel and L_E with zeros
    if (work_id < n*n) {
        for (int slice = 0; slice < n; slice++) {
            int idx = IDX(work_id/n, work_id%n, slice, n);
            L_APP[base + idx] = L_channel[base + idx];
            L_E[base + idx] = 0.0;
        }
    }
    __syncthreads();

    // FIXED: Match C code iteration structure exactly
    while (iteration_count <= max_iterations && !batch_converged) {

        // ===== COLUMNS PHASE =====
        if (threadIdx.x == 0) {
            phase_count += 0.5;  // FIXED: Increment like C code
        }
        __syncthreads();

        if (work_id < n*n) {
            int slice = work_id / n;
            int col = work_id % n;

            double vec_in[16];
            double vec_L_APP[16];
            double vec_L_E[16];

            // FIXED: Use correct alpha indexing matching C code
            for (int row = 0; row < n; row++) {
                vec_in[row] = L_channel[base + IDX(row, col, slice, n)] +
                             d_alpha_const[2*iteration_count-2] * L_E[base + IDX(row, col, slice, n)];
            }

            int N_guess = SOGRAND_bitSO_device_vectorized(vec_L_APP, vec_L_E, vec_in,
                                                         d_H_const, n, k, L, Tmax, thres, even);

            for (int row = 0; row < n; row++) {
                L_APP[base + IDX(row, col, slice, n)] = vec_L_APP[row];
                L_E[base + IDX(row, col, slice, n)] = vec_L_E[row];
            }

            if (N_guess > 0) {
                atomicAdd(&total_NG_count, N_guess);
            }
        }
        __syncthreads();

        // FIXED: Check convergence after columns phase like C code
        if (threadIdx.x == 0) {
            if (check_convergence_single_block(L_APP, n, k, batch_id, cw_size)) {
                batch_converged = 1;
            }
        }
        __syncthreads();

        if (batch_converged) break;  // FIXED: Early termination after columns

        // ===== ROWS PHASE =====
        if (threadIdx.x == 0) {
            phase_count += 0.5;  // FIXED: Increment like C code
        }
        __syncthreads();

        if (work_id < n*n) {
            int slice = work_id / n;
            int row = work_id % n;

            double vec_in[16];
            double vec_L_APP[16];
            double vec_L_E[16];

            // FIXED: Use correct alpha indexing matching C code
            for (int col = 0; col < n; col++) {
                vec_in[col] = L_channel[base + IDX(row, col, slice, n)] +
                             d_alpha_const[2*iteration_count-1] * L_E[base + IDX(row, col, slice, n)];
            }

            int N_guess = SOGRAND_bitSO_device_vectorized(vec_L_APP, vec_L_E, vec_in,
                                                         d_H_const, n, k, L, Tmax, thres, even);

            for (int col = 0; col < n; col++) {
                L_APP[base + IDX(row, col, slice, n)] = vec_L_APP[col];
                L_E[base + IDX(row, col, slice, n)] = vec_L_E[col];
            }

            if (N_guess > 0) {
                atomicAdd(&total_NG_count, N_guess);
            }
        }
        __syncthreads();

        // FIXED: Check convergence after rows phase like C code
        if (threadIdx.x == 0) {
            if (check_convergence_single_block(L_APP, n, k, batch_id, cw_size)) {
                batch_converged = 1;
            }
        }
        __syncthreads();

        if (batch_converged) break;  // FIXED: Early termination after rows

        // ===== SLICES PHASE =====
        if (threadIdx.x == 0) {
            phase_count += 0.5;  // FIXED: Increment like C code
        }
        __syncthreads();

        if (work_id < n*n) {
            int row = work_id / n;
            int col = work_id % n;

            double vec_in[16];
            double vec_L_APP[16];
            double vec_L_E[16];

            // FIXED: Use correct alpha indexing matching C code
            for (int slice = 0; slice < n; slice++) {
                vec_in[slice] = L_channel[base + IDX(row, col, slice, n)] +
                               d_alpha_const[2*iteration_count-1] * L_E[base + IDX(row, col, slice, n)];
            }

            int N_guess = SOGRAND_bitSO_device_vectorized(vec_L_APP, vec_L_E, vec_in,
                                                         d_H_const, n, k, L, Tmax, thres, even);

            for (int slice = 0; slice < n; slice++) {
                L_APP[base + IDX(row, col, slice, n)] = vec_L_APP[slice];
                L_E[base + IDX(row, col, slice, n)] = vec_L_E[slice];
            }

            if (N_guess > 0) {
                atomicAdd(&total_NG_count, N_guess);
            }
        }
        __syncthreads();

        // FIXED: Check convergence after slices phase and increment iteration
        if (threadIdx.x == 0) {
            if (check_convergence_single_block(L_APP, n, k, batch_id, cw_size)) {
                batch_converged = 1;
            } else {
                iteration_count++;  // FIXED: Only increment if not converged
            }
        }
        __syncthreads();

        if (batch_converged) break;  // FIXED: Early termination after slices
    }

    // Store final results
    if (threadIdx.x == 0) {
        NG_count[batch_id] = total_NG_count;
        convergence_flags[batch_id] = batch_converged;
        iter_count_output[batch_id] = phase_count;  // FIXED: Store actual iteration count
    }
}

// ===================== Helper Functions Implementation =====================
int** create_int_matrix(int rows, int cols) {
  int** matrix = (int**)malloc(rows * sizeof(int*));
  for(int i = 0; i < rows; i++) {
    matrix[i] = (int*)calloc(cols, sizeof(int));
  }
  return matrix;
}

void free_int_matrix(int** matrix, int rows) {
  for(int i = 0; i < rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

int* koopman2matlab(const char* k_poly, int* poly_len) {
  long long dec_val = strtoll(k_poly, NULL, 16);
  int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
  *poly_len = len + 1;
  int* poly = (int*)malloc(sizeof(int) * (*poly_len));

  for (int i = 0; i < len; i++) {
      poly[i] = (dec_val >> (len - 1 - i)) & 1;
  }
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
  else if (r==8 && k<=8) hex_poly = "0xeb";
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

// ===================== Main Function =====================
int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <input_llr_file> <output_file>\n", argv[0]);
    return 1;
  }

  const char* input_filename = argv[1];
  const char* output_filename = argv[2];

  // Parameters
  const int n = 16;
  const int k = 8;
  const int cw_size = n * n * n;
  const int codeword_block_size = cw_size;
  const int L = 3;
  const int Imax = 30;
  const uint64_t Tmax = UINT64_MAX;
  const double p_ET = 1e-5;
  const double thres = 1.0 - p_ET;

  // Initialize alpha array
  double alpha[100];
  for (int i = 0; i < 100; i++) alpha[i] = 0.7;

  // Get G and H matrices
  int** G = create_int_matrix(k, n);
  int** H = create_int_matrix(n - k, n);
  getGH_sys_CRC(n, k, G, H);

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

  // Flatten matrices for GPU
  uint8_t* H_flat = (uint8_t*)malloc((n-k) * n * sizeof(uint8_t));
  for (int i = 0; i < n-k; i++) {
      for (int j = 0; j < n; j++) {
          H_flat[i*n + j] = (uint8_t)H[i][j];
      }
  }

  int* G_flat = (int*)malloc(k * n * sizeof(int));
  for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
          G_flat[i*n + j] = G[i][j];
      }
  }

  // GPU setup
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 65536));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * 1024 * 1024));

  // Copy matrices to constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(d_H_const, H_flat, (n-k) * n * sizeof(uint8_t)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_G_const, G_flat, k * n * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_alpha_const, alpha, 100 * sizeof(double)));

  // Open files
  FILE* fin = fopen(input_filename, "rb");
  FILE* fout = fopen(output_filename, "wb");
  if (!fin || !fout) {
      fprintf(stderr, "Error opening files\n");
      return 1;
  }

  // GPU configuration
  int device;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int streams_used = MIN(MAX_STREAMS, prop.multiProcessorCount);
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  size_t bytes_per_batch = BATCH_PER_STREAM * cw_size * sizeof(double) * 4; // L_channel, L_APP, L_E, temp_buffer
  size_t max_batches_in_memory = (free_mem * 0.8) / bytes_per_batch;

  printf("Fixed GPU-Only Version\n");
  printf("Using %d streams with batch size %d\n", streams_used, BATCH_PER_STREAM);
  printf("GPU: %s, SM count: %d\n", prop.name, prop.multiProcessorCount);
  printf("Available memory: %.2f GB\n", free_mem / 1e9);

  // Allocate streams
  StreamData* streams = (StreamData*)calloc(streams_used, sizeof(StreamData));

  for (int s = 0; s < streams_used; s++) {
      StreamData* S = &streams[s];
      CUDA_CHECK(cudaStreamCreate(&S->stream));
      CUDA_CHECK(cudaEventCreate(&S->event));

      // Device memory allocation
      CUDA_CHECK(cudaMalloc(&S->d_input, (size_t)BATCH_PER_STREAM * codeword_block_size * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&S->d_L_channel, (size_t)BATCH_PER_STREAM * codeword_block_size * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&S->d_L_APP, (size_t)BATCH_PER_STREAM * codeword_block_size * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&S->d_L_E, (size_t)BATCH_PER_STREAM * codeword_block_size * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&S->d_temp_buffer, (size_t)BATCH_PER_STREAM * codeword_block_size * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&S->d_convergence_flags, (size_t)BATCH_PER_STREAM * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&S->d_NG_count, (size_t)BATCH_PER_STREAM * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&S->d_iter_count, (size_t)BATCH_PER_STREAM * sizeof(double)));

      // Host memory allocation
      CUDA_CHECK(cudaHostAlloc(&S->h_input_buffer,
                               (size_t)BATCH_PER_STREAM * codeword_block_size * sizeof(double),
                               cudaHostAllocDefault));
      CUDA_CHECK(cudaHostAlloc(&S->h_output_buffer,
                               (size_t)BATCH_PER_STREAM * codeword_block_size * sizeof(double),
                               cudaHostAllocDefault));
      CUDA_CHECK(cudaHostAlloc(&S->h_convergence_flags,
                               (size_t)BATCH_PER_STREAM * sizeof(int),
                               cudaHostAllocDefault));
      CUDA_CHECK(cudaHostAlloc(&S->h_NG_count,
                               (size_t)BATCH_PER_STREAM * sizeof(int),
                               cudaHostAllocDefault));
      CUDA_CHECK(cudaHostAlloc(&S->h_iter_count,
                               (size_t)BATCH_PER_STREAM * sizeof(double),
                               cudaHostAllocDefault));

      S->blocks_processed = 0;
      S->batch_id = 0;
  }

  // Statistics
  int total_blocks = 0;
  long long total_NG = 0;
  double total_iterations = 0;
  clock_t start_time = clock();

  // Processing variables
  int current_stream = 0;
  int batches_in_flight = 0;
  size_t next_batch_id = 0;
  bool done_reading = false;

  // Output buffer
  unsigned char global_byte_out = 0;
  int global_bit_count = 0;

  // ===================== FIXED PROCESSING LOOP =====================
  while (!done_reading || batches_in_flight > 0) {
      // Launch new work
      if (!done_reading && batches_in_flight < streams_used) {
          StreamData* S = &streams[current_stream];

          size_t blocks_to_read = BATCH_PER_STREAM;
          size_t blocks_read = fread(S->h_input_buffer,
                                    sizeof(double) * codeword_block_size,
                                    blocks_to_read, fin);

          if (blocks_read == 0) {
              done_reading = true;
          } else {
              S->blocks_processed = blocks_read;
              S->batch_id = next_batch_id++;

              // Initialize device arrays
              CUDA_CHECK(cudaMemsetAsync(S->d_NG_count, 0,
                                        (size_t)blocks_read * sizeof(int),
                                        S->stream));
              CUDA_CHECK(cudaMemsetAsync(S->d_convergence_flags, 0,
                                        (size_t)blocks_read * sizeof(int),
                                        S->stream));
              CUDA_CHECK(cudaMemsetAsync(S->d_iter_count, 0,
                                        (size_t)blocks_read * sizeof(double),
                                        S->stream));

              // Copy input to device
              CUDA_CHECK(cudaMemcpyAsync(S->d_L_channel, S->h_input_buffer,
                                        (size_t)blocks_read * codeword_block_size * sizeof(double),
                                        cudaMemcpyHostToDevice, S->stream));

              // FIXED: Call kernel with proper iteration tracking
              int threads_per_block = 256;
              int grid_size = blocks_read;

              unified_sogrand_iterations_kernel<<<grid_size, threads_per_block, 0, S->stream>>>(
                  S->d_L_channel,
                  S->d_L_APP,
                  S->d_L_E,
                  S->d_temp_buffer,
                  S->d_convergence_flags,
                  S->d_NG_count,
                  S->d_iter_count,  // FIXED: Pass iteration count output
                  n, k, L, Tmax, thres, even,
                  Imax, blocks_read);

              // Copy results back
              CUDA_CHECK(cudaMemcpyAsync(S->h_output_buffer, S->d_L_APP,
                                        (size_t)blocks_read * codeword_block_size * sizeof(double),
                                        cudaMemcpyDeviceToHost, S->stream));

              CUDA_CHECK(cudaMemcpyAsync(S->h_NG_count, S->d_NG_count,
                                        (size_t)blocks_read * sizeof(int),
                                        cudaMemcpyDeviceToHost, S->stream));

              // FIXED: Copy iteration counts back
              CUDA_CHECK(cudaMemcpyAsync(S->h_iter_count, S->d_iter_count,
                                        (size_t)blocks_read * sizeof(double),
                                        cudaMemcpyDeviceToHost, S->stream));

              CUDA_CHECK(cudaEventRecord(S->event, S->stream));
              batches_in_flight++;
              current_stream = (current_stream + 1) % streams_used;
          }
      }

      // Process completed batches
      for (int s = 0; s < streams_used; s++) {
          StreamData* S = &streams[s];

          if (S->blocks_processed > 0 && cudaEventQuery(S->event) == cudaSuccess) {
              double* output = S->h_output_buffer;
              for (int b = 0; b < S->blocks_processed; b++) {
                  total_NG += S->h_NG_count[b];
                  // FIXED: Use actual iteration count from GPU
                  total_iterations += S->h_iter_count[b];

                  // Extract message bits
                  for (int slice = 0; slice < k; slice++) {
                      for (int row = 0; row < k; row++) {
                          for (int col = 0; col < k; col++) {
                              int tensor_idx = b * codeword_block_size + IDX(row, col, slice, n);
                              int bit = (output[tensor_idx] > 0.0) ? 0 : 1;

                              global_byte_out = (global_byte_out << 1) | bit;
                              global_bit_count++;

                              if (global_bit_count == 8) {
                                  fwrite(&global_byte_out, 1, 1, fout);
                                  global_byte_out = 0;
                                  global_bit_count = 0;
                              }
                          }
                      }
                  }
              }

              total_blocks += S->blocks_processed;
              S->blocks_processed = 0;
              batches_in_flight--;
          }
      }

      // Progress reporting
      if (total_blocks % 100 == 0 && total_blocks > 0) {
          clock_t current_time = clock();
          double elapsed = ((double)(current_time - start_time)) / CLOCKS_PER_SEC;
          printf("Processed %d blocks in %.2f seconds (%.2f blocks/sec)\n",
                 total_blocks, elapsed, total_blocks / elapsed);
      }
  }

  // Write remaining bits
  if (global_bit_count > 0) {
      global_byte_out <<= (8 - global_bit_count);
      fwrite(&global_byte_out, 1, 1, fout);
  }

  // Wait for completion
  for (int s = 0; s < streams_used; s++) {
      CUDA_CHECK(cudaStreamSynchronize(streams[s].stream));
  }

  // Final statistics
  clock_t end_time = clock();
  double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

  printf("Decoding complete. %d block(s) decoded.\n", total_blocks);

  if (total_blocks > 0) {
      printf("Average iterations per block: %.2f\n", total_iterations / total_blocks);
      printf("Average NG per block: %.2f\n", (double)total_NG / total_blocks);
      printf("Average NG per info bit: %.2f\n", (double)total_NG / (total_blocks * k * k * k));
      printf("Total GPU time: %.2f seconds\n", cpu_time);
      printf("Throughput: %.2f blocks/sec\n", total_blocks / cpu_time);
  }

  // Cleanup
  for (int s = 0; s < streams_used; s++) {
      StreamData* S = &streams[s];

      CUDA_CHECK(cudaFree(S->d_input));
      CUDA_CHECK(cudaFree(S->d_L_channel));
      CUDA_CHECK(cudaFree(S->d_L_APP));
      CUDA_CHECK(cudaFree(S->d_L_E));
      CUDA_CHECK(cudaFree(S->d_temp_buffer));
      CUDA_CHECK(cudaFree(S->d_convergence_flags));
      CUDA_CHECK(cudaFree(S->d_NG_count));
      CUDA_CHECK(cudaFree(S->d_iter_count));

      CUDA_CHECK(cudaFreeHost(S->h_input_buffer));
      CUDA_CHECK(cudaFreeHost(S->h_output_buffer));
      CUDA_CHECK(cudaFreeHost(S->h_convergence_flags));
      CUDA_CHECK(cudaFreeHost(S->h_NG_count));
      CUDA_CHECK(cudaFreeHost(S->h_iter_count));

      CUDA_CHECK(cudaStreamDestroy(S->stream));
      CUDA_CHECK(cudaEventDestroy(S->event));
  }

  free(streams);
  free(H_flat);
  free(G_flat);
  free_int_matrix(G, k);
  free_int_matrix(H, n - k);
  fclose(fin);
  fclose(fout);

  return 0;
}
