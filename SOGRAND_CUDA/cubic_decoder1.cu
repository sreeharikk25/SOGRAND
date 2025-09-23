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

// ===================== Simple Optimized Configuration =====================
#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)
#define MAX_BLOCKS_PER_KERNEL 65535
#define MAX_CW_SIZE 4096

// ====================================================

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static const double EPSILON = 1e-40;

namespace cg = cooperative_groups;

// Constant memory for H and G matrices
__constant__ uint8_t d_H_const[8*16];
__constant__ int d_G_const[8*16];
__constant__ double d_alpha_const[100];

// Keep original indexing - don't change memory patterns
__host__ __device__ __forceinline__ int IDX_FAST(int row, int col, int slice) {
  return (slice << 8) + (row << 4) + col;  // slice*256 + row*16 + col for n=16
}

#define CUDA_CHECK(call) \
do { \
  cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1);} \
} while(0)

// FIX 1: Warp-level reductions to reduce atomic contention
__device__ __forceinline__ int warp_reduce_sum(int val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Simple vectorized memory operations (keep existing approach)
__device__ __forceinline__ double4 load_double4_fast(const double* addr) {
  return *reinterpret_cast<const double4*>(addr);
}

__device__ __forceinline__ double2 load_double2_fast(const double* addr) {
  return *reinterpret_cast<const double2*>(addr);
}

__device__ __forceinline__ void store_double4_fast(double* addr, double4 val) {
  *reinterpret_cast<double4*>(addr) = val;
}

__device__ __forceinline__ void store_double2_fast(double* addr, double2 val) {
  *reinterpret_cast<double2*>(addr) = val;
}

// Vector helper functions (unchanged)
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

// ===================== Device Helper Functions (SOGRAND Logic UNCHANGED) =====================
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
    double4 llr_vec = load_double4_fast(&llr[i]);
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

// ===================== SOGRAND Device Implementation (Logic UNCHANGED) =====================
__device__ void sogrand_main_logic_device(double* chat_list, double* s_list, double* T_val,
                                          double* curL_val, double* pNL_val, double* APP_list,
                                          double* llr, const uint8_t* H_flat, int n, int s,
                                          int IC, uint64_t L, uint64_t Tmax, double thres, int even,
                                          uint8_t* workspace) {
  // Use workspace to reduce register pressure
  size_t* perm = (size_t*)workspace;
  uint8_t* cHD = workspace + 16 * sizeof(size_t);
  uint8_t* TEP = cHD + 16;
  uint8_t* c = TEP + 16;
  double* absL = (double*)(c + 16);
  int32_t* u = (int32_t*)(absL + 16);
  int32_t* d = u + 16;
  int32_t* D = d + 16;

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

  // Full enumeration loop - exactly following C code (UNCHANGED)
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
                                              unsigned long Tmax, double thres, int even,
                                              uint8_t* workspace) {
  double chat_list[16*3];
  double s_list[4*3];
  double T_val, curL_val, pNL_val;
  double APP_list[3];

  sogrand_main_logic_device(chat_list, s_list, &T_val, &curL_val, &pNL_val,
                           APP_list, llr, H_flat, n, n-k, -1, L, Tmax, thres, even,
                           workspace);

  int curL = (int)curL_val;
  if (curL == 0) {
    // Vectorized initialization
    int i = 0;
    for (; i + 3 < n; i += 4) {
      double4 llr_vec = load_double4_fast(&llr[i]);
      double4 zero_vec = make_double4_safe(0.0, 0.0, 0.0, 0.0);
      store_double4_fast(&L_APP[i], llr_vec);
      store_double4_fast(&L_E[i], zero_vec);
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

    // Vectorized probability calculations
    int i = 0;
    for (; i + 1 < n; i += 2) {
      double2 llr_vec = load_double2_fast(&llr[i]);
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

    // Vectorized final calculations
    i = 0;
    for (; i + 1 < n; i += 2) {
      double2 p0_vec = make_double2_safe(p0[i], p0[i+1]);
      double2 p1_vec = make_double2_safe(p1[i], p1[i+1]);
      double2 pp0_vec = make_double2_safe(pp0[i], pp0[i+1]);
      double2 pp1_vec = make_double2_safe(pp1[i], pp1[i+1]);
      double2 llr_vec = load_double2_fast(&llr[i]);

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

      store_double2_fast(&L_APP[i], L_APP_vec);
      store_double2_fast(&L_E[i], L_E_vec);
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

// ===================== Simple Convergence Check (keep original approach) =====================
__device__ int check_convergence_optimized(const double* __restrict__ L_APP,
                                          int n, int k, int batch_idx, int cw_size,
                                          uint8_t* shared_c_HD,
                                          uint8_t* shared_c_test,
                                          int* shared_workspace) {
  if (cw_size > MAX_CW_SIZE) {
    return 0;
  }

  const double* batch_L_APP = L_APP + (size_t)batch_idx * cw_size;
  const int tid = threadIdx.x;
  const int total_threads = blockDim.x;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;
  const int warps_per_block = (total_threads + WARP_SIZE - 1) / WARP_SIZE;

  for (int idx = tid; idx < cw_size; idx += total_threads) {
    shared_c_HD[idx] = (batch_L_APP[idx] > 0.0) ? 0 : 1;
    shared_c_test[idx] = 0;
  }
  __syncthreads();

  const int total_msg_bits = k * k * k;
  for (int linear = tid; linear < total_msg_bits; linear += total_threads) {
    int slice = linear / (k * k);
    int rem = linear % (k * k);
    int row = rem / k;
    int col = rem % k;
    shared_c_test[IDX_FAST(row, col, slice)] = shared_c_HD[IDX_FAST(row, col, slice)];
  }
  __syncthreads();

  const int parity_span = n - k;
  if (parity_span > 0) {
    const int row_tasks = k * k;
    for (int task = warp_id; task < row_tasks; task += warps_per_block) {
      int slice = task / k;
      int row = task % k;
      for (int col = k + lane_id; col < n; col += WARP_SIZE) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; ++msg_bit_idx) {
          parity_val ^= (shared_c_test[IDX_FAST(row, msg_bit_idx, slice)] &
                         d_G_const[msg_bit_idx * n + col]);
        }
        if (col < n) {
          shared_c_test[IDX_FAST(row, col, slice)] = parity_val & 1;
        }
      }
    }
    __syncthreads();

    const int col_tasks = k * n;
    for (int task = warp_id; task < col_tasks; task += warps_per_block) {
      int slice = task / n;
      int col = task % n;
      for (int row = k + lane_id; row < n; row += WARP_SIZE) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; ++msg_bit_idx) {
          parity_val ^= (shared_c_test[IDX_FAST(msg_bit_idx, col, slice)] &
                         d_G_const[msg_bit_idx * n + row]);
        }
        shared_c_test[IDX_FAST(row, col, slice)] = parity_val & 1;
      }
    }
    __syncthreads();

    const int slice_tasks = n * n;
    for (int task = warp_id; task < slice_tasks; task += warps_per_block) {
      int row = task / n;
      int col = task % n;
      for (int slice = k + lane_id; slice < n; slice += WARP_SIZE) {
        int parity_val = 0;
        for (int msg_bit_idx = 0; msg_bit_idx < k; ++msg_bit_idx) {
          parity_val ^= (shared_c_test[IDX_FAST(row, col, msg_bit_idx)] &
                         d_G_const[msg_bit_idx * n + slice]);
        }
        shared_c_test[IDX_FAST(row, col, slice)] = parity_val & 1;
      }
    }
    __syncthreads();
  }

  int mismatch = 0;
  for (int idx = tid; idx < cw_size; idx += total_threads) {
    mismatch |= (shared_c_test[idx] != shared_c_HD[idx]);
  }

  unsigned int active_mask = __activemask();
  unsigned int ballot = __ballot_sync(active_mask, mismatch != 0);
  if (lane_id == 0) {
    shared_workspace[warp_id] = (ballot != 0);
  }
  __syncthreads();

  if (warp_id == 0) {
    int lane_val = (lane_id < warps_per_block) ? shared_workspace[lane_id] : 0;
    int warp_sum = warp_reduce_sum(lane_val);
    if (lane_id == 0) {
      shared_workspace[0] = warp_sum;
    }
  }
  __syncthreads();

  return (shared_workspace[0] == 0) ? 1 : 0;
}

// ===================== SIMPLE OPTIMIZED GPU KERNEL =====================
__global__ void optimized_sogrand_kernel(
    const double* __restrict__ L_channel,
    double* __restrict__ L_APP,
    double* __restrict__ L_E,
    int* __restrict__ convergence_flags,
    int* __restrict__ NG_count,
    double* __restrict__ iter_count_output,
    int n, int k, int L, uint64_t Tmax, double thres, int even,
    int max_iterations, int total_blocks) {

    const int batch_id = blockIdx.x;
    if (batch_id >= total_blocks) return;

    const int work_id = threadIdx.x;

    const int cw_size = n * n * n;
    const size_t base = (size_t)batch_id * cw_size;

    // FIX 2: Reduced synchronization - simple shared memory
    __shared__ int iteration_count;
    __shared__ int batch_converged;
    __shared__ int warp_NG_totals[WARPS_PER_BLOCK];
    __shared__ double phase_count;
    __shared__ uint8_t shared_c_HD[MAX_CW_SIZE];
    __shared__ uint8_t shared_c_test[MAX_CW_SIZE];
    __shared__ int convergence_workspace[WARPS_PER_BLOCK + 1];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (threadIdx.x == 0) {
        iteration_count = 1;
        batch_converged = 0;
        phase_count = 0.0;
    }
    if (threadIdx.x < WARPS_PER_BLOCK) {
        warp_NG_totals[threadIdx.x] = 0;
    }
    __syncthreads();

    // Initialize arrays
    if (work_id < n*n) {
        for (int slice = 0; slice < n; slice++) {
            int idx = IDX_FAST(work_id/n, work_id%n, slice);
            L_APP[base + idx] = L_channel[base + idx];
            L_E[base + idx] = 0.0;
        }
    }
    __syncthreads();

    // Main iteration loop with simple optimization
    while (iteration_count <= max_iterations && !batch_converged) {

        int local_NG_count = 0;

        // ===== COLUMNS PHASE =====
        if (threadIdx.x == 0) {
            phase_count += 0.5;
        }

        if (work_id < n*n) {
            int slice = work_id / n;
            int col = work_id % n;

            double vec_in[16];
            double vec_L_APP[16];
            double vec_L_E[16];
            uint8_t local_workspace[512];

            for (int row = 0; row < n; row++) {
                int idx = IDX_FAST(row, col, slice);
                vec_in[row] = L_channel[base + idx] +
                             d_alpha_const[2*iteration_count-2] * L_E[base + idx];
            }

            int N_guess = SOGRAND_bitSO_device_vectorized(vec_L_APP, vec_L_E, vec_in,
                                                         d_H_const, n, k, L, Tmax, thres, even,
                                                         local_workspace);

            for (int row = 0; row < n; row++) {
                int idx = IDX_FAST(row, col, slice);
                L_APP[base + idx] = vec_L_APP[row];
                L_E[base + idx] = vec_L_E[row];
            }

            local_NG_count += N_guess;
        }

        // FIX 1: Warp-level NG reduction instead of atomics
        int warp_NG = warp_reduce_sum(local_NG_count);
        if (lane_id == 0) {
            atomicAdd(&warp_NG_totals[warp_id], warp_NG);
        }

        __syncthreads();

        // Check convergence cooperatively
        int converged_flag = check_convergence_optimized(
            L_APP, n, k, batch_id, cw_size,
            shared_c_HD, shared_c_test, convergence_workspace);
        if (threadIdx.x == 0 && converged_flag) {
            batch_converged = 1;
        }
        __syncthreads();

        if (batch_converged) break;

        // ===== ROWS PHASE =====
        if (threadIdx.x == 0) {
            phase_count += 0.5;
        }

        if (work_id < n*n) {
            int slice = work_id / n;
            int row = work_id % n;

            double vec_in[16];
            double vec_L_APP[16];
            double vec_L_E[16];
            uint8_t local_workspace[512];

            for (int col = 0; col < n; col++) {
                int idx = IDX_FAST(row, col, slice);
                vec_in[col] = L_channel[base + idx] +
                             d_alpha_const[2*iteration_count-1] * L_E[base + idx];
            }

            int N_guess = SOGRAND_bitSO_device_vectorized(vec_L_APP, vec_L_E, vec_in,
                                                         d_H_const, n, k, L, Tmax, thres, even,
                                                         local_workspace);

            for (int col = 0; col < n; col++) {
                int idx = IDX_FAST(row, col, slice);
                L_APP[base + idx] = vec_L_APP[col];
                L_E[base + idx] = vec_L_E[col];
            }

            // FIX 1: Accumulate locally, reduce per warp
            local_NG_count += N_guess;
        }

        warp_NG = warp_reduce_sum(local_NG_count);
        if (lane_id == 0) {
            atomicAdd(&warp_NG_totals[warp_id], warp_NG);
        }

        __syncthreads();

        converged_flag = check_convergence_optimized(
            L_APP, n, k, batch_id, cw_size,
            shared_c_HD, shared_c_test, convergence_workspace);
        if (threadIdx.x == 0 && converged_flag) {
            batch_converged = 1;
        }
        __syncthreads();

        if (batch_converged) break;

        // ===== SLICES PHASE =====
        if (threadIdx.x == 0) {
            phase_count += 0.5;
        }

        if (work_id < n*n) {
            int row = work_id / n;
            int col = work_id % n;

            double vec_in[16];
            double vec_L_APP[16];
            double vec_L_E[16];
            uint8_t local_workspace[512];

            for (int slice = 0; slice < n; slice++) {
                int idx = IDX_FAST(row, col, slice);
                vec_in[slice] = L_channel[base + idx] +
                               d_alpha_const[2*iteration_count-1] * L_E[base + idx];
            }

            int N_guess = SOGRAND_bitSO_device_vectorized(vec_L_APP, vec_L_E, vec_in,
                                                         d_H_const, n, k, L, Tmax, thres, even,
                                                         local_workspace);

            for (int slice = 0; slice < n; slice++) {
                int idx = IDX_FAST(row, col, slice);
                L_APP[base + idx] = vec_L_APP[slice];
                L_E[base + idx] = vec_L_E[slice];
            }

            local_NG_count += N_guess;
        }

        warp_NG = warp_reduce_sum(local_NG_count);
        if (lane_id == 0) {
            atomicAdd(&warp_NG_totals[warp_id], warp_NG);
        }

        __syncthreads();

        // Final convergence check and iteration increment
        converged_flag = check_convergence_optimized(
            L_APP, n, k, batch_id, cw_size,
            shared_c_HD, shared_c_test, convergence_workspace);
        if (threadIdx.x == 0) {
            if (converged_flag) {
                batch_converged = 1;
            } else {
                iteration_count++;
            }
        }
        __syncthreads();

        if (batch_converged) break;
    }

    // FIX 1: Final reduction with single atomic per block
    if (threadIdx.x == 0) {
        int total_NG = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            total_NG += warp_NG_totals[w];
        }

        NG_count[batch_id] = total_NG;
        convergence_flags[batch_id] = batch_converged;
        iter_count_output[batch_id] = phase_count;
    }
}

// ===================== Helper Functions Implementation (Unchanged) =====================
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

// ===================== Main Function (Unchanged Structure) =====================
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
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024));

  // Copy matrices to constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(d_H_const, H_flat, (n-k) * n * sizeof(uint8_t)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_G_const, G_flat, k * n * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_alpha_const, alpha, 100 * sizeof(double)));

  // ===================== SINGLE-SHOT APPROACH =====================

  // Open input file and get size
  FILE* fin = fopen(input_filename, "rb");
  if (!fin) {
      fprintf(stderr, "Error opening input file\n");
      return 1;
  }

  // Get file size and calculate number of blocks
  fseek(fin, 0, SEEK_END);
  long file_size = ftell(fin);
  rewind(fin);

  int total_blocks = file_size / (cw_size * sizeof(double));
  printf("Optimized Single-Shot GPU Version\n");
  printf("Input file size: %.2f KB\n", file_size / 1024.0);
  printf("Total blocks to process: %d\n", total_blocks);

  if (total_blocks <= 0) {
      fprintf(stderr, "Error: No valid blocks found in input file\n");
      fclose(fin);
      return 1;
  }

  // Calculate memory requirements
  size_t input_size = total_blocks * cw_size * sizeof(double);
  size_t output_size = total_blocks * cw_size * sizeof(double) * 2; // L_APP + L_E
  size_t stats_size = total_blocks * (sizeof(int) * 2 + sizeof(double)); // flags + NG + iter
  size_t total_gpu_memory = input_size + output_size + stats_size;

  printf("GPU memory required: %.2f MB\n", total_gpu_memory / (1024.0 * 1024.0));

  // Check GPU memory
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  printf("Available GPU memory: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));

  if (total_gpu_memory > free_mem * 0.8) {
      fprintf(stderr, "Error: Not enough GPU memory (need %.2f MB, have %.2f MB)\n",
              total_gpu_memory / (1024.0 * 1024.0),
              free_mem * 0.8 / (1024.0 * 1024.0));
      fclose(fin);
      return 1;
  }

  // Allocate host memory
  double* h_input = (double*)malloc(input_size);
  double* h_output = (double*)malloc(total_blocks * cw_size * sizeof(double));
  int* h_NG_count = (int*)malloc(total_blocks * sizeof(int));
  double* h_iter_count = (double*)malloc(total_blocks * sizeof(double));

  if (!h_input || !h_output || !h_NG_count || !h_iter_count) {
      fprintf(stderr, "Error: Host memory allocation failed\n");
      fclose(fin);
      return 1;
  }

  // Read entire file into host memory
  printf("Reading entire file into memory...\n");
  size_t bytes_read = fread(h_input, 1, input_size, fin);
  fclose(fin);

  if (bytes_read != input_size) {
      fprintf(stderr, "Error: Failed to read entire file (read %zu, expected %zu)\n",
              bytes_read, input_size);
      free(h_input);
      free(h_output);
      free(h_NG_count);
      free(h_iter_count);
      return 1;
  }

  // Allocate GPU memory
  printf("Allocating GPU memory...\n");
  double* d_L_channel;
  double* d_L_APP;
  double* d_L_E;
  int* d_convergence_flags;
  int* d_NG_count;
  double* d_iter_count;

  CUDA_CHECK(cudaMalloc(&d_L_channel, input_size));
  CUDA_CHECK(cudaMalloc(&d_L_APP, total_blocks * cw_size * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_L_E, total_blocks * cw_size * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_convergence_flags, total_blocks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_NG_count, total_blocks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_iter_count, total_blocks * sizeof(double)));

  // Copy data to GPU
  printf("Copying data to GPU...\n");
  CUDA_CHECK(cudaMemcpy(d_L_channel, h_input, input_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_convergence_flags, 0, total_blocks * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_NG_count, 0, total_blocks * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_iter_count, 0, total_blocks * sizeof(double)));

  // Launch optimized kernel
  printf("Launching optimized SOGRAND decoder kernel...\n");
  clock_t start_time = clock();

  // Calculate grid and block dimensions
  int threads_per_block = THREADS_PER_BLOCK;
  int blocks_per_grid = MIN(total_blocks, MAX_BLOCKS_PER_KERNEL);

  if (total_blocks <= MAX_BLOCKS_PER_KERNEL) {
      // Single kernel launch for all blocks
      optimized_sogrand_kernel<<<blocks_per_grid, threads_per_block>>>(
          d_L_channel, d_L_APP, d_L_E, d_convergence_flags, d_NG_count, d_iter_count,
          n, k, L, Tmax, thres, even, Imax, total_blocks);
  } else {
      // Multiple kernel launches if too many blocks
      int blocks_processed = 0;
      while (blocks_processed < total_blocks) {
          int blocks_this_launch = MIN(MAX_BLOCKS_PER_KERNEL, total_blocks - blocks_processed);

          optimized_sogrand_kernel<<<blocks_this_launch, threads_per_block>>>(
              d_L_channel + blocks_processed * cw_size,
              d_L_APP + blocks_processed * cw_size,
              d_L_E + blocks_processed * cw_size,
              d_convergence_flags + blocks_processed,
              d_NG_count + blocks_processed,
              d_iter_count + blocks_processed,
              n, k, L, Tmax, thres, even, Imax, blocks_this_launch);

          blocks_processed += blocks_this_launch;
      }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  clock_t end_time = clock();

  // Copy results back
  printf("Copying results back from GPU...\n");
  CUDA_CHECK(cudaMemcpy(h_output, d_L_APP, total_blocks * cw_size * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_NG_count, d_NG_count, total_blocks * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_iter_count, d_iter_count, total_blocks * sizeof(double), cudaMemcpyDeviceToHost));

  // Write output
  printf("Writing output file...\n");
  FILE* fout = fopen(output_filename, "wb");
  if (!fout) {
      fprintf(stderr, "Error opening output file\n");
      return 1;
  }

  unsigned char global_byte_out = 0;
  int global_bit_count = 0;

  for (int b = 0; b < total_blocks; b++) {
      for (int slice = 0; slice < k; slice++) {
          for (int row = 0; row < k; row++) {
              for (int col = 0; col < k; col++) {
                  int tensor_idx = b * cw_size + IDX_FAST(row, col, slice);
                  int bit = (h_output[tensor_idx] > 0.0) ? 0 : 1;

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

  if (global_bit_count > 0) {
      global_byte_out <<= (8 - global_bit_count);
      fwrite(&global_byte_out, 1, 1, fout);
  }

  fclose(fout);

  // Calculate statistics
  long long total_NG = 0;
  double total_iterations = 0;

  for (int i = 0; i < total_blocks; i++) {
      total_NG += h_NG_count[i];
      total_iterations += h_iter_count[i];
  }

  double gpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

  printf("Decoding complete. %d block(s) decoded.\n", total_blocks);
  printf("Average iterations per block: %.2f\n", total_iterations / total_blocks);
  printf("Average NG per block: %.2f\n", (double)total_NG / total_blocks);
  printf("Average NG per info bit: %.2f\n", (double)total_NG / (total_blocks * k * k * k));
  printf("Total GPU time: %.2f seconds\n", gpu_time);
  printf("Throughput: %.2f blocks/sec\n", total_blocks / gpu_time);

  // Cleanup
  CUDA_CHECK(cudaFree(d_L_channel));
  CUDA_CHECK(cudaFree(d_L_APP));
  CUDA_CHECK(cudaFree(d_L_E));
  CUDA_CHECK(cudaFree(d_convergence_flags));
  CUDA_CHECK(cudaFree(d_NG_count));
  CUDA_CHECK(cudaFree(d_iter_count));

  free(h_input);
  free(h_output);
  free(h_NG_count);
  free(h_iter_count);
  free(H_flat);
  free(G_flat);
  free_int_matrix(G, k);
  free_int_matrix(H, n - k);

  return 0;
}
