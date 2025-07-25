#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128
#define MAX_LIST_SIZE 3
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define Inf 0x7fffffff

// Constants for n=15, k=10
__constant__ int d_G[10*15];  // Generator matrix
__constant__ int d_H[5*15];   // Parity check matrix

// 3D tensor access helpers
__host__ __device__ inline int tensor_idx(int i, int j, int k, int n) {
    return k * n * n + j * n + i;
}

// Helper functions that work on both host and device
__host__ __device__ inline double fmax_safe(double a, double b) {
    return (a > b) ? a : b;
}

__host__ __device__ inline double fmin_safe(double a, double b) {
    return (a < b) ? a : b;
}

// SOGRAND state for n=15
struct SOGRANDState15 {
    double llr[15];
    double absL[15];
    int perm[15];
    uint8_t cHD[15];
    uint8_t c[15];
    uint8_t TEP[15];
    double chat_list[15 * MAX_LIST_SIZE];
    double PM_list[MAX_LIST_SIZE];
    int curL;
    int T;
    double pNL;
};

// Device functions
__device__ double JacLog_cuda(double x) {
    if(x > 30) return x;
    else if(x < -30) return 0.0;
    else return log(1.0 + exp(x));
}

__device__ double getPM_HD_cuda(double *absL, int n) {
    double pm = 0;
    for(int i = 0; i < n; i++){
        pm += JacLog_cuda(-absL[i]);
    }
    return pm;
}

__device__ double getPM_cuda(uint8_t *TEP, double *absL, double PM_HD, int n) {
    double pm = PM_HD;
    for(int i = 0; i < n; i++){
        if (TEP[i] == 1)
            pm += (JacLog_cuda(absL[i]) - JacLog_cuda(-absL[i]));
    }
    return pm;
}

__device__ bool parity_check_cuda(uint8_t* c, int n, int s) {
    for (int j = 0; j < s; j++) {
        uint8_t syndrome = 0;
        for (int i = 0; i < n; i++) {
            syndrome ^= (c[i] * d_H[j*n + i]);
        }
        if (syndrome == 1) return false;
    }
    return true;
}

// Improved SOGRAND SISO decoder
__device__ void sogrand_siso_cuda_fixed(double* L_APP, double* L_E, double* llr, 
                                        int n, int k, SOGRANDState15* state) {
    // Hard decision
    for (int i = 0; i < n; i++) {
        state->cHD[i] = (llr[i] > 0.0) ? 0 : 1;
        state->absL[i] = fabs(llr[i]);
        state->perm[i] = i;
    }
    
    // Sort by reliability (bubble sort - OK for small n)
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (state->absL[j] > state->absL[j+1]) {
                double temp = state->absL[j];
                state->absL[j] = state->absL[j+1];
                state->absL[j+1] = temp;
                int temp_perm = state->perm[j];
                state->perm[j] = state->perm[j+1];
                state->perm[j+1] = temp_perm;
            }
        }
    }
    
    // Initialize
    for (int i = 0; i < n; i++) {
        state->c[i] = state->cHD[i];
        state->TEP[i] = 0;
    }
    state->curL = 0;
    state->T = 1;
    state->pNL = 1.0;
    
    double PM_HD = getPM_HD_cuda(state->absL, n);
    
    // Check hard decision
    if (parity_check_cuda(state->c, n, n-k)) {
        for (int i = 0; i < n; i++) {
            state->chat_list[i] = state->c[i];
        }
        state->PM_list[0] = getPM_cuda(state->TEP, state->absL, PM_HD, n);
        state->curL = 1;
        state->pNL -= exp(-state->PM_list[0]);
    }
    
    // ORBGRAND search with limited complexity
    int IC = 0; // Simplified - no intercept
    int wt_max = min(60, IC*n + n*(n+1)/2); // Limit maximum weight
    
    for (int wt = IC + 1; wt <= wt_max && state->curL < MAX_LIST_SIZE && state->T < 1000; wt++) {
        // For each Hamming weight w
        int w_start = 1;
        int w_end = min(n, (int)((1 + 2*(n + IC) - sqrt(fmax_safe(0, pow(1 + 2*(n + IC), 2) - 8*wt))) / 2));
        
        for (int w = w_start; w <= w_end && state->curL < MAX_LIST_SIZE; w++) {
            int W = wt - IC*w;
            if (W < w*(w+1)/2) continue;
            
            // Simple case: flip first w least reliable bits
            for (int i = 0; i < n; i++) state->TEP[i] = 0;
            for (int i = 0; i < w; i++) state->TEP[i] = 1;
            
            // Apply TEP
            for (int i = 0; i < n; i++) {
                state->c[state->perm[i]] = state->cHD[state->perm[i]] ^ state->TEP[i];
            }
            
            state->T++;
            double pm = getPM_cuda(state->TEP, state->absL, PM_HD, n);
            state->pNL -= exp(-pm);
            
            if (parity_check_cuda(state->c, n, n-k)) {
                for (int i = 0; i < n; i++) {
                    state->chat_list[state->curL * n + i] = state->c[i];
                }
                state->PM_list[state->curL] = pm;
                state->curL++;
            }
        }
    }
    
    // Compute soft output
    if (state->curL == 0) {
        for (int i = 0; i < n; i++) {
            L_APP[i] = llr[i];
            L_E[i] = 0;
        }
    } else {
        double pp0[15], pp1[15];
        for (int i = 0; i < n; i++) {
            pp1[i] = 1.0 / (1.0 + exp(llr[i]));
            pp0[i] = 1.0 - pp1[i];
            pp1[i] = fmax_safe(pp1[i], 1e-9);
            pp1[i] = fmin_safe(pp1[i], 1.0 - 1e-9);
            pp0[i] = fmax_safe(pp0[i], 1e-9);
            pp0[i] = fmin_safe(pp0[i], 1.0 - 1e-9);
        }
        
        // Compute probabilities using path metrics
        double p0[15] = {0}, p1[15] = {0};
        for (int l = 0; l < state->curL; l++) {
            double p = exp(-state->PM_list[l]);
            for (int i = 0; i < n; i++) {
                if (state->chat_list[l * n + i] == 1) {
                    p1[i] += p;
                } else {
                    p0[i] += p;
                }
            }
        }
        
        // Add probability of not being in list
        double p_notinlist = fmax_safe(state->pNL, 1e-9);
        for (int i = 0; i < n; i++) {
            p0[i] += p_notinlist * pp0[i];
            p1[i] += p_notinlist * pp1[i];
        }
        
        // Compute LLRs
        for (int i = 0; i < n; i++) {
            L_APP[i] = log(fmax_safe(p0[i], 1e-30)) - log(fmax_safe(p1[i], 1e-30));
            L_E[i] = L_APP[i] - llr[i];
        }
    }
}

// Kernels for cubic decoding
__global__ void decode_columns_cubic_kernel_fixed(double* L_channel, double* L_APP, double* L_E,
                                                  double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || slice >= n || col >= n) return;
    
    __shared__ SOGRANDState15 states[4];
    SOGRANDState15* state = &states[threadIdx.x % 4];
    
    int offset = block_id * n * n * n;
    
    // Prepare input
    double input[15];
    for (int row = 0; row < n; row++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[row] = L_channel[idx] + alpha * L_E[idx];
    }
    
    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_fixed(L_APP_vec, L_E_vec, input, n, k, state);
    
    // Write results
    for (int row = 0; row < n; row++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[row];
        L_E[idx] = L_E_vec[row];
    }
}

__global__ void decode_rows_cubic_kernel_fixed(double* L_channel, double* L_APP, double* L_E,
                                               double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || slice >= n || row >= n) return;
    
    __shared__ SOGRANDState15 states[4];
    SOGRANDState15* state = &states[threadIdx.x % 4];
    
    int offset = block_id * n * n * n;
    
    // Prepare input
    double input[15];
    for (int col = 0; col < n; col++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[col] = L_channel[idx] + alpha * L_E[idx];
    }
    
    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_fixed(L_APP_vec, L_E_vec, input, n, k, state);
    
    // Write results
    for (int col = 0; col < n; col++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[col];
        L_E[idx] = L_E_vec[col];
    }
}

__global__ void decode_slices_cubic_kernel_fixed(double* L_channel, double* L_APP, double* L_E,
                                                double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || row >= n || col >= n) return;
    
    __shared__ SOGRANDState15 states[4];
    SOGRANDState15* state = &states[threadIdx.x % 4];
    
    int offset = block_id * n * n * n;
    
    // Prepare input
    double input[15];
    for (int slice = 0; slice < n; slice++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[slice] = L_channel[idx] + alpha * L_E[idx];
    }
    
    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_fixed(L_APP_vec, L_E_vec, input, n, k, state);
    
    // Write results
    for (int slice = 0; slice < n; slice++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[slice];
        L_E[idx] = L_E_vec[slice];
    }
}

// Host-side CRC matrix generation
int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));
    for (int i = 0; i < len; i++) poly[i] = (dec_val >> (len - 1 - i)) & 1;
    poly[len] = 1;
    return poly;
}

void getGH_sys_CRC(int n, int k, int* G_flat, int* H_flat) {
    const char* hex_poly = NULL;
    int r = n - k;

    // Select polynomial based on code parameters
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

    // Temporary matrices
    int** P = (int**)malloc(k * sizeof(int*));
    for(int i=0; i<k; ++i) P[i] = (int*)calloc(r, sizeof(int));
    int* msg_poly = (int*)calloc(k + r, sizeof(int));

    // Generate parity matrix P
    for (int i = 0; i < k; i++) {
        memset(msg_poly, 0, (k + r) * sizeof(int));
        msg_poly[i] = 1;

        // Polynomial division
        for (int j = 0; j < k; j++) {
            if (msg_poly[j] == 1) {
                for (int l = 0; l < poly_len; l++) {
                    msg_poly[j + l] ^= poly[l];
                }
            }
        }
        
        // Extract remainder (parity bits)
        for (int j = 0; j < r; j++) P[i][j] = msg_poly[k + j];
    }

    // Build G matrix: G = [I_k | P]
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            G_flat[i * n + j] = (i == j) ? 1 : 0;
        }
        for (int j = 0; j < r; j++) {
            G_flat[i * n + (k + j)] = P[i][j];
        }
    }

    // Build H matrix: H = [P^T | I_r]
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < k; j++) {
            H_flat[i * n + j] = P[j][i];
        }
        for (int j = 0; j < r; j++) {
            H_flat[i * n + (k + j)] = (i == j) ? 1 : 0;
        }
    }

    // Cleanup
    free(poly);
    free(msg_poly);
    for(int i=0; i<k; ++i) free(P[i]);
    free(P);
}

// Host function for cubic decoding
void decode_cubic_cuda_fixed(double* h_llr_buffer, int* h_bit_buffer, int num_blocks,
                            int n, int k, int Imax, double* alpha) {
    size_t tensor_size = n * n * n * sizeof(double);
    
    // Allocate device memory
    double *d_L_channel, *d_L_APP, *d_L_E;
    
    CHECK_CUDA(cudaMalloc(&d_L_channel, num_blocks * tensor_size));
    CHECK_CUDA(cudaMalloc(&d_L_APP, num_blocks * tensor_size));
    CHECK_CUDA(cudaMalloc(&d_L_E, num_blocks * tensor_size));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_L_channel, h_llr_buffer, num_blocks * tensor_size, 
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_L_E, 0, num_blocks * tensor_size));
    
    // Setup grid dimensions
    dim3 threadsPerBlock(16);
    dim3 blocksColumns((n + threadsPerBlock.x - 1) / threadsPerBlock.x, n, num_blocks);
    dim3 blocksRows((n + threadsPerBlock.x - 1) / threadsPerBlock.x, n, num_blocks);
    dim3 blocksSlices((n + threadsPerBlock.x - 1) / threadsPerBlock.x, n, num_blocks);
    
    // Iterative decoding
    for (int iter = 1; iter <= Imax; iter++) {
        // Stage 1: Decode columns
        decode_columns_cubic_kernel_fixed<<<blocksColumns, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter-2], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Stage 2: Decode rows  
        decode_rows_cubic_kernel_fixed<<<blocksRows, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter-1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Stage 3: Decode slices (use same alpha as rows)
        decode_slices_cubic_kernel_fixed<<<blocksSlices, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter-1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // TODO: Add early termination check
    }
    
    // Extract message bits
    double* h_L_APP = (double*)malloc(num_blocks * tensor_size);
    CHECK_CUDA(cudaMemcpy(h_L_APP, d_L_APP, num_blocks * tensor_size, 
                          cudaMemcpyDeviceToHost));
    
    // Hard decision to get bits from [0:k, 0:k, 0:k]
    for (int b = 0; b < num_blocks; b++) {
        int bit_idx = 0;
        for (int slice = 0; slice < k; slice++) {
            for (int row = 0; row < k; row++) {
                for (int col = 0; col < k; col++) {
                    int llr_idx = b * n * n * n + tensor_idx(row, col, slice, n);
                    h_bit_buffer[b * k * k * k + bit_idx] = (h_L_APP[llr_idx] > 0) ? 0 : 1;
                    bit_idx++;
                }
            }
        }
    }
    
    free(h_L_APP);
    CHECK_CUDA(cudaFree(d_L_channel));
    CHECK_CUDA(cudaFree(d_L_APP));
    CHECK_CUDA(cudaFree(d_L_E));
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
    const int Imax = 30;
    
    // Initialize alpha array
    double alpha[100];
    for(int i = 0; i < 100; i++) alpha[i] = 0.7;
    
    // Generate CRC matrices
    int h_G[10*15], h_H[5*15];
    getGH_sys_CRC(n, k, h_G, h_H);
    
    // Copy to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_G, h_G, sizeof(h_G)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_H, h_H, sizeof(h_H)));

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("CUDA Cubic Decoder Fixed (n=%d, k=%d)...\n", n, k);

    // Process in batches
    const int BATCH_SIZE = 16;
    double* llr_batch = (double*)malloc(BATCH_SIZE * codeword_block_size * sizeof(double));
    int* bit_batch = (int*)malloc(BATCH_SIZE * message_block_size * sizeof(int));
    
    int total_blocks = 0;
    
    while (true) {
        // Read batch
        int blocks_read = 0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            size_t read = fread(&llr_batch[i * codeword_block_size], 
                               sizeof(double), codeword_block_size, fin);
            if (read != codeword_block_size) break;
            blocks_read++;
        }
        
        if (blocks_read == 0) break;
        
        // Decode batch on GPU
        decode_cubic_cuda_fixed(llr_batch, bit_batch, blocks_read, n, k, Imax, alpha);
        
        // Convert bits to bytes and write
        for (int b = 0; b < blocks_read; b++) {
            unsigned char byte_out = 0;
            int bit_count_out = 0;
            
            for (int i = 0; i < message_block_size; i++) {
                byte_out = (byte_out << 1) | bit_batch[b * message_block_size + i];
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
        
        total_blocks += blocks_read;
    }

    printf("Decoding complete. %d block(s) decoded.\n", total_blocks);

    free(llr_batch);
    free(bit_batch);
    fclose(fin);
    fclose(fout);

    return 0;
}