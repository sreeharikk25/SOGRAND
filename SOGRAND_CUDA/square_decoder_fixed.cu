#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define MAX_LIST_SIZE 4
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define Inf 0x7fffffff

// Constants
__constant__ int d_G[25*31];  // Generator matrix
__constant__ int d_H[6*31];   // Parity check matrix

// Helper functions
__host__ __device__ inline double fmax_safe(double a, double b) {
    return (a > b) ? a : b;
}

__host__ __device__ inline double fmin_safe(double a, double b) {
    return (a < b) ? a : b;
}

// Structure for SOGRAND state
struct SOGRANDState {
    double llr[31];
    double absL[31];
    int perm[31];
    uint8_t cHD[31];
    uint8_t c[31];
    uint8_t TEP[31];
    double chat_list[31 * MAX_LIST_SIZE];
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

// Improved SOGRAND kernel for a single row/column
__device__ void sogrand_siso_cuda_fixed(double* L_APP, double* L_E, double* llr, 
                                        int n, int k, SOGRANDState* state) {
    // Hard decision
    for (int i = 0; i < n; i++) {
        state->cHD[i] = (llr[i] > 0.0) ? 0 : 1;
        state->absL[i] = fabs(llr[i]);
        state->perm[i] = i;
    }
    
    // Sort by reliability (bubble sort for small n)
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
    int IC = 0; // Simplified intercept
    int wt_max = min(120, IC*n + n*(n+1)/2); // Limit maximum weight
    
    for (int wt = IC + 1; wt <= wt_max && state->curL < MAX_LIST_SIZE && state->T < 2000; wt++) {
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
        double pp0[31], pp1[31];
        for (int i = 0; i < n; i++) {
            pp1[i] = 1.0 / (1.0 + exp(llr[i]));
            pp0[i] = 1.0 - pp1[i];
            pp1[i] = fmax_safe(pp1[i], 1e-9);
            pp1[i] = fmin_safe(pp1[i], 1.0 - 1e-9);
            pp0[i] = fmax_safe(pp0[i], 1e-9);
            pp0[i] = fmin_safe(pp0[i], 1.0 - 1e-9);
        }
        
        // Compute probabilities using path metrics
        double p0[31] = {0}, p1[31] = {0};
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

// Kernel for row decoding
__global__ void decode_rows_kernel_fixed(double* L_channel, double* L_APP, double* L_E,
                                        double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || row >= n) return;
    
    __shared__ SOGRANDState states[8];
    SOGRANDState* state = &states[threadIdx.x % 8];
    
    int offset = block_id * n * n;
    
    // Prepare input
    double input[31];
    for (int col = 0; col < n; col++) {
        int idx = offset + row * n + col;
        input[col] = L_channel[idx] + alpha * L_E[idx];
    }
    
    // Run SOGRAND
    double L_APP_vec[31], L_E_vec[31];
    sogrand_siso_cuda_fixed(L_APP_vec, L_E_vec, input, n, k, state);
    
    // Write results
    for (int col = 0; col < n; col++) {
        int idx = offset + row * n + col;
        L_APP[idx] = L_APP_vec[col];
        L_E[idx] = L_E_vec[col];
    }
}

// Kernel for column decoding  
__global__ void decode_columns_kernel_fixed(double* L_channel, double* L_APP, double* L_E,
                                           double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_id >= num_blocks || col >= n) return;
    
    __shared__ SOGRANDState states[8];
    SOGRANDState* state = &states[threadIdx.x % 8];
    
    int offset = block_id * n * n;
    
    // Prepare input
    double input[31];
    for (int row = 0; row < n; row++) {
        int idx = offset + row * n + col;
        input[row] = L_channel[idx] + alpha * L_E[idx];
    }
    
    // Run SOGRAND
    double L_APP_vec[31], L_E_vec[31];
    sogrand_siso_cuda_fixed(L_APP_vec, L_E_vec, input, n, k, state);
    
    // Write results
    for (int row = 0; row < n; row++) {
        int idx = offset + row * n + col;
        L_APP[idx] = L_APP_vec[row];
        L_E[idx] = L_E_vec[row];
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

// Host function for square decoding
void decode_square_cuda_fixed(double* h_llr_buffer, int* h_bit_buffer, int num_blocks,
                             int n, int k, int Imax, double* alpha) {
    size_t block_size = n * n * sizeof(double);
    
    // Allocate device memory
    double *d_L_channel, *d_L_APP, *d_L_E;
    
    CHECK_CUDA(cudaMalloc(&d_L_channel, num_blocks * block_size));
    CHECK_CUDA(cudaMalloc(&d_L_APP, num_blocks * block_size));
    CHECK_CUDA(cudaMalloc(&d_L_E, num_blocks * block_size));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_L_channel, h_llr_buffer, num_blocks * block_size, 
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_L_E, 0, num_blocks * block_size));
    
    // Setup grid dimensions
    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, num_blocks);
    
    // Iterative decoding
    for (int iter = 1; iter <= Imax; iter++) {
        // Decode rows
        decode_rows_kernel_fixed<<<blocksPerGrid, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter-2], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // TODO: Add early termination check here
        
        // Decode columns
        decode_columns_kernel_fixed<<<blocksPerGrid, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter-1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // TODO: Add early termination check here
    }
    
    // Extract message bits
    double* h_L_APP = (double*)malloc(num_blocks * block_size);
    CHECK_CUDA(cudaMemcpy(h_L_APP, d_L_APP, num_blocks * block_size, 
                          cudaMemcpyDeviceToHost));
    
    // Hard decision to get bits from [0:k, 0:k]
    for (int b = 0; b < num_blocks; b++) {
        int bit_idx = 0;
        for (int row = 0; row < k; row++) {
            for (int col = 0; col < k; col++) {
                int llr_idx = b * n * n + row * n + col;
                h_bit_buffer[b * k * k + bit_idx] = (h_L_APP[llr_idx] > 0) ? 0 : 1;
                bit_idx++;
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

    const int n = 31;
    const int k = 25;
    const int codeword_block_size = n * n;
    const int message_block_size = k * k;
    const int Imax = 20;
    
    // Initialize alpha array
    double alpha[50];
    for(int i = 0; i < 50; i++) alpha[i] = 0.5;
    
    // Generate CRC matrices
    int h_G[25*31], h_H[6*31];
    getGH_sys_CRC(n, k, h_G, h_H);
    
    // Copy to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_G, h_G, sizeof(h_G)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_H, h_H, sizeof(h_H)));

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("CUDA Square Decoder Fixed (n=%d, k=%d)...\n", n, k);

    // Process in batches
    const int BATCH_SIZE = 64;
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
        decode_square_cuda_fixed(llr_batch, bit_batch, blocks_read, n, k, Imax, alpha);
        
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