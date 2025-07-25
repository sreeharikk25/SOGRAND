#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

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

// Constants for n=15, k=10
__constant__ int d_G[10*15];  // Generator matrix
__constant__ int d_H[5*15];   // Parity check matrix

// 3D tensor access helpers - both host and device versions
__host__ __device__ inline int tensor_idx(int i, int j, int k, int n) {
    return k * n * n + j * n + i;
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
    double s_list[4 * MAX_LIST_SIZE];
    int curL;
    double T;
};

// Simplified SOGRAND for cubic code
__device__ void sogrand_siso_cuda_15(double* L_APP, double* L_E, double* llr,
                                     int n, int k, SOGRANDState15* state) {
    // Hard decision
    for (int i = 0; i < n; i++) {
        state->cHD[i] = (llr[i] > 0.0) ? 0 : 1;
        state->absL[i] = fabs(llr[i]);
        state->perm[i] = i;
    }

    // Simple bubble sort for reliability ordering
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
    }
    state->curL = 0;
    state->T = 1;

    // Check hard decision
    bool valid = true;
    for (int j = 0; j < (n-k); j++) {
        uint8_t syndrome = 0;
        for (int i = 0; i < n; i++) {
            syndrome ^= (state->c[i] * d_H[j*n + i]);
        }
        if (syndrome != 0) {
            valid = false;
            break;
        }
    }

    if (valid) {
        for (int i = 0; i < n; i++) {
            state->chat_list[i] = state->c[i];
        }
        state->curL = 1;
    }

    // Limited TEP search
    int max_flips = min(4, n);  // Increased from 3 to 4 for better performance
    for (int w = 1; w <= max_flips && state->curL < MAX_LIST_SIZE; w++) {
        // Flip w least reliable bits
        for (int i = 0; i < n; i++) state->TEP[i] = 0;
        for (int i = 0; i < w; i++) state->TEP[i] = 1;

        // Apply TEP
        for (int i = 0; i < n; i++) {
            state->c[state->perm[i]] = state->cHD[state->perm[i]] ^ state->TEP[i];
        }

        state->T++;

        // Check validity
        valid = true;
        for (int j = 0; j < (n-k); j++) {
            uint8_t syndrome = 0;
            for (int i = 0; i < n; i++) {
                syndrome ^= (state->c[i] * d_H[j*n + i]);
            }
            if (syndrome != 0) {
                valid = false;
                break;
            }
        }

        if (valid) {
            for (int i = 0; i < n; i++) {
                state->chat_list[state->curL * n + i] = state->c[i];
            }
            state->curL++;
        }
    }

    // Compute APP
    if (state->curL == 0) {
        // No valid codewords found, use channel LLRs
        for (int i = 0; i < n; i++) {
            L_APP[i] = llr[i];
            L_E[i] = 0;
        }
    } else {
        double pp0[15], pp1[15];
        for (int i = 0; i < n; i++) {
            pp1[i] = 1.0 / (1.0 + exp(llr[i]));
            pp0[i] = 1.0 - pp1[i];
        }

        double p0[15] = {0}, p1[15] = {0};
        double weight = 1.0 / state->curL;

        for (int l = 0; l < state->curL; l++) {
            for (int i = 0; i < n; i++) {
                if (state->chat_list[l * n + i] == 1) {
                    p1[i] += weight;
                } else {
                    p0[i] += weight;
                }
            }
        }

        // Mix with channel probabilities
        for (int i = 0; i < n; i++) {
            p0[i] = p0[i] * 0.8 + pp0[i] * 0.2;  // Adjusted mixing ratio
            p1[i] = p1[i] * 0.8 + pp1[i] * 0.2;
            L_APP[i] = log(fmax(p0[i], 1e-30)) - log(fmax(p1[i], 1e-30));
            L_E[i] = L_APP[i] - llr[i];
        }
    }
}

// Kernel for column decoding with reduced shared memory
__global__ void decode_columns_cubic_kernel(double* L_channel, double* L_APP, double* L_E,
                                           double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || slice >= n || col >= n) return;

    __shared__ SOGRANDState15 states[4];
    SOGRANDState15* state = &states[threadIdx.x % 4];

    int offset = block_id * n * n * n;

    // Prepare input for this column
    double input[15];
    for (int row = 0; row < n; row++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[row] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_15(L_APP_vec, L_E_vec, input, n, k, state);

    // Write results
    for (int row = 0; row < n; row++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[row];
        L_E[idx] = L_E_vec[row];
    }
}

// Kernel for row decoding with reduced shared memory
__global__ void decode_rows_cubic_kernel(double* L_channel, double* L_APP, double* L_E,
                                         double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || slice >= n || row >= n) return;

    __shared__ SOGRANDState15 states[4];
    SOGRANDState15* state = &states[threadIdx.x % 4];

    int offset = block_id * n * n * n;

    // Prepare input for this row
    double input[15];
    for (int col = 0; col < n; col++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[col] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_15(L_APP_vec, L_E_vec, input, n, k, state);

    // Write results
    for (int col = 0; col < n; col++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[col];
        L_E[idx] = L_E_vec[col];
    }
}

// Kernel for slice decoding with reduced shared memory
__global__ void decode_slices_cubic_kernel(double* L_channel, double* L_APP, double* L_E,
                                          double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || row >= n || col >= n) return;

    __shared__ SOGRANDState15 states[4];
    SOGRANDState15* state = &states[threadIdx.x % 4];

    int offset = block_id * n * n * n;

    // Prepare input for this slice vector
    double input[15];
    for (int slice = 0; slice < n; slice++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[slice] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_15(L_APP_vec, L_E_vec, input, n, k, state);

    // Write results
    for (int slice = 0; slice < n; slice++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[slice];
        L_E[idx] = L_E_vec[slice];
    }
}

// Proper matrix initialization for (15,10) CRC code
void init_matrices_cubic(int* G_flat, int* H_flat) {
    const int n = 15;
    const int k = 10;
    const int m = n - k; // 5

    // CRC polynomial: 0x15 = 10101 in binary = x^4 + x^2 + 1
    const int crc_poly = 0x15;

    // First, create the parity matrix P (k x m)
    int P[10][5];

    // For systematic CRC code, each row of P is computed by dividing x^(m+i) by g(x)
    for (int i = 0; i < k; i++) {
        // Start with x^(m+i) = x^(5+i)
        unsigned int dividend = 1 << (m + k - 1 - i);

        // Perform polynomial division
        for (int j = 0; j < k; j++) {
            if (dividend & (1 << (n - 1 - j))) {
                dividend ^= (crc_poly << (n - m - 1 - j));
            }
        }

        // The remainder gives us the parity bits for this row
        for (int j = 0; j < m; j++) {
            P[i][j] = (dividend >> (m - 1 - j)) & 1;
        }
    }

    // Construct generator matrix G = [I_k | P]
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if (j < k) {
                // Identity part
                G_flat[i * n + j] = (i == j) ? 1 : 0;
            } else {
                // Parity part
                G_flat[i * n + j] = P[i][j - k];
            }
        }
    }

    // Construct parity check matrix H = [P^T | I_m]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (j < k) {
                // P transpose part
                H_flat[i * n + j] = P[j][i];
            } else {
                // Identity part
                H_flat[i * n + j] = ((j - k) == i) ? 1 : 0;
            }
        }
    }
}

// Host function for cubic decoding
void decode_cubic_cuda(double* h_llr_buffer, int* h_bit_buffer, int num_blocks,
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
    for (int iter = 0; iter < Imax; iter++) {
        // Stage 1: Decode columns
        decode_columns_cubic_kernel<<<blocksColumns, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Stage 2: Decode rows
        decode_rows_cubic_kernel<<<blocksRows, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter+1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Stage 3: Decode slices
        decode_slices_cubic_kernel<<<blocksSlices, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter+1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
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

    // Initialize alpha array (damping factors)
    double alpha[100];
    for(int i = 0; i < 100; i++) alpha[i] = 0.7;

    // Setup matrices
    int h_G[10*15], h_H[5*15];
    init_matrices_cubic(h_G, h_H);

    // Copy to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_G, h_G, sizeof(h_G)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_H, h_H, sizeof(h_H)));

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("CUDA Cubic Decoder (n=%d, k=%d)...\n", n, k);

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
        decode_cubic_cuda(llr_batch, bit_batch, blocks_read, n, k, Imax, alpha);

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
