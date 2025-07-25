#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

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

// Constants
__constant__ int d_G[25*31];  // Generator matrix
__constant__ int d_H[6*31];   // Parity check matrix

// Structure for SOGRAND state
struct SOGRANDState {
    double llr[31];
    double absL[31];
    int perm[31];
    uint8_t cHD[31];
    uint8_t c[31];
    uint8_t TEP[31];
    double chat_list[31 * MAX_LIST_SIZE];
    double s_list[4 * MAX_LIST_SIZE];
    double APP_list[MAX_LIST_SIZE];
    int curL;
    double T;
    double pNL;
};

// CUDA kernel for hard decision
__device__ void hard_decision_cuda(double* llr, uint8_t* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = (llr[i] > 0.0) ? 0 : 1;
    }
}

// CUDA kernel for parity check
__device__ bool parity_check_cuda(uint8_t* c, int n, int s) {
    for (int j = 0; j < s; j++) {
        uint8_t syndrome = 0;
        for (int i = 0; i < n; i++) {
            syndrome ^= (c[i] * d_H[j*n + i]);
        }
        if (syndrome != 0) return false;  // Fixed: should be != 0, not == 1
    }
    return true;
}

// Simplified SOGRAND kernel for a single row/column
__device__ void sogrand_siso_cuda(double* L_APP, double* L_E, double* llr,
                                  int n, int k, SOGRANDState* state) {
    // Initialize
    hard_decision_cuda(llr, state->cHD, n);

    // Sort by reliability (simplified - using bubble sort for now)
    for (int i = 0; i < n; i++) {
        state->absL[i] = fabs(llr[i]);
        state->perm[i] = i;
    }

    // Simple bubble sort (can be optimized with parallel sorting)
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

    // Copy hard decision
    for (int i = 0; i < n; i++) {
        state->c[i] = state->cHD[i];
    }

    state->curL = 0;
    state->T = 1;

    // Check if hard decision is valid
    if (parity_check_cuda(state->c, n, n-k)) {
        for (int i = 0; i < n; i++) {
            state->chat_list[i] = state->c[i];
        }
        state->curL = 1;
    }

    // Simplified TEP generation (limited search)
    int max_flips = min(4, n);
    for (int w = 1; w <= max_flips && state->curL < MAX_LIST_SIZE; w++) {
        // Flip w least reliable bits
        for (int i = 0; i < n; i++) state->TEP[i] = 0;
        for (int i = 0; i < w; i++) state->TEP[i] = 1;

        // Apply TEP
        for (int i = 0; i < n; i++) {
            state->c[state->perm[i]] = state->cHD[state->perm[i]] ^ state->TEP[i];
        }

        state->T++;

        if (parity_check_cuda(state->c, n, n-k)) {
            for (int i = 0; i < n; i++) {
                state->chat_list[state->curL * n + i] = state->c[i];
            }
            state->curL++;
        }
    }

    // Compute APP (simplified)
    if (state->curL == 0) {
        for (int i = 0; i < n; i++) {
            L_APP[i] = llr[i];
            L_E[i] = 0;
        }
    } else {
        // Simplified APP computation
        double pp0[31], pp1[31];
        for (int i = 0; i < n; i++) {
            pp1[i] = 1.0 / (1.0 + exp(llr[i]));
            pp0[i] = 1.0 - pp1[i];
        }

        double p0[31] = {0}, p1[31] = {0};
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

        for (int i = 0; i < n; i++) {
            p0[i] = p0[i] * 0.9 + pp0[i] * 0.1;
            p1[i] = p1[i] * 0.9 + pp1[i] * 0.1;
            L_APP[i] = log(fmax(p0[i], 1e-30)) - log(fmax(p1[i], 1e-30));
            L_E[i] = L_APP[i] - llr[i];
        }
    }
}

// Kernel for row decoding with reduced shared memory usage
__global__ void decode_rows_kernel(double* L_channel, double* L_APP, double* L_E,
                                   double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || row >= n) return;

    __shared__ SOGRANDState states[8];

    int offset = block_id * n * n;
    SOGRANDState* state = &states[threadIdx.x % 8];

    // Prepare input
    double input[31];
    for (int col = 0; col < n; col++) {
        int idx = offset + row * n + col;
        input[col] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_row[31], L_E_row[31];
    sogrand_siso_cuda(L_APP_row, L_E_row, input, n, k, state);

    // Write results
    for (int col = 0; col < n; col++) {
        int idx = offset + row * n + col;
        L_APP[idx] = L_APP_row[col];
        L_E[idx] = L_E_row[col];
    }
}

// Kernel for column decoding with reduced shared memory usage
__global__ void decode_columns_kernel(double* L_channel, double* L_APP, double* L_E,
                                     double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || col >= n) return;

    __shared__ SOGRANDState states[8];

    int offset = block_id * n * n;
    SOGRANDState* state = &states[threadIdx.x % 8];

    // Prepare input
    double input[31];
    for (int row = 0; row < n; row++) {
        int idx = offset + row * n + col;
        input[row] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_col[31], L_E_col[31];
    sogrand_siso_cuda(L_APP_col, L_E_col, input, n, k, state);

    // Write results
    for (int row = 0; row < n; row++) {
        int idx = offset + row * n + col;
        L_APP[idx] = L_APP_col[row];
        L_E[idx] = L_E_col[row];
    }
}

// Host function for square decoding
void decode_square_cuda(double* h_llr_buffer, int* h_bit_buffer, int num_blocks,
                       int n, int k, int Imax, double* alpha) {
    size_t matrix_size = n * n * sizeof(double);

    // Allocate device memory
    double *d_L_channel, *d_L_APP, *d_L_E;

    CHECK_CUDA(cudaMalloc(&d_L_channel, num_blocks * matrix_size));
    CHECK_CUDA(cudaMalloc(&d_L_APP, num_blocks * matrix_size));
    CHECK_CUDA(cudaMalloc(&d_L_E, num_blocks * matrix_size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_L_channel, h_llr_buffer, num_blocks * matrix_size,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_L_E, 0, num_blocks * matrix_size));

    // Setup grid dimensions
    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, num_blocks);

    // Iterative decoding
    for (int iter = 0; iter < Imax; iter++) {
        // Decode rows
        decode_rows_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Decode columns
        decode_columns_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter+1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Extract message bits
    double* h_L_APP = (double*)malloc(num_blocks * matrix_size);
    CHECK_CUDA(cudaMemcpy(h_L_APP, d_L_APP, num_blocks * matrix_size,
                          cudaMemcpyDeviceToHost));

    // Hard decision to get bits
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                int llr_idx = b * n * n + i * n + j;
                int bit_idx = b * k * k + i * k + j;
                h_bit_buffer[bit_idx] = (h_L_APP[llr_idx] > 0) ? 0 : 1;
            }
        }
    }

    free(h_L_APP);
    CHECK_CUDA(cudaFree(d_L_channel));
    CHECK_CUDA(cudaFree(d_L_APP));
    CHECK_CUDA(cudaFree(d_L_E));
}

// CRC polynomial conversion
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

// Proper matrix initialization for square code (copied from encoder)
void init_matrices_square(int* G_flat, int* H_flat, int n, int k) {
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

    // Generate parity matrix P
    int** P = (int**)malloc(k * sizeof(int*));
    for(int i = 0; i < k; i++) P[i] = (int*)malloc(r * sizeof(int));
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

    // Build generator matrix G = [I_k | P]
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            G_flat[i * n + j] = (i == j) ? 1 : 0;
        }
        for (int j = 0; j < r; j++) {
            G_flat[i * n + k + j] = P[i][j];
        }
    }

    // Build parity check matrix H = [P^T | I_r]
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < k; j++) {
            H_flat[i * n + j] = P[j][i];
        }
        for (int j = 0; j < r; j++) {
            H_flat[i * n + k + j] = (i == j) ? 1 : 0;
        }
    }

    // Cleanup
    free(poly);
    free(msg_poly);
    for(int i = 0; i < k; i++) free(P[i]);
    free(P);
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

    // Setup generator and parity check matrices
    int h_G[25*31], h_H[6*31];
    init_matrices_square(h_G, h_H, n, k);

    // Copy matrices to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_G, h_G, sizeof(h_G)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_H, h_H, sizeof(h_H)));

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("CUDA Square Decoder (n=%d, k=%d)...\n", n, k);

    // Process multiple blocks at once for better GPU utilization
    const int BATCH_SIZE = 32;
    double* llr_batch = (double*)malloc(BATCH_SIZE * codeword_block_size * sizeof(double));
    int* bit_batch = (int*)malloc(BATCH_SIZE * message_block_size * sizeof(int));

    int total_blocks = 0;

    while (true) {
        // Read batch of blocks
        int blocks_read = 0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            size_t read = fread(&llr_batch[i * codeword_block_size],
                               sizeof(double), codeword_block_size, fin);
            if (read != codeword_block_size) break;
            blocks_read++;
        }

        if (blocks_read == 0) break;

        // Decode batch on GPU
        decode_square_cuda(llr_batch, bit_batch, blocks_read, n, k, Imax, alpha);

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
