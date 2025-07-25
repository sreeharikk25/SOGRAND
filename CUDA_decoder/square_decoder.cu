#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define MAX_LIST_SIZE 4
#define MAX_SEARCH_WEIGHT 6
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

// Enhanced SOGRAND state structure
struct SOGRANDState {
    double llr[31];
    double absL[31];
    int perm[31];
    uint8_t cHD[31];
    uint8_t c[31];
    uint8_t TEP[31];
    uint8_t chat_list[31 * MAX_LIST_SIZE];
    double PM_list[MAX_LIST_SIZE];
    int curL;
    int T;
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
        if (syndrome != 0) return false;
    }
    return true;
}

// Compute path metric
__device__ double compute_PM(uint8_t* TEP, double* absL, int* perm, int n) {
    double pm = 0.0;
    for (int i = 0; i < n; i++) {
        if (TEP[i] == 1) {
            pm += absL[perm[i]];
        }
    }
    return pm;
}

// Enhanced SOGRAND kernel
__device__ void sogrand_siso_cuda(double* L_APP, double* L_E, double* llr,
                                  int n, int k, SOGRANDState* state) {
    // Hard decision
    hard_decision_cuda(llr, state->cHD, n);

    // Sort by reliability
    for (int i = 0; i < n; i++) {
        state->absL[i] = fabs(llr[i]);
        state->perm[i] = i;
    }

    // Bubble sort (can be optimized)
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (state->absL[state->perm[j]] > state->absL[state->perm[j+1]]) {
                // Swap indices
                int temp = state->perm[j];
                state->perm[j] = state->perm[j+1];
                state->perm[j+1] = temp;
            }
        }
    }

    // Initialize
    state->curL = 0;
    state->T = 0;
    state->pNL = 1.0;

    // Initialize TEP and check hard decision
    for (int i = 0; i < n; i++) {
        state->TEP[i] = 0;
        state->c[i] = state->cHD[i];
    }

    // Check hard decision (w=0)
    state->T++;
    if (parity_check_cuda(state->c, n, n-k)) {
        for (int i = 0; i < n; i++) {
            state->chat_list[i] = state->c[i];
        }
        state->PM_list[0] = 0.0;
        state->curL = 1;
        state->pNL -= exp(-state->PM_list[0]);
    }

    // Search with increasing weight
    for (int w = 1; w <= MAX_SEARCH_WEIGHT && state->curL < MAX_LIST_SIZE && state->T < 1000; w++) {
        // Generate all combinations of w flips among first positions
        int max_pos = min(n, 2*w + 5); // Limit search space

        // Simple enumeration for small w
        if (w == 1) {
            for (int i = 0; i < max_pos; i++) {
                // Set TEP
                for (int j = 0; j < n; j++) state->TEP[j] = 0;
                state->TEP[i] = 1;

                // Apply TEP
                for (int j = 0; j < n; j++) {
                    state->c[state->perm[j]] = state->cHD[state->perm[j]] ^ state->TEP[j];
                }

                state->T++;

                if (parity_check_cuda(state->c, n, n-k)) {
                    double pm = compute_PM(state->TEP, state->absL, state->perm, n);
                    state->pNL -= exp(-pm);

                    // Add to list
                    if (state->curL < MAX_LIST_SIZE) {
                        for (int j = 0; j < n; j++) {
                            state->chat_list[state->curL * n + j] = state->c[j];
                        }
                        state->PM_list[state->curL] = pm;
                        state->curL++;
                    } else {
                        // Replace worst if better
                        int worst_idx = 0;
                        double worst_pm = state->PM_list[0];
                        for (int l = 1; l < MAX_LIST_SIZE; l++) {
                            if (state->PM_list[l] > worst_pm) {
                                worst_pm = state->PM_list[l];
                                worst_idx = l;
                            }
                        }
                        if (pm < worst_pm) {
                            for (int j = 0; j < n; j++) {
                                state->chat_list[worst_idx * n + j] = state->c[j];
                            }
                            state->PM_list[worst_idx] = pm;
                        }
                    }
                }
            }
        } else if (w == 2) {
            for (int i = 0; i < max_pos-1; i++) {
                for (int j = i+1; j < max_pos; j++) {
                    // Set TEP
                    for (int l = 0; l < n; l++) state->TEP[l] = 0;
                    state->TEP[i] = 1;
                    state->TEP[j] = 1;

                    // Apply TEP
                    for (int l = 0; l < n; l++) {
                        state->c[state->perm[l]] = state->cHD[state->perm[l]] ^ state->TEP[l];
                    }

                    state->T++;

                    if (parity_check_cuda(state->c, n, n-k)) {
                        double pm = compute_PM(state->TEP, state->absL, state->perm, n);
                        state->pNL -= exp(-pm);

                        if (state->curL < MAX_LIST_SIZE) {
                            for (int l = 0; l < n; l++) {
                                state->chat_list[state->curL * n + l] = state->c[l];
                            }
                            state->PM_list[state->curL] = pm;
                            state->curL++;
                        }
                    }
                }
            }
        }
        // For w > 2, limit search further
    }

    state->pNL = fmax(state->pNL, 1e-10);

    // Compute APP based on list
    if (state->curL == 0) {
        // No valid codewords found
        for (int i = 0; i < n; i++) {
            L_APP[i] = llr[i];
            L_E[i] = 0;
        }
    } else {
        // Compute channel probabilities
        double pp0[31], pp1[31];
        for (int i = 0; i < n; i++) {
            double exp_llr = exp(llr[i]);
            pp0[i] = exp_llr / (1.0 + exp_llr);
            pp1[i] = 1.0 / (1.0 + exp_llr);
        }

        // Compute weights from path metrics
        double weights[MAX_LIST_SIZE];
        double weight_sum = 0.0;
        for (int l = 0; l < state->curL; l++) {
            weights[l] = exp(-state->PM_list[l]);
            weight_sum += weights[l];
        }

        // Normalize weights
        if (weight_sum > 0) {
            for (int l = 0; l < state->curL; l++) {
                weights[l] /= weight_sum;
            }
        }

        // Compute posterior probabilities
        double p0[31] = {0}, p1[31] = {0};

        for (int l = 0; l < state->curL; l++) {
            for (int i = 0; i < n; i++) {
                if (state->chat_list[l * n + i] == 1) {
                    p1[i] += weights[l];
                } else {
                    p0[i] += weights[l];
                }
            }
        }

        // Mix with channel probabilities using pNL
        for (int i = 0; i < n; i++) {
            p0[i] = p0[i] * (1.0 - state->pNL) + pp0[i] * state->pNL;
            p1[i] = p1[i] * (1.0 - state->pNL) + pp1[i] * state->pNL;

            // Ensure no zeros
            p0[i] = fmax(p0[i], 1e-30);
            p1[i] = fmax(p1[i], 1e-30);

            // Compute LLRs
            L_APP[i] = log(p0[i]) - log(p1[i]);
            L_E[i] = L_APP[i] - llr[i];
        }
    }
}

// Kernel for row decoding - NO RACE CONDITIONS
__global__ void decode_rows_kernel(double* L_channel, double* L_APP, double* L_E,
                                   double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || row >= n) return;

    // Each thread gets its own state - no sharing between threads
    SOGRANDState state;

    int offset = block_id * n * n;

    // Prepare input for this row
    double input[31];
    for (int col = 0; col < n; col++) {
        int idx = offset + row * n + col;
        input[col] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_row[31], L_E_row[31];
    sogrand_siso_cuda(L_APP_row, L_E_row, input, n, k, &state);

    // Write results - each thread writes to different memory locations
    for (int col = 0; col < n; col++) {
        int idx = offset + row * n + col;
        L_APP[idx] = L_APP_row[col];
        L_E[idx] = L_E_row[col];
    }
}

// Kernel for column decoding - NO RACE CONDITIONS
__global__ void decode_columns_kernel(double* L_channel, double* L_APP, double* L_E,
                                     double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || col >= n) return;

    // Each thread gets its own state
    SOGRANDState state;

    int offset = block_id * n * n;

    // Prepare input for this column
    double input[31];
    for (int row = 0; row < n; row++) {
        int idx = offset + row * n + col;
        input[row] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_col[31], L_E_col[31];
    sogrand_siso_cuda(L_APP_col, L_E_col, input, n, k, &state);

    // Write results - each thread writes to different memory locations
    for (int row = 0; row < n; row++) {
        int idx = offset + row * n + col;
        L_APP[idx] = L_APP_col[row];
        L_E[idx] = L_E_col[row];
    }
}

// Early termination check kernel
__global__ void check_early_termination_kernel(double* L_APP, bool* converged,
                                              int n, int k, int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    // Only thread 0 in each block does the check
    if (threadIdx.x == 0) {
        int offset = block_id * n * n;
        converged[block_id] = true;

        // Get hard decisions for systematic part
        uint8_t u_hat[25*25];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                u_hat[i * k + j] = (L_APP[offset + i * n + j] > 0) ? 0 : 1;
            }
        }

        // Re-encode and check
        // This is simplified - ideally would use the actual encoding
        // For now, just set converged to false to continue iterations
        converged[block_id] = false;
    }
}

// Host function for square decoding
void decode_square_cuda(double* h_llr_buffer, int* h_bit_buffer, int num_blocks,
                       int n, int k, int Imax, double* alpha) {
    size_t matrix_size = n * n * sizeof(double);

    // Allocate device memory
    double *d_L_channel, *d_L_APP, *d_L_E;
    bool *d_converged;

    CHECK_CUDA(cudaMalloc(&d_L_channel, num_blocks * matrix_size));
    CHECK_CUDA(cudaMalloc(&d_L_APP, num_blocks * matrix_size));
    CHECK_CUDA(cudaMalloc(&d_L_E, num_blocks * matrix_size));
    CHECK_CUDA(cudaMalloc(&d_converged, num_blocks * sizeof(bool)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_L_channel, h_llr_buffer, num_blocks * matrix_size,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_L_E, 0, num_blocks * matrix_size));

    // Setup grid dimensions
    dim3 threadsPerBlock(16);  // Reduced for larger state size
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, num_blocks);

    // Iterative decoding
    for (int iter = 0; iter < Imax; iter++) {
        // Decode rows
        decode_rows_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // Decode columns
        decode_columns_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter+1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // Check early termination every few iterations
        if ((iter + 1) % 5 == 0 && iter < Imax - 1) {
            check_early_termination_kernel<<<num_blocks, 1>>>(
                d_L_APP, d_converged, n, k, num_blocks);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
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
    CHECK_CUDA(cudaFree(d_converged));
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

// Matrix initialization
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

    // Initialize alpha array (damping factors - as per MATLAB)
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
            for (int byte_idx = 0; byte_idx < (message_block_size + 7) / 8; byte_idx++) {
                unsigned char byte_out = 0;
                for (int bit_idx = 0; bit_idx < 8 && (byte_idx * 8 + bit_idx) < message_block_size; bit_idx++) {
                    byte_out = (byte_out << 1) | bit_batch[b * message_block_size + byte_idx * 8 + bit_idx];
                }
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
