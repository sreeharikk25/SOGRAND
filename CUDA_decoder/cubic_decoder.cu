#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128
#define MAX_LIST_SIZE 3
#define MAX_SEARCH_WEIGHT 5
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

// Enhanced SOGRAND state for n=15
struct SOGRANDState15 {
    double llr[15];
    double absL[15];
    int perm[15];
    uint8_t cHD[15];
    uint8_t c[15];
    uint8_t TEP[15];
    uint8_t chat_list[15 * MAX_LIST_SIZE];
    double PM_list[MAX_LIST_SIZE];
    int curL;
    int T;
    double pNL;
};

// Parity check for cubic code
__device__ bool parity_check_cuda_15(uint8_t* c, int n, int s) {
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
__device__ double compute_PM_15(uint8_t* TEP, double* absL, int* perm, int n) {
    double pm = 0.0;
    for (int i = 0; i < n; i++) {
        if (TEP[i] == 1) {
            pm += absL[perm[i]];
        }
    }
    return pm;
}

// Enhanced SOGRAND for cubic code
__device__ void sogrand_siso_cuda_15(double* L_APP, double* L_E, double* llr,
                                     int n, int k, SOGRANDState15* state) {
    // Hard decision
    for (int i = 0; i < n; i++) {
        state->cHD[i] = (llr[i] > 0.0) ? 0 : 1;
        state->absL[i] = fabs(llr[i]);
        state->perm[i] = i;
    }

    // Bubble sort for reliability ordering
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (state->absL[state->perm[j]] > state->absL[state->perm[j+1]]) {
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
    if (parity_check_cuda_15(state->c, n, n-k)) {
        for (int i = 0; i < n; i++) {
            state->chat_list[i] = state->c[i];
        }
        state->PM_list[0] = 0.0;
        state->curL = 1;
        state->pNL -= exp(-state->PM_list[0]);
    }

    // Search with increasing weight
    for (int w = 1; w <= MAX_SEARCH_WEIGHT && state->curL < MAX_LIST_SIZE && state->T < 500; w++) {
        int max_pos = min(n, 2*w + 3); // Limit search space for n=15

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

                if (parity_check_cuda_15(state->c, n, n-k)) {
                    double pm = compute_PM_15(state->TEP, state->absL, state->perm, n);
                    state->pNL -= exp(-pm);

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

                    if (parity_check_cuda_15(state->c, n, n-k)) {
                        double pm = compute_PM_15(state->TEP, state->absL, state->perm, n);
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
        // For w > 2, further limit search
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
        double pp0[15], pp1[15];
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
        double p0[15] = {0}, p1[15] = {0};

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

// Kernel for column decoding - NO RACE CONDITIONS
__global__ void decode_columns_cubic_kernel(double* L_channel, double* L_APP, double* L_E,
                                           double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || slice >= n || col >= n) return;

    // Each thread gets its own state
    SOGRANDState15 state;

    int offset = block_id * n * n * n;

    // Prepare input for this column
    double input[15];
    for (int row = 0; row < n; row++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[row] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_15(L_APP_vec, L_E_vec, input, n, k, &state);

    // Write results - each thread writes to different locations
    for (int row = 0; row < n; row++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[row];
        L_E[idx] = L_E_vec[row];
    }
}

// Kernel for row decoding - NO RACE CONDITIONS
__global__ void decode_rows_cubic_kernel(double* L_channel, double* L_APP, double* L_E,
                                         double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || slice >= n || row >= n) return;

    // Each thread gets its own state
    SOGRANDState15 state;

    int offset = block_id * n * n * n;

    // Prepare input for this row
    double input[15];
    for (int col = 0; col < n; col++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[col] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_15(L_APP_vec, L_E_vec, input, n, k, &state);

    // Write results - each thread writes to different locations
    for (int col = 0; col < n; col++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[col];
        L_E[idx] = L_E_vec[col];
    }
}

// Kernel for slice decoding - NO RACE CONDITIONS
__global__ void decode_slices_cubic_kernel(double* L_channel, double* L_APP, double* L_E,
                                          double alpha, int n, int k, int num_blocks) {
    int block_id = blockIdx.z;
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_id >= num_blocks || row >= n || col >= n) return;

    // Each thread gets its own state
    SOGRANDState15 state;

    int offset = block_id * n * n * n;

    // Prepare input for this slice vector
    double input[15];
    for (int slice = 0; slice < n; slice++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        input[slice] = L_channel[idx] + alpha * L_E[idx];
    }

    // Run SOGRAND
    double L_APP_vec[15], L_E_vec[15];
    sogrand_siso_cuda_15(L_APP_vec, L_E_vec, input, n, k, &state);

    // Write results - each thread writes to different locations
    for (int slice = 0; slice < n; slice++) {
        int idx = offset + tensor_idx(row, col, slice, n);
        L_APP[idx] = L_APP_vec[slice];
        L_E[idx] = L_E_vec[slice];
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
    dim3 threadsPerBlock(8);  // Reduced due to larger state
    dim3 blocksColumns((n + threadsPerBlock.x - 1) / threadsPerBlock.x, n, num_blocks);
    dim3 blocksRows((n + threadsPerBlock.x - 1) / threadsPerBlock.x, n, num_blocks);
    dim3 blocksSlices((n + threadsPerBlock.x - 1) / threadsPerBlock.x, n, num_blocks);

    // Iterative decoding
    for (int iter = 0; iter < Imax; iter++) {
        // Stage 1: Decode columns
        decode_columns_cubic_kernel<<<blocksColumns, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // Stage 2: Decode rows
        decode_rows_cubic_kernel<<<blocksRows, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter+1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // Stage 3: Decode slices
        decode_slices_cubic_kernel<<<blocksSlices, threadsPerBlock>>>(
            d_L_channel, d_L_APP, d_L_E, alpha[2*iter+1], n, k, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
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

// Proper matrix initialization for cubic code
void init_matrices_cubic(int* G_flat, int* H_flat) {
    const int n = 15;
    const int k = 10;
    const char* hex_poly = "0x15"; // For (15,10) code

    int poly_len;
    int* poly = koopman2matlab(hex_poly, &poly_len);
    int r = n - k;

    // Generate parity matrix P
    int** P = (int**)malloc(k * sizeof(int*));
    for(int i = 0; i < k; i++) P[i] = (int*)malloc(r * sizeof(int));
    int* msg_poly = (int*)calloc(k + r, sizeof(int));

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

    // Build parity check matrix H = [P^T | I_m]
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

    const int n = 15;
    const int k = 10;
    const int codeword_block_size = n * n * n;
    const int message_block_size = k * k * k;
    const int Imax = 30;

    // Initialize alpha array (damping factors - as per MATLAB)
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
