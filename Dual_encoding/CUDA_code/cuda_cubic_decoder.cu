#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Inf 0x7fffffff
#define MAX_BLOCKS 200

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Improved SOGRAND - more thorough error correction
__device__ void improved_sogrand_device(double* L_APP, double* L_E, int* N_guess, double* llr,
                                        uint8_t* H_flat, int n, int k, int L, double thres, int even) {
    // Local arrays
    uint8_t cHD[15], c[15];
    double absL[15];
    size_t perm[15];
    double PM_list[16]; // Store more candidates
    uint8_t codeword_list[16*15]; // Store more codewords
    int valid_count = 0;

    // Initialize
    for(int i = 0; i < n; i++) {
        perm[i] = i;
        cHD[i] = (llr[i] > 0.0) ? 0 : 1;
        absL[i] = fabs(llr[i]);
        c[i] = cHD[i];
    }

    int NG = 1;

    // Check hard decision first
    int valid = 1;
    for (int j = 0; j < (n-k); j++) {
        uint8_t syndrome = 0;
        for (int i = 0; i < n; i++)
            syndrome ^= (c[i] * H_flat[j*n + i]);
        if (syndrome == 1) {
            valid = 0;
            break;
        }
    }

    if (valid) {
        for(int i = 0; i < n; i++) {
            L_APP[i] = llr[i];
            L_E[i] = 0.0;
        }
        if (N_guess) *N_guess = NG;
        return;
    }

    // Sort by reliability
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (absL[j] < absL[j+1]) {
                double temp = absL[j];
                absL[j] = absL[j+1];
                absL[j+1] = temp;

                size_t temp_perm = perm[j];
                perm[j] = perm[j+1];
                perm[j+1] = temp_perm;
            }
        }
    }

    // More thorough search - try more error patterns

    // Weight 1 errors - try more positions
    for (int pos = n-1; pos >= n-8 && pos >= 0 && valid_count < 8; pos--) {
        for(int i = 0; i < n; i++) c[i] = cHD[i];
        c[perm[pos]] = 1 - c[perm[pos]];
        NG++;

        int is_valid = 1;
        for (int j = 0; j < (n-k); j++) {
            uint8_t syndrome = 0;
            for (int i = 0; i < n; i++)
                syndrome ^= (c[i] * H_flat[j*n + i]);
            if (syndrome == 1) {
                is_valid = 0;
                break;
            }
        }

        if (is_valid && valid_count < 16) {
            double pm = absL[pos]; // Simple reliability-based metric
            PM_list[valid_count] = pm;
            for(int i = 0; i < n; i++) {
                codeword_list[valid_count * n + i] = c[i];
            }
            valid_count++;
        }
    }

    // Weight 2 errors - try more combinations
    for (int pos1 = n-1; pos1 >= n-6 && pos1 >= 0 && valid_count < 12; pos1--) {
        for (int pos2 = pos1-1; pos2 >= n-8 && pos2 >= 0 && valid_count < 12; pos2--) {
            for(int i = 0; i < n; i++) c[i] = cHD[i];
            c[perm[pos1]] = 1 - c[perm[pos1]];
            c[perm[pos2]] = 1 - c[perm[pos2]];
            NG++;

            int is_valid = 1;
            for (int j = 0; j < (n-k); j++) {
                uint8_t syndrome = 0;
                for (int i = 0; i < n; i++)
                    syndrome ^= (c[i] * H_flat[j*n + i]);
                if (syndrome == 1) {
                    is_valid = 0;
                    break;
                }
            }

            if (is_valid && valid_count < 16) {
                double pm = absL[pos1] + absL[pos2];
                PM_list[valid_count] = pm;
                for(int i = 0; i < n; i++) {
                    codeword_list[valid_count * n + i] = c[i];
                }
                valid_count++;
            }
        }
    }

    // Weight 3 errors - try a few
    for (int pos1 = n-1; pos1 >= n-4 && pos1 >= 0 && valid_count < 14; pos1--) {
        for (int pos2 = pos1-1; pos2 >= n-6 && pos2 >= 0 && valid_count < 14; pos2--) {
            for (int pos3 = pos2-1; pos3 >= n-8 && pos3 >= 0 && valid_count < 14; pos3--) {
                for(int i = 0; i < n; i++) c[i] = cHD[i];
                c[perm[pos1]] = 1 - c[perm[pos1]];
                c[perm[pos2]] = 1 - c[perm[pos2]];
                c[perm[pos3]] = 1 - c[perm[pos3]];
                NG++;

                int is_valid = 1;
                for (int j = 0; j < (n-k); j++) {
                    uint8_t syndrome = 0;
                    for (int i = 0; i < n; i++)
                        syndrome ^= (c[i] * H_flat[j*n + i]);
                    if (syndrome == 1) {
                        is_valid = 0;
                        break;
                    }
                }

                if (is_valid && valid_count < 16) {
                    double pm = absL[pos1] + absL[pos2] + absL[pos3];
                    PM_list[valid_count] = pm;
                    for(int i = 0; i < n; i++) {
                        codeword_list[valid_count * n + i] = c[i];
                    }
                    valid_count++;
                }
            }
        }
    }

    if (valid_count > 0) {
        // Find best codeword (lowest path metric = most reliable)
        int best_idx = 0;
        double best_pm = PM_list[0];
        for (int i = 1; i < valid_count; i++) {
            if (PM_list[i] < best_pm) {
                best_pm = PM_list[i];
                best_idx = i;
            }
        }

        // Use best codeword
        for(int i = 0; i < n; i++) {
            uint8_t best_bit = codeword_list[best_idx * n + i];
            if (best_bit == cHD[i]) {
                L_APP[i] = llr[i];
            } else {
                L_APP[i] = -llr[i]; // Flip LLR for corrected bit
            }
            L_E[i] = L_APP[i] - llr[i];
        }
    } else {
        // No valid codeword found, use hard decision
        for(int i = 0; i < n; i++) {
            L_APP[i] = llr[i];
            L_E[i] = 0.0;
        }
    }

    if (N_guess) *N_guess = NG;
}

// CUDA kernels with improved algorithm
__global__ void improved_columns_kernel(double* input_data, double* L_APP_data, double* L_E_data,
                                        uint8_t* H_flat, int n, int k, int L, double thres,
                                        int even, int* NG_total, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.x;
    int col = threadIdx.x;

    if (block_id >= num_blocks || slice >= n || col >= n) return;

    int block_offset = block_id * n * n * n;
    double vec_in[15], vec_L_APP[15], vec_L_E[15];
    int N_guess = 0;

    for(int row = 0; row < n; row++) {
        int idx = block_offset + slice * n * n + row * n + col;
        vec_in[row] = input_data[idx];
    }

    improved_sogrand_device(vec_L_APP, vec_L_E, &N_guess, vec_in, H_flat, n, k, L, thres, even);

    for(int row = 0; row < n; row++) {
        int idx = block_offset + slice * n * n + row * n + col;
        L_APP_data[idx] = vec_L_APP[row];
        L_E_data[idx] = vec_L_E[row];
    }

    atomicAdd(NG_total, N_guess);
}

__global__ void improved_rows_kernel(double* input_data, double* L_APP_data, double* L_E_data,
                                     uint8_t* H_flat, int n, int k, int L, double thres,
                                     int even, int* NG_total, int num_blocks) {
    int block_id = blockIdx.z;
    int slice = blockIdx.x;
    int row = threadIdx.x;

    if (block_id >= num_blocks || slice >= n || row >= n) return;

    int block_offset = block_id * n * n * n;
    double vec_in[15], vec_L_APP[15], vec_L_E[15];
    int N_guess = 0;

    for(int col = 0; col < n; col++) {
        int idx = block_offset + slice * n * n + row * n + col;
        vec_in[col] = input_data[idx];
    }

    improved_sogrand_device(vec_L_APP, vec_L_E, &N_guess, vec_in, H_flat, n, k, L, thres, even);

    for(int col = 0; col < n; col++) {
        int idx = block_offset + slice * n * n + row * n + col;
        L_APP_data[idx] = vec_L_APP[col];
        L_E_data[idx] = vec_L_E[col];
    }

    atomicAdd(NG_total, N_guess);
}

__global__ void improved_slices_kernel(double* input_data, double* L_APP_data, double* L_E_data,
                                       uint8_t* H_flat, int n, int k, int L, double thres,
                                       int even, int* NG_total, int num_blocks) {
    int block_id = blockIdx.z;
    int col = blockIdx.x;
    int row = threadIdx.x;

    if (block_id >= num_blocks || col >= n || row >= n) return;

    int block_offset = block_id * n * n * n;
    double vec_in[15], vec_L_APP[15], vec_L_E[15];
    int N_guess = 0;

    for(int slice_idx = 0; slice_idx < n; slice_idx++) {
        int idx = block_offset + slice_idx * n * n + row * n + col;
        vec_in[slice_idx] = input_data[idx];
    }

    improved_sogrand_device(vec_L_APP, vec_L_E, &N_guess, vec_in, H_flat, n, k, L, thres, even);

    for(int slice_idx = 0; slice_idx < n; slice_idx++) {
        int idx = block_offset + slice_idx * n * n + row * n + col;
        L_APP_data[idx] = vec_L_APP[slice_idx];
        L_E_data[idx] = vec_L_E[slice_idx];
    }

    atomicAdd(NG_total, N_guess);
}

// Proper early termination
__global__ void proper_early_termination_kernel(double* L_APP_data, int* G_flat, int n, int k,
                                                int* results, int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    extern __shared__ int shared_mem[];
    int* c_HD_data = shared_mem;
    int* c_test_data = &shared_mem[n*n*n];

    int block_offset = block_id * n * n * n;

    // Hard decision
    for(int i = 0; i < n*n*n; i++) {
        c_HD_data[i] = (L_APP_data[block_offset + i] > 0) ? 0 : 1;
    }

    // Copy systematic part
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = 0; col < k; col++) {
                int idx = slice * n * n + row * n + col;
                c_test_data[idx] = c_HD_data[idx];
            }
        }
    }

    // Encode and check
    // Encode rows
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = k; col < n; col++) {
                int parity_val = 0;
                for (int msg_bit_idx = 0; msg_bit_idx < k; msg_bit_idx++) {
                    int msg_idx = slice * n * n + row * n + msg_bit_idx;
                    parity_val += c_test_data[msg_idx] * G_flat[msg_bit_idx * n + col];
                }
                int col_idx = slice * n * n + row * n + col;
                c_test_data[col_idx] = parity_val % 2;
            }
        }
    }

    // Quick check - count differences in parity positions only
    int error_count = 0;
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = k; col < n; col++) {
                int test_idx = slice * n * n + row * n + col;
                if (c_test_data[test_idx] != c_HD_data[test_idx]) {
                    error_count++;
                }
            }
        }
    }

    results[block_id] = (error_count <= 2) ? 1 : 0; // Allow small number of errors
}

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
    const char* hex_poly = "0x15";
    int r = n - k;

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

int proper_early_termination_cuda(double* L_APP_data, int** G, int n, int k, int num_blocks) {
    int* G_flat;
    CUDA_CHECK(cudaMallocManaged(&G_flat, k * n * sizeof(int)));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            G_flat[i * n + j] = G[i][j];
        }
    }

    int* results;
    CUDA_CHECK(cudaMallocManaged(&results, num_blocks * sizeof(int)));

    size_t shared_mem_size = 2 * n * n * n * sizeof(int);
    proper_early_termination_kernel<<<num_blocks, 1, shared_mem_size>>>(L_APP_data, G_flat, n, k, results, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    int converged_count = 0;
    for (int i = 0; i < num_blocks; i++) {
        if (results[i] == 1) converged_count++;
    }

    CUDA_CHECK(cudaFree(G_flat));
    CUDA_CHECK(cudaFree(results));

    return (converged_count >= (num_blocks * 3) / 4) ? 1 : 0; // 75% threshold
}

void batch_hard_decision(double* llr, int* bits, int num_blocks, int message_block_size) {
    int k = 10;
    int n = 15;

    for (int block_id = 0; block_id < num_blocks; block_id++) {
        int bit_idx = 0;
        int block_offset = block_id * n * n * n;
        int output_offset = block_id * message_block_size;

        for (int slice = 0; slice < k; slice++) {
            for (int row = 0; row < k; row++) {
                for (int col = 0; col < k; col++) {
                    int tensor_idx = block_offset + slice * n * n + row * n + col;
                    bits[output_offset + bit_idx++] = (llr[tensor_idx] > 0) ? 0 : 1;
                }
            }
        }
    }
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

    int** G = (int**)malloc(k * sizeof(int*));
    int** H = (int**)malloc((n-k) * sizeof(int*));
    for(int i = 0; i < k; i++) G[i] = (int*)malloc(n * sizeof(int));
    for(int i = 0; i < (n-k); i++) H[i] = (int*)malloc(n * sizeof(int));
    getGH_sys_CRC(n, k, G, H);

    uint8_t* H_flat;
    CUDA_CHECK(cudaMallocManaged(&H_flat, (n-k) * n * sizeof(uint8_t)));
    for (int i = 0; i < (n-k); i++) {
        for (int j = 0; j < n; j++) {
            H_flat[i*n + j] = (uint8_t)H[i][j];
        }
    }

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
    const int Imax = 10;
    const double p_ET = 1e-5;
    const double thres = 1.0 - p_ET;

    double alpha[50];
    for(int i = 0; i < 50; i++) alpha[i] = 0.7;

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("Improved CUDA Decoding %s to %s using cubic code (n=%d, k=%d)...\n", input_filename, output_filename, n, k);

    // Get file size to calculate exact number of blocks
    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    int total_expected_blocks = (int)(file_size / (codeword_block_size * sizeof(double)));
    int blocks_processed = 0;

    printf("File size: %ld bytes, Expected blocks: %d\n", file_size, total_expected_blocks);

    // Pre-allocate GPU memory
    double* d_L_channel;
    double* d_L_APP;
    double* d_L_E;
    double* d_L_A;
    double* d_input;
    int* d_NG_total;

    const int max_batch_size = MAX_BLOCKS;
    size_t batch_memory_size = max_batch_size * codeword_block_size * sizeof(double);

    CUDA_CHECK(cudaMallocManaged(&d_L_channel, batch_memory_size));
    CUDA_CHECK(cudaMallocManaged(&d_L_APP, batch_memory_size));
    CUDA_CHECK(cudaMallocManaged(&d_L_E, batch_memory_size));
    CUDA_CHECK(cudaMallocManaged(&d_L_A, batch_memory_size));
    CUDA_CHECK(cudaMallocManaged(&d_input, batch_memory_size));
    CUDA_CHECK(cudaMallocManaged(&d_NG_total, sizeof(int)));

    double* h_llr_batch = (double*)malloc(batch_memory_size);
    int* h_bit_batch = (int*)malloc(max_batch_size * message_block_size * sizeof(int));

    int total_NG = 0;
    double total_iterations = 0;
    int total_blocks = 0;

    clock_t start_time = clock();

    // Batch processing loop
    while (!feof(fin) && blocks_processed < total_expected_blocks) {
        // Read batch of blocks
        int blocks_in_batch = 0;
        while (blocks_in_batch < max_batch_size && !feof(fin) && blocks_processed + blocks_in_batch < total_expected_blocks) {
            size_t offset = blocks_in_batch * codeword_block_size;
            size_t read = fread(&h_llr_batch[offset], sizeof(double), codeword_block_size, fin);
            if (read == codeword_block_size) {
                blocks_in_batch++;
            } else if (read > 0) {
                printf("Warning: Incomplete block read (%zu doubles)\n", read);
                break;
            }
        }

        if (blocks_in_batch == 0) break;
        blocks_processed += blocks_in_batch;

        size_t batch_size = blocks_in_batch * codeword_block_size * sizeof(double);
        memcpy(d_L_channel, h_llr_batch, batch_size);
        CUDA_CHECK(cudaMemset(d_L_E, 0, batch_size));

        double batch_iterations = 0;
        int batch_NG = 0;

        for (int iter = 1; iter <= Imax; iter++) {
            // Columns processing
            batch_iterations += 0.5;
            *d_NG_total = 0;

            for(int idx = 0; idx < blocks_in_batch * codeword_block_size; idx++) {
                d_L_A[idx] = alpha[2*iter-2] * d_L_E[idx];
                d_input[idx] = d_L_channel[idx] + d_L_A[idx];
            }

            dim3 grid1(n, 1, blocks_in_batch);
            dim3 block1(n, 1, 1);
            improved_columns_kernel<<<grid1, block1>>>(
                d_input, d_L_APP, d_L_E, H_flat, n, k, L, thres, even, d_NG_total, blocks_in_batch);
            CUDA_CHECK(cudaDeviceSynchronize());

            batch_NG += *d_NG_total;

            if (proper_early_termination_cuda(d_L_APP, G, n, k, blocks_in_batch)) break;

            // Rows processing
            batch_iterations += 0.5;
            *d_NG_total = 0;

            for(int idx = 0; idx < blocks_in_batch * codeword_block_size; idx++) {
                d_L_A[idx] = alpha[2*iter-1] * d_L_E[idx];
                d_input[idx] = d_L_channel[idx] + d_L_A[idx];
            }

            dim3 grid2(n, 1, blocks_in_batch);
            dim3 block2(n, 1, 1);
            improved_rows_kernel<<<grid2, block2>>>(
                d_input, d_L_APP, d_L_E, H_flat, n, k, L, thres, even, d_NG_total, blocks_in_batch);
            CUDA_CHECK(cudaDeviceSynchronize());

            batch_NG += *d_NG_total;

            if (proper_early_termination_cuda(d_L_APP, G, n, k, blocks_in_batch)) break;

            // Slices processing
            batch_iterations += 0.5;
            *d_NG_total = 0;

            for(int idx = 0; idx < blocks_in_batch * codeword_block_size; idx++) {
                d_input[idx] = d_L_channel[idx] + d_L_A[idx];
            }

            dim3 grid3(n, 1, blocks_in_batch);
            dim3 block3(n, 1, 1);
            improved_slices_kernel<<<grid3, block3>>>(
                d_input, d_L_APP, d_L_E, H_flat, n, k, L, thres, even, d_NG_total, blocks_in_batch);
            CUDA_CHECK(cudaDeviceSynchronize());

            batch_NG += *d_NG_total;

            if (proper_early_termination_cuda(d_L_APP, G, n, k, blocks_in_batch)) break;
        }

        total_NG += batch_NG;
        total_iterations += batch_iterations;
        total_blocks += blocks_in_batch;

        // Output only the expected number of message blocks
        batch_hard_decision(d_L_APP, h_bit_batch, blocks_in_batch, message_block_size);

        unsigned char byte_out = 0;
        int bit_count_out = 0;

        for (int block_id = 0; block_id < blocks_in_batch; block_id++) {
            int offset = block_id * message_block_size;
            for (int i = 0; i < message_block_size; i++) {
                byte_out = (byte_out << 1) | h_bit_batch[offset + i];
                bit_count_out++;
                if (bit_count_out == 8) {
                    fwrite(&byte_out, 1, 1, fout);
                    byte_out = 0;
                    bit_count_out = 0;
                }
            }
        }

        if (bit_count_out > 0) {
            byte_out <<= (8 - bit_count_out);
            fwrite(&byte_out, 1, 1, fout);
            bit_count_out = 0;
        }
    }

    // Handle exact file size - truncate if necessary
    long output_size = ftell(fout);
    long expected_output_size = (total_expected_blocks * message_block_size + 7) / 8; // Convert bits to bytes

    if (output_size > expected_output_size) {
        printf("Truncating output file from %ld to %ld bytes\n", output_size, expected_output_size);
        fclose(fout);
        // Reopen and truncate
        fout = fopen(output_filename, "r+b");
        if (fout) {
            fseek(fout, expected_output_size, SEEK_SET);
            // Truncate file (platform specific)
            #ifdef _WIN32
                _chsize_s(_fileno(fout), expected_output_size);
            #else
                if (ftruncate(fileno(fout), expected_output_size) != 0) {
                    perror("ftruncate failed");
                }
            #endif
            fclose(fout);
        }
    } else {
        fclose(fout);
    }

    clock_t end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Improved CUDA Decoding complete. %d block(s) decoded.\n", total_blocks);
    printf("Average iterations per block: %.2f\n", total_iterations / total_blocks);
    printf("Average NG per block: %.2f\n", (double)total_NG / total_blocks);
    printf("Average NG per info bit: %.2f\n", (double)total_NG / (total_blocks * k * k * k));
    printf("Total time: %.3f seconds\n", cpu_time_used);
    printf("Throughput: %.1f blocks/second\n", total_blocks / cpu_time_used);

    // Cleanup
    fclose(fin);

    CUDA_CHECK(cudaFree(d_L_channel));
    CUDA_CHECK(cudaFree(d_L_APP));
    CUDA_CHECK(cudaFree(d_L_E));
    CUDA_CHECK(cudaFree(d_L_A));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_NG_total));
    CUDA_CHECK(cudaFree(H_flat));

    free(h_llr_batch);
    free(h_bit_batch);

    for(int i = 0; i < k; i++) free(G[i]);
    for(int i = 0; i < (n-k); i++) free(H[i]);
    free(G);
    free(H);

    return 0;
}
                
