#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Inf 0x7fffffff

// --- Data Structures for 3D Tensor ---
typedef struct {
    double* data;
    int dim1, dim2, dim3;
} Tensor_d;

typedef struct {
    int* data;
    int dim1, dim2, dim3;
} Tensor_i;

// --- Forward Declarations ---
Tensor_d create_tensor_d(int d1, int d2, int d3);
Tensor_i create_tensor_i(int d1, int d2, int d3);
void free_tensor_d(Tensor_d t);
void free_tensor_i(Tensor_i t);
double get_tensor_d(Tensor_d t, int i, int j, int k);
int get_tensor_i(Tensor_i t, int i, int j, int k);
void set_tensor_d(Tensor_d t, int i, int j, int k, double val);
void set_tensor_i(Tensor_i t, int i, int j, int k, int val);
int** create_int_matrix(int rows, int cols);
void free_int_matrix(int** matrix, int rows);
int* koopman2matlab(const char* k_poly, int* poly_len);
void getGH_sys_CRC(int n, int k, int** G, int** H);
int early_termination(Tensor_d L_APP, int** G, int n, int k);
void hard_decision(double* llr, int* bits, int length);
void SOGRAND_bitSO(double* L_APP, double* L_E, int* N_guess, double* llr, int** H_matrix, int n, int k, int L, uint64_t Tmax, double thres, int even);
void sogrand_main_logic(double* chat_list, double* s_list, double* T_val, double* curL_val, double* pNL_val, double* APP_list, double* llr, uint8_t* H_flat, int n, int s, int IC, uint64_t L, uint64_t Tmax, double thres, int even);
uint8_t ParityCheck(uint8_t *c, uint8_t *H, uint64_t n, uint64_t s);
int32_t findMax(int32_t a, int32_t b);
void HardDec(uint8_t *c, double *llr, uint64_t n);
int parity(uint8_t array[], uint64_t n);
double prob_parity(int parity_cHD, double *absL, uint64_t n);
void AddTEP(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n);
double JacLog(double x);
void QuickSort (double *a, size_t *perm, uint64_t n);
double getPM_HD(double *absL, uint64_t n);
double getPM(uint8_t *TEP, double *absL, double PM_HD, uint64_t n);
double getLConf(double *pNL, double P_notGuess, uint64_t cur_L, double *score, uint64_t s, uint8_t even);
void mountain_build(int32_t *u, int32_t k, int32_t w, int32_t W1, int32_t n1);
void getAPP(uint64_t cur_L, double *score, double *APP);

// --- Main Application ---
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

    const int L = 3;
    const int Imax = 30;
    const uint64_t Tmax = UINT64_MAX;
    const double p_ET = 1e-5;
    const double thres = 1.0 - p_ET;
    
    // Initialize alpha array with 0.7 for all iterations (matching MATLAB)
    double alpha[100];
    for(int i = 0; i < 100; i++) alpha[i] = 0.7;

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("Decoding %s to %s using cubic code (n=%d, k=%d)...\n", input_filename, output_filename, n, k);

    double llr_buffer[codeword_block_size];
    int bit_buffer[message_block_size];
    unsigned char byte_out = 0;
    int bit_count_out = 0;
    
    int total_NG = 0;
    int total_NG_p = 0;
    double total_iterations = 0;
    int block_count = 0;

    while (fread(llr_buffer, sizeof(double), codeword_block_size, fin) == codeword_block_size) {
        block_count++;
        Tensor_d L_channel = create_tensor_d(n, n, n);
        memcpy(L_channel.data, llr_buffer, codeword_block_size * sizeof(double));

        Tensor_d L_APP = create_tensor_d(n, n, n);
        Tensor_d L_E = create_tensor_d(n, n, n);
        Tensor_d L_A = create_tensor_d(n, n, n);
        Tensor_d input = create_tensor_d(n, n, n);

        // Initialize L_E to zeros
        memset(L_E.data, 0, codeword_block_size * sizeof(double));

        double vec_in[n], vec_L_APP[n], vec_L_E_vec[n];
        double n_iter = 0;
        int NG = 0;
        int NG_p = 0;

        for (int iter = 1; iter <= Imax; iter++) {
            // Columns
            int NGmax = 0;
            n_iter += 0.5;
            
            // L_A = alpha(2*i-1) * L_E
            for(int idx = 0; idx < codeword_block_size; idx++) {
                L_A.data[idx] = alpha[2*iter-2] * L_E.data[idx];
            }
            
            // input = L_channel + L_A
            for(int idx = 0; idx < codeword_block_size; idx++) {
                input.data[idx] = L_channel.data[idx] + L_A.data[idx];
            }

            for (int slice = 0; slice < n; slice++) {
                for (int col = 0; col < n; col++) {
                    for(int row=0; row<n; ++row) vec_in[row] = get_tensor_d(input, row, col, slice);
                    int N_guess = 0;
                    SOGRAND_bitSO(vec_L_APP, vec_L_E_vec, &N_guess, vec_in, H, n, k, L, Tmax, thres, even);
                    NG += N_guess;
                    if (N_guess > NGmax) NGmax = N_guess;
                    for(int row=0; row<n; ++row) {
                        set_tensor_d(L_APP, row, col, slice, vec_L_APP[row]);
                        set_tensor_d(L_E, row, col, slice, vec_L_E_vec[row]);
                    }
                }
            }
            NG_p += NGmax;
            
            if (early_termination(L_APP, G, n, k)) break;

            // Rows
            NGmax = 0;
            n_iter += 0.5;
            
            // L_A = alpha(2*i) * L_E
            for(int idx = 0; idx < codeword_block_size; idx++) {
                L_A.data[idx] = alpha[2*iter-1] * L_E.data[idx];
            }
            
            // input = L_channel + L_A
            for(int idx = 0; idx < codeword_block_size; idx++) {
                input.data[idx] = L_channel.data[idx] + L_A.data[idx];
            }

            for (int slice = 0; slice < n; slice++) {
                for (int row = 0; row < n; row++) {
                    for(int col=0; col<n; ++col) vec_in[col] = get_tensor_d(input, row, col, slice);
                    int N_guess = 0;
                    SOGRAND_bitSO(vec_L_APP, vec_L_E_vec, &N_guess, vec_in, H, n, k, L, Tmax, thres, even);
                    NG += N_guess;
                    if (N_guess > NGmax) NGmax = N_guess;
                    for(int col=0; col<n; ++col) {
                        set_tensor_d(L_APP, row, col, slice, vec_L_APP[col]);
                        set_tensor_d(L_E, row, col, slice, vec_L_E_vec[col]);
                    }
                }
            }
            NG_p += NGmax;
            
            if (early_termination(L_APP, G, n, k)) break;

            // Slices
            NGmax = 0;
            n_iter += 0.5;
            
            // L_A = alpha(2*i) * L_E (using same alpha as previous dimension)
            for(int idx = 0; idx < codeword_block_size; idx++) {
                L_A.data[idx] = alpha[2*iter-1] * L_E.data[idx];
            }
            
            // input = L_channel + L_A
            for(int idx = 0; idx < codeword_block_size; idx++) {
                input.data[idx] = L_channel.data[idx] + L_A.data[idx];
            }

            for (int col = 0; col < n; col++) {
                for (int row = 0; row < n; row++) {
                    for(int slice_idx=0; slice_idx<n; ++slice_idx) vec_in[slice_idx] = get_tensor_d(input, row, col, slice_idx);
                    int N_guess = 0;
                    SOGRAND_bitSO(vec_L_APP, vec_L_E_vec, &N_guess, vec_in, H, n, k, L, Tmax, thres, even);
                    NG += N_guess;
                    if (N_guess > NGmax) NGmax = N_guess;
                    for(int slice_idx=0; slice_idx<n; ++slice_idx) {
                        set_tensor_d(L_APP, row, col, slice_idx, vec_L_APP[slice_idx]);
                        set_tensor_d(L_E, row, col, slice_idx, vec_L_E_vec[slice_idx]);
                    }
                }
            }
            NG_p += NGmax;
            
            if (early_termination(L_APP, G, n, k)) break;
        }

        total_NG += NG;
        total_NG_p += NG_p;
        total_iterations += n_iter;

        // Hard Decode the final L_APP to get the codeword
        hard_decision(L_APP.data, bit_buffer, message_block_size);

        // Convert bit_buffer to bytes and write to output file
        for (int i = 0; i < message_block_size; i++) {
            byte_out = (byte_out << 1) | bit_buffer[i];
            bit_count_out++;
            if (bit_count_out == 8) {
                fwrite(&byte_out, 1, 1, fout);
                byte_out = 0;
                bit_count_out = 0;
            }
        }

        free_tensor_d(L_channel);
        free_tensor_d(L_APP);
        free_tensor_d(L_E);
        free_tensor_d(L_A);
        free_tensor_d(input);
    }

    if (bit_count_out > 0) {
        byte_out <<= (8 - bit_count_out);
        fwrite(&byte_out, 1, 1, fout);
    }

    clock_t end_time = clock();
    double cpu_time_used = ((double) end_time) / CLOCKS_PER_SEC;

    printf("Decoding complete. %d block(s) decoded.\n", block_count);
    printf("Average iterations per block: %.2f\n", total_iterations / block_count);
    printf("Average NG per block: %.2f\n", (double)total_NG / block_count);
    printf("Average NG per info bit: %.2f\n", (double)total_NG / (block_count * k * k * k));
    printf("Average NG_p per block: %.2f\n", (double)total_NG_p / block_count);
    printf("Total CPU time: %.2f seconds\n", cpu_time_used);
    
    fclose(fin);
    fclose(fout);
    free_int_matrix(G, k);
    free_int_matrix(H, n - k);

    return 0;
}


int early_termination(Tensor_d L_APP, int** G, int n, int k) {
    // 1. Get the hard decision from the final LLRs
    Tensor_i c_HD = create_tensor_i(n, n, n);
    for(int i=0; i<n*n*n; ++i) {
        c_HD.data[i] = (L_APP.data[i] > 0) ? 0 : 1;
    }

    // 2. Perform a SYSTEMATIC re-encoding of the message part of c_HD
    Tensor_i c_test = create_tensor_i(n, n, n);
    int temp_vec_k[k];

    // 2a. Copy the systematic message part from the hard decision
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                set_tensor_i(c_test, i, j, l, get_tensor_i(c_HD, i, j, l));
            }
        }
    }

    // 2b. Encode rows (calculating parity for columns k through n-1)
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for(int j=0; j<k; ++j) temp_vec_k[j] = get_tensor_i(c_test, row, j, slice);
            for (int col = k; col < n; col++) {
                 int parity_val = 0;
                 for (int msg_bit_idx=0; msg_bit_idx<k; ++msg_bit_idx) {
                     parity_val += temp_vec_k[msg_bit_idx] * G[msg_bit_idx][col];
                 }
                 set_tensor_i(c_test, row, col, slice, parity_val % 2);
            }
        }
    }

    // 2c. Encode columns (calculating parity for rows k through n-1)
    for (int slice = 0; slice < k; slice++) {
        for (int col = 0; col < n; col++) {
            for(int j=0; j<k; ++j) temp_vec_k[j] = get_tensor_i(c_test, j, col, slice);
            for (int row = k; row < n; row++) {
                 int parity_val = 0;
                 for (int msg_bit_idx=0; msg_bit_idx<k; ++msg_bit_idx) {
                     parity_val += temp_vec_k[msg_bit_idx] * G[msg_bit_idx][row];
                 }
                 set_tensor_i(c_test, row, col, slice, parity_val % 2);
            }
        }
    }

    // 2d. Encode slices across all rows and columns
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for(int j=0; j<k; ++j) temp_vec_k[j] = get_tensor_i(c_test, row, col, j);
            for (int slice = k; slice < n; slice++) {
                 int parity_val = 0;
                 for (int msg_bit_idx=0; msg_bit_idx<k; ++msg_bit_idx) {
                     parity_val += temp_vec_k[msg_bit_idx] * G[msg_bit_idx][slice];
                 }
                 set_tensor_i(c_test, row, col, slice, parity_val % 2);
            }
        }
    }

    // 3. Check if re-encoded c_test matches c_HD
    int match = 1;
    for(int i=0; i<n*n*n; ++i) {
        if (c_test.data[i] != c_HD.data[i]) {
            match = 0;
            break;
        }
    }

    free_tensor_i(c_HD);
    free_tensor_i(c_test);
    return match;
}

void hard_decision(double* llr, int* bits, int length) {
    // Extract information bits from the cubic tensor
    // The info bits are in positions [0:k, 0:k, 0:k]
    int k = 10; // hardcoded for now, can be passed as parameter
    int n = 15;
    int bit_idx = 0;
    
    // Ensure we don't exceed the expected length
    int expected_length = k * k * k;
    if (length != expected_length) {
        fprintf(stderr, "Warning: hard_decision length mismatch\n");
    }
    
    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for (int col = 0; col < k; col++) {
                int tensor_idx = slice * n * n + row * n + col;
                bits[bit_idx++] = (llr[tensor_idx] > 0) ? 0 : 1;
            }
        }
    }
}

Tensor_i create_tensor_i(int d1, int d2, int d3) {
    Tensor_i t;
    t.dim1 = d1; t.dim2 = d2; t.dim3 = d3;
    t.data = (int*)calloc(d1 * d2 * d3, sizeof(int));
    return t;
}

Tensor_d create_tensor_d(int d1, int d2, int d3) {
    Tensor_d t;
    t.dim1 = d1; t.dim2 = d2; t.dim3 = d3;
    t.data = (double*)calloc(d1 * d2 * d3, sizeof(double));
    return t;
}

void free_tensor_i(Tensor_i t) { free(t.data); }
void free_tensor_d(Tensor_d t) { free(t.data); }

int get_tensor_i(Tensor_i t, int i, int j, int k) { return t.data[k*t.dim1*t.dim2 + j*t.dim1 + i]; }
void set_tensor_i(Tensor_i t, int i, int j, int k, int val) { t.data[k*t.dim1*t.dim2 + j*t.dim1 + i] = val; }
double get_tensor_d(Tensor_d t, int i, int j, int k) { return t.data[k*t.dim1*t.dim2 + j*t.dim1 + i]; }
void set_tensor_d(Tensor_d t, int i, int j, int k, double val) { t.data[k*t.dim1*t.dim2 + j*t.dim1 + i] = val; }

int** create_int_matrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for(int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
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

void SOGRAND_bitSO(double* L_APP, double* L_E, int* N_guess, double* llr, int** H_matrix, int n, int k, int L, uint64_t Tmax, double thres, int even) {
    double* chat_list = (double*)malloc(sizeof(double) * n * L);
    double* s_list = (double*)malloc(sizeof(double) * 4 * L);
    double T_val, curL_val, pNL_val;
    double* APP_list = (double*)malloc(sizeof(double) * L);

    uint8_t* H_flat = (uint8_t*)malloc(sizeof(uint8_t) * (n-k) * n);
    for (int i = 0; i < (n-k); i++) {
        for (int j = 0; j < n; j++) {
            H_flat[i*n + j] = (uint8_t)H_matrix[i][j];
        }
    }

    sogrand_main_logic(chat_list, s_list, &T_val, &curL_val, &pNL_val, APP_list, llr, H_flat, n, n - k, -1, L, Tmax, thres, even);

    int curL = (int)curL_val;
    if (curL == 0) {
        for(int i=0; i<n; ++i) L_APP[i] = llr[i];
        for(int i=0; i<n; ++i) L_E[i] = 0;
    } else {
        double* PM = (double*)malloc(sizeof(double) * curL);
        for(int i=0; i<curL; ++i) PM[i] = s_list[4*i + 1];
        double p_notinlist = fmax(pNL_val, 1e-9);

        double pp1[n], pp0[n];
        for(int i=0; i<n; ++i) {
            pp1[i] = 1.0 / (1.0 + exp(llr[i]));
            pp0[i] = 1.0 - pp1[i];
            pp1[i] = fmax(pp1[i], 1e-9); pp1[i] = fmin(pp1[i], 1.0 - 1e-9);
            pp0[i] = fmax(pp0[i], 1e-9); pp0[i] = fmin(pp0[i], 1.0 - 1e-9);
        }

        double p[curL];
        for(int i=0; i<curL; ++i) p[i] = exp(-PM[i]);

        double p1[n], p0[n];
        memset(p1, 0, n * sizeof(double));
        memset(p0, 0, n * sizeof(double));

        for(int i=0; i<n; ++i) {
            for(int j=0; j<curL; ++j) {
                p1[i] += chat_list[i + j*n] * p[j];
                p0[i] += (1.0 - chat_list[i + j*n]) * p[j];
            }
        }

        for(int i=0; i<n; ++i) {
            p0[i] += p_notinlist * pp0[i];
            p1[i] += p_notinlist * pp1[i];
        }

        for(int i=0; i<n; ++i) {
            L_APP[i] = log(fmax(p0[i], 1e-30)) - log(fmax(p1[i], 1e-30));
            L_E[i] = L_APP[i] - llr[i];
        }
        free(PM);
    }

    if (N_guess) *N_guess = (int)T_val;

    free(chat_list);
    free(s_list);
    free(APP_list);
    free(H_flat);
}

void sogrand_main_logic(double* chat_list, double* s_list, double* T_val, double* curL_val, double* pNL_val, double* APP_list, double* llr, uint8_t* H_flat, int n, int s, int IC, uint64_t L, uint64_t Tmax, double thres, int even){
    size_t *perm = calloc(n, sizeof(size_t));
    uint8_t *cHD = calloc(n, sizeof(uint8_t));
    uint8_t *TEP = calloc(n, sizeof(uint8_t));
    uint8_t *c   = calloc(n, sizeof(uint8_t));
    double *absL = calloc(n, sizeof(double));
    int32_t *u   = calloc(n, sizeof(int32_t));
    int32_t *d   = calloc(n, sizeof(int32_t));
    int32_t *D   = calloc(n, sizeof(int32_t));

    for(size_t i = 0; i < n; i++) perm[i] = i;
    for(size_t i = 0; i < 4*L; i++) s_list[i] = 0; // Initialize s_list
    for(size_t i = 0; i < L; i++) APP_list[i] = 0; // Initialize APP_list

    uint64_t cur_L = 0;
    HardDec(cHD, llr, n);
    uint8_t parity_cHD = parity(cHD,n);
    pNL_val[0] = 0.0;

    if (Tmax==0) Tmax=Inf;

    for (size_t i = 0; i < n; i++) {
        TEP[i] = 0;
        absL[i] = fabs(llr[i]);
    }

    double P_notGuess = 1.0;
    if (even==1) P_notGuess = prob_parity(parity_cHD,absL,n);

    double PM_HD = getPM_HD(absL, n);
    QuickSort(absL, perm, n);

    if (IC < 0){
        if (round((double)n/2) > 1) {
            double beta = (absL[(uint64_t)round((double)n/2) - 1] - absL[0]) / (round((double)n/2) - 1);
            IC = (beta > 1e-9) ? findMax((int32_t)round(absL[0]/beta - 1), 0) : 0;
        } else {
            IC = 0;
        }
    }

    AddTEP(c, cHD, TEP, perm, n);
    T_val[0] = 1;
    if (parity_cHD==0 || even==0) pNL_val[0] = exp(-getPM(TEP, absL, PM_HD, n));

    if (ParityCheck(c, H_flat, n, s) == 1){
        APP_list[0] = getPM(TEP, absL, PM_HD, n);
        for(size_t i=0; i<n; i++) chat_list[i] = c[i];
        s_list[1] = 1; // T
        s_list[2] = T_val[0]; // N
        s_list[3] = getLConf(pNL_val, P_notGuess, cur_L, s_list, s, even); // Conf
        cur_L++;
        if ((s_list[3] > thres) || (cur_L == L)){
            getAPP(cur_L, s_list, APP_list);
            curL_val[0] = cur_L;
            free(perm); free(cHD); free(TEP); free(c); free(absL); free(u); free(D); free(d);
            return;
        }
    }

    int32_t wt = IC + 1;
    while ((cur_L < L) && (T_val[0] < Tmax)) {
        int32_t w = 1;
        double temp_sqrt = pow(1+2*((double)n+(double)IC), 2.0) - 8*wt;
        if (temp_sqrt >= 0) {
            w = findMax(1, (int32_t)ceil((1+2*((double)n+(double)IC) - sqrt(temp_sqrt))/2.0));
        } else {
             wt++; continue;
        }

        if (even==1 && (w%2 != parity_cHD)) w++;

        while (w <= n) {
            int32_t W = wt - IC*w;
            if (W < w*(w+1)/2) break;

            int32_t W1 = W - w*(w+1)/2;
            int32_t n1 = n - w;
            for (size_t i = 0; i < w; i++) u[i] = 0;

            mountain_build(u,0,w,W1,n1);

            int mountain_iter_guard = 0;
            do {
                for (size_t i = 0; i < n; i++) TEP[i] = 0;
                for (size_t i = 0; i < w; i++) TEP[i+u[i]] = 1;
                AddTEP(c, cHD, TEP, perm, n);
                T_val[0]++;
                if (parity_cHD==0 || even==0) pNL_val[0] = exp(-getPM(TEP, absL, PM_HD, n));

                if (ParityCheck(c, H_flat, n, s) == 1){
                    APP_list[cur_L] = wt;
                    for(size_t i=0; i<n; i++) chat_list[cur_L*n + i] = c[i];
                    s_list[4*cur_L+1] = wt; // T
                    s_list[4*cur_L+2] = T_val[0]; // N
                    s_list[4*cur_L+3] = getLConf(pNL_val, P_notGuess, cur_L, s_list, s, even); // Conf
                    cur_L++;
                    if ((s_list[4*(cur_L-1)+3] > thres) || (cur_L == L)){
                        getAPP(cur_L, s_list, APP_list);
                        curL_val[0] = cur_L;
                        free(perm); free(cHD); free(TEP); free(c); free(absL); free(u); free(D); free(d);
                        return;
                    }
                }

                for (size_t i = 0; i < w - 1; i++) d[i] = u[i+1] - u[i];
                d[w-1] = 0;
                D[w-1] = d[w-1];
                for (int i = w - 2; i >= 0; i--) D[i] = D[i+1] + d[i];

                if (D[0] < 2) break;

                int32_t k_mt = 0;
                for (int i = w-1; i > 0; i--) {
                    if (D[i] >= 2) { k_mt = i; break; }
                }
                u[k_mt]++;
                mountain_build(u,k_mt,w,W1,n1);

            } while (++mountain_iter_guard < 100000); // Safety break

            w++;
            if (even==1 && (w%2 != parity_cHD)) w++;
        }
        wt++;
    }

    curL_val[0] = cur_L;
    getAPP(cur_L, s_list, APP_list);
    free(perm); free(cHD); free(TEP); free(c); free(absL); free(u); free(D); free(d);
}

uint8_t ParityCheck(uint8_t *c, uint8_t *H, uint64_t n, uint64_t s) {
    for (size_t j = 0; j < s; j++){
        uint8_t syndrome = 0;
        for (size_t i = 0; i < n; i++) syndrome ^= (c[i] * H[j*n + i]);
        if (syndrome == 1) return 0;
    }
    return 1;
}

int32_t findMax(int32_t a, int32_t b) { return !(b > a) ? a : b; }
void HardDec(uint8_t *c, double *llr, uint64_t n) { for (size_t i = 0; i < n; i++) c[i] = (llr[i] > 0.0) ? 0 : 1; }
int parity(uint8_t array[], uint64_t n) { int sum = 0; for (uint64_t i = 0; i < n; i++) sum += array[i]; return sum % 2; }
double prob_parity(int parity_cHD, double *absL, uint64_t n) { double p_e = 1.0; for (uint64_t i = 0; i < n; i++) { p_e *= (1.0 - 2.0 * exp(-absL[i]) / (1.0 + exp(-absL[i]))); } p_e = 0.5 * (1.0 + p_e); return (parity_cHD == 0) ? p_e : 1.0 - p_e; }
void AddTEP(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n) { for (size_t i = 0; i < n; i++) c[perm[i]] = cHD[perm[i]] ^ TEP[i]; }
double JacLog(double x) { if (x > 30) return x; if (x < -30) return 0.0; return log(1.0 + exp(x)); }
void QuickSort(double *a, size_t *perm, uint64_t n) { if (n < 2) return; double p = a[n / 2]; uint64_t i = 0, j = n - 1; while (i <= j) { while (a[i] < p) i++; while (a[j] > p) j--; if (i <= j) { double t = a[i]; a[i] = a[j]; a[j] = t; size_t tt = perm[i]; perm[i] = perm[j]; perm[j] = tt; i++; j--; } } if (j > 0) QuickSort(a, perm, j + 1); if (i < n) QuickSort(a + i, perm + i, n - i); }
double getPM_HD(double *absL, uint64_t n) { double pm = 0; for(size_t i=0; i<n; i++) pm += JacLog(-absL[i]); return pm; }
double getPM(uint8_t *TEP, double *absL, double PM_HD, uint64_t n) { double pm = PM_HD; for(size_t i = 0; i < n; i++) { if (TEP[i] == 1) pm += (JacLog(absL[i]) - JacLog(-absL[i])); } return pm; }
double getLConf(double *pNL, double P_notGuess, uint64_t cur_L, double *score, uint64_t s, uint8_t even) {
    double P_pos = 0.0;
    for(size_t i = 0; i <= cur_L; i++) P_pos += exp(-score[4*i+1]);
    if(even==1) s--;
    double P_neg = pow(2.0, -(double)s) * P_notGuess;
    pNL[0] = P_neg;
    return (P_pos + P_neg > 1e-9) ? (P_pos / (P_pos + P_neg)) : 1.0;
}
void mountain_build(int32_t *u, int32_t k, int32_t w, int32_t W1, int32_t n1){
    for(size_t i = k + 1; i < w; i++)
        u[i] = u[k];
    uint64_t W2 = W1;
    for(size_t i = 0; i < w; i++)
        W2 -= u[i];
    uint64_t q = (uint64_t)floor( (double)W2 / (double)(n1 - u[k]) );
    uint64_t r = W2 - q*(n1 - u[k]);
    if (q != 0){
        for(size_t i = w-q; i < w; i++)
            u[i] = n1;
    }
    if (w > q)
        u[w-q-1] = u[w-q-1] + r;
}
void getAPP(uint64_t cur_L, double *score, double *APP) { if (cur_L == 0) return; double P_pos = 0.0; for(size_t i=0; i<cur_L; i++) P_pos += exp(-score[4*i+1]); if (P_pos < 1e-30) return; double den = score[4*(cur_L-1)+3] / P_pos; for(size_t i=0; i<cur_L; i++) APP[i] = exp(-score[4*i+1]) * den; }
