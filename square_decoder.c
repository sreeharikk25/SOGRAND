#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define Inf 0x7fffffff

// --- Forward Declarations ---
double** create_double_matrix(int rows, int cols);
int** create_int_matrix(int rows, int cols);
void free_double_matrix(double** matrix, int rows);
void free_int_matrix(int** matrix, int rows);
int* koopman2matlab(const char* k_poly, int* poly_len);
void getGH_sys_CRC(int n, int k, int** G, int** H);
int early_termination(double** L_APP, int** G, int n, int k);
void hard_decision_2d(double** L_APP, int* bits, int n, int k);
void SOGRAND_bitSO(double* L_APP, double* L_E, int* N_guess, double* llr, int** H_matrix, int n, int k, int L, uint64_t Tmax, double thres, int even);
void sogrand_main_logic(double *chat, double *score, double *T, double *curL, double *pNL, double *APP, double *llr, uint8_t *H, uint64_t n, uint64_t s, int32_t IC, uint64_t L, uint64_t Tmax, double thres, uint8_t even);

// SOGRAND C helper functions
uint8_t ParityCheck(uint8_t *c, uint8_t *H, uint64_t n, uint64_t s);
int32_t findMax(int32_t a, int32_t b);
void HardDec(uint8_t *c, double *llr, uint64_t n);
int parity(uint8_t array[], uint64_t n);
double prob_parity(int parity_cHD, double *absL, uint64_t n);
void AddTEP(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n);
double JacLog(double x);
void QuickSort(double *a, size_t *perm, uint64_t n);
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

    const int n = 31;
    const int k = 25;
    const int codeword_block_size = n * n;
    const int message_block_size = k * k;

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

    const int L = 4;
    const int Imax = 20;
    const uint64_t Tmax = UINT64_MAX;
    const double p_ET = 1e-5;
    const double thres = 1.0 - p_ET;
    
    // Initialize alpha array with 0.5 for all iterations (matching MATLAB)
    double alpha[50];
    for(int i = 0; i < 50; i++) alpha[i] = 0.5;

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    printf("Decoding %s to %s using square/product code (n=%d, k=%d)...\n", input_filename, output_filename, n, k);

    double llr_buffer[codeword_block_size];
    int bit_buffer[message_block_size];
    unsigned char byte_out = 0;
    int bit_count_out = 0;
    
    int total_NG = 0;
    int total_NG_p = 0;
    double total_iterations = 0;
    int block_count = 0;

    clock_t start_time = clock();

    while (fread(llr_buffer, sizeof(double), codeword_block_size, fin) == codeword_block_size) {
        block_count++;
        double** L_channel = create_double_matrix(n, n);
        for(int idx = 0; idx < codeword_block_size; idx++) {
            L_channel[idx / n][idx % n] = llr_buffer[idx];
        }

        double** L_APP = create_double_matrix(n, n);
        double** L_E = create_double_matrix(n, n);
        double** L_A = create_double_matrix(n, n);
        
        // Initialize L_E to zeros
        for(int r = 0; r < n; r++) {
            for(int c = 0; c < n; c++) {
                L_E[r][c] = 0.0;
            }
        }
        
        double n_iter = 0;
        int NG = 0;
        int NG_p = 0;

        for (int iter = 1; iter <= Imax; iter++) {
            // Rows
            int NGmax = 0;
            n_iter += 0.5;
            
            double** input_matrix = create_double_matrix(n, n);
            for(int r = 0; r < n; r++) {
                for(int c = 0; c < n; c++) {
                    input_matrix[r][c] = L_channel[r][c] + alpha[2*iter-2] * L_E[r][c];
                }
            }
            
            for (int row = 0; row < n; row++) {
                int N_guess = 0;
                SOGRAND_bitSO(&L_APP[row][0], &L_E[row][0], &N_guess, &input_matrix[row][0], H, n, k, L, Tmax, thres, even);
                NG += N_guess;
                if (N_guess > NGmax) NGmax = N_guess;
            }
            NG_p += NGmax;
            free_double_matrix(input_matrix, n);
            
            // Early termination check
            if (early_termination(L_APP, G, n, k)) break;
            
            // Columns
            NGmax = 0;
            n_iter += 0.5;
            
            input_matrix = create_double_matrix(n, n);
            for(int r = 0; r < n; r++) {
                for(int c = 0; c < n; c++) {
                    input_matrix[r][c] = L_channel[r][c] + alpha[2*iter-1] * L_E[r][c];
                }
            }
            
            for (int col = 0; col < n; col++) {
                double input_col_vec[n], L_APP_col_vec[n], L_E_col_vec[n];
                for(int r = 0; r < n; r++) input_col_vec[r] = input_matrix[r][col];
                int N_guess = 0;
                SOGRAND_bitSO(L_APP_col_vec, L_E_col_vec, &N_guess, input_col_vec, H, n, k, L, Tmax, thres, even);
                NG += N_guess;
                if (N_guess > NGmax) NGmax = N_guess;
                for(int r = 0; r < n; r++) { 
                    L_APP[r][col] = L_APP_col_vec[r]; 
                    L_E[r][col] = L_E_col_vec[r]; 
                }
            }
            NG_p += NGmax;
            free_double_matrix(input_matrix, n);
            
            // Early termination check
            if (early_termination(L_APP, G, n, k)) break;
        }

        total_NG += NG;
        total_NG_p += NG_p;
        total_iterations += n_iter;

        // Hard Decode the final L_APP to get the message bits
        hard_decision_2d(L_APP, bit_buffer, n, k);

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

        free_double_matrix(L_channel, n);
        free_double_matrix(L_APP, n);
        free_double_matrix(L_E, n);
        free_double_matrix(L_A, n);
    }

    if (bit_count_out > 0) {
        byte_out <<= (8 - bit_count_out);
        fwrite(&byte_out, 1, 1, fout);
    }

    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Decoding complete. %d block(s) decoded.\n", block_count);
    printf("Average iterations per block: %.2f\n", total_iterations / block_count);
    printf("Average NG per block: %.2f\n", (double)total_NG / block_count);
    printf("Average NG per info bit: %.2f\n", (double)total_NG / (block_count * k * k));
    printf("Average NG_p per block: %.2f\n", (double)total_NG_p / block_count);
    printf("Total CPU time: %.2f seconds\n", cpu_time_used);

    fclose(fin);
    fclose(fout);
    free_int_matrix(G, k);
    free_int_matrix(H, n - k);

    return 0;
}

// --- Helper Functions ---

int early_termination(double** L_APP, int** G, int n, int k) {
    // 1. Get the hard decision from the final LLRs
    int** c_HD = create_int_matrix(n, n);
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++) {
            c_HD[r][c] = (L_APP[r][c] > 0) ? 0 : 1;
        }
    }

    // 2. Re-encode systematically
    int** c_test = create_int_matrix(n, n);
    
    // Copy the systematic part
    for(int r = 0; r < k; r++) {
        for(int c = 0; c < k; c++) {
            c_test[r][c] = c_HD[r][c];
        }
    }
    
    // Encode rows
    for (int row = 0; row < k; row++) {
        for (int col = k; col < n; col++) {
            int parity = 0;
            for (int j = 0; j < k; j++) {
                parity += c_test[row][j] * G[j][col];
            }
            c_test[row][col] = parity % 2;
        }
    }
    
    // Encode columns
    for (int col = 0; col < n; col++) {
        for (int row = k; row < n; row++) {
            int parity = 0;
            for (int j = 0; j < k; j++) {
                parity += c_test[j][col] * G[j][row];
            }
            c_test[row][col] = parity % 2;
        }
    }
    
    // 3. Check if re-encoded matches hard decision
    int match = 1;
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++) {
            if (c_test[r][c] != c_HD[r][c]) {
                match = 0;
                break;
            }
        }
        if (!match) break;
    }
    
    free_int_matrix(c_HD, n);
    free_int_matrix(c_test, n);
    return match;
}

void hard_decision_2d(double** L_APP, int* bits, int n, int k) {
    // Extract information bits from positions [0:k, 0:k]
    int bit_idx = 0;
    for(int r = 0; r < k; r++) {
        for(int c = 0; c < k; c++) {
            bits[bit_idx++] = (L_APP[r][c] > 0) ? 0 : 1;
        }
    }
}

// --- Existing Functions ---

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

int* koopman2matlab(const char* k_poly, int* poly_len) {
    long long dec_val = strtoll(k_poly, NULL, 16);
    int len = (dec_val > 0) ? floor(log2(dec_val)) + 1 : 1;
    *poly_len = len + 1;
    int* poly = (int*)malloc(sizeof(int) * (*poly_len));

    for (int i = len - 1; i >= 0; i--) {
        poly[i] = (dec_val >> (len - 1 - i)) & 1;
    }
    poly[len] = 1;

    return poly;
}

void sogrand_main_logic(double *chat, double *score, double *T, double *curL, double *pNL, double *APP, double *llr, uint8_t *H, uint64_t n, uint64_t s, int32_t IC, uint64_t L, uint64_t Tmax, double thres, uint8_t even){
    /* Create vectors */
    size_t *perm = calloc(n, sizeof(size_t));
    uint8_t *cHD = calloc(n, sizeof(uint8_t));
    uint8_t parity_cHD;
    uint8_t *TEP = calloc(n, sizeof(uint8_t));
    uint8_t *c   = calloc(n, sizeof(uint8_t));
    double *absL = calloc(n, sizeof(double));
    int32_t *u   = calloc(n, sizeof(int32_t));
    int32_t *d   = calloc(n, sizeof(int32_t));
    int32_t *D   = calloc(n, sizeof(int32_t));
    for(size_t i = 0; i < n; i++)
        perm[i] = i;
    for(size_t i = 0; i < 4*L; i++)
        score[i] = 0;
    for(size_t i = 0; i < L; i++)
        APP[i] = 0;
    uint64_t cur_L = 0;
    /* Initialize */
    HardDec(cHD, llr, n);
    parity_cHD = parity(cHD,n);
    pNL[0] = 0.0;

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
    T[0] = 1;
    if (parity_cHD==0 || even==0) P_notGuess -= exp(-getPM(TEP, absL, PM_HD, n));

    if (ParityCheck(c, H, n, s) == 1){
        score[1] = getPM(TEP, absL, PM_HD, n);
        for(size_t i=0; i<n; i++) chat[i] = c[i];
        score[2] = 1;
        score[3] = getLConf(pNL, P_notGuess, cur_L, score, s, even);
        cur_L++;
        if ((score[3] > thres) || (cur_L == L)){
            getAPP(cur_L, score, APP);
            curL[0] = cur_L;
            free(perm); free(cHD); free(TEP); free(c); free(absL); free(u); free(D); free(d);
            return;
        }
    }

    int32_t wt = IC + 1;
    while ((cur_L < L) && (T[0] < Tmax)) {
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
                T[0]++;
                P_notGuess -= exp(-getPM(TEP, absL, PM_HD, n));

                if (ParityCheck(c, H, n, s) == 1){
                    score[4*cur_L] = wt;
                    score[4*cur_L+1] = getPM(TEP, absL, PM_HD, n);
                    for(size_t i=0; i<n; i++) chat[cur_L*n + i] = c[i];
                    score[4*cur_L+2] = T[0];
                    score[4*cur_L+3] = getLConf(pNL, P_notGuess, cur_L, score, s, even);
                    cur_L++;
                    if ((score[4*(cur_L-1)+3] > thres) || (cur_L == L)){
                        getAPP(cur_L, score, APP);
                        curL[0] = cur_L;
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

            } while (++mountain_iter_guard < n*n);

            w++;
            if (even==1 && (w%2 != parity_cHD)) w++;
        }
        wt++;
    }

    curL[0] = cur_L;
    getAPP(cur_L, score, APP);
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

void HardDec(uint8_t *c, double *llr, uint64_t n) {
    for (size_t i = 0; i < n; i++) c[i] = (llr[i] > 0.0) ? 0 : 1;
}

int parity(uint8_t array[], uint64_t n) {
    int sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += array[i];
    return sum % 2;
}

double prob_parity(int parity_cHD, double *absL, uint64_t n) {
    double prob_even = 1.0;
    for (uint64_t i = 0; i < n; i++) {
        prob_even *= (1.0 - 2.0 * exp(-absL[i]) / (1.0 + exp(-absL[i])));
    }
    prob_even = 0.5 * (1.0 + prob_even);
    return (parity_cHD == 0) ? prob_even : 1.0 - prob_even;
}

void AddTEP(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n) {
    for (size_t i = 0; i < n; i++) c[perm[i]] = cHD[perm[i]] ^ TEP[i];
}

double JacLog(double x) {
    if (x > 50) return x;
    if (x < -50) return 0.0;
    return log(1.0 + exp(x));
}

void QuickSort(double *a, size_t *perm, uint64_t n) {
    if (n < 2) return;
    double p = a[n / 2];
    uint64_t i = 0, j = n - 1;
    while (i <= j) {
        while (a[i] < p) i++;
        while (a[j] > p) j--;
        if (i <= j) {
            double t = a[i]; a[i] = a[j]; a[j] = t;
            size_t tt = perm[i]; perm[i] = perm[j]; perm[j] = tt;
            i++; j--;
        }
    }
    if (j > 0) QuickSort(a, perm, j + 1);
    if (i < n) QuickSort(a + i, perm + i, n - i);
}

double getPM_HD(double *absL, uint64_t n) {
    double pm = 0;
    for(size_t i=0; i<n; i++) pm += JacLog(-absL[i]);
    return pm;
}

double getPM(uint8_t *TEP, double *absL, double PM_HD, uint64_t n) {
    double pm = PM_HD;
    for(size_t i = 0; i < n; i++) {
        if (TEP[i] == 1) pm += (JacLog(absL[i]) - JacLog(-absL[i]));
    }
    return pm;
}

double getLConf(double *pNL, double P_notGuess, uint64_t cur_L, double *score, uint64_t s, uint8_t even) {
    double P_positive = 0.0;
    for(size_t i = 0; i < cur_L; i++) P_positive += exp(-score[4*i+1]);

    if(even==1) s--;

    double P_negative = pow(2.0, -(double)s) * P_notGuess;
    pNL[0] = P_negative;
    return (P_positive + P_negative > 1e-9) ? (P_positive / (P_positive + P_negative)) : 1.0;
}

void mountain_build(int32_t *u, int32_t k, int32_t w, int32_t W1, int32_t n1){
    for(size_t i = k + 1; i < w; i++) u[i] = u[k];
    int64_t W2 = W1;
    for(size_t i = 0; i < w; i++) W2 -= u[i];

    if (n1 - u[k] == 0) return;

    uint64_t q = (W2 >= 0) ? (uint64_t)floor((double)W2 / (n1 - u[k])) : 0;
    uint64_t r = W2 - q * (n1 - u[k]);
    if (q > 0 && w > q) {
        for(size_t i = w - q; i < w; i++) u[i] = n1;
    }
    if (w > q) u[w - q - 1] += r;
}

void getAPP(uint64_t cur_L, double *score, double *APP) {
    if (cur_L == 0) return;
    double P_positive = 0.0;
    for(size_t i=0; i<cur_L; i++) P_positive += exp(-score[4*i+1]);
    if (P_positive < 1e-30) return;
    double denominator = score[4*(cur_L-1)+3] / P_positive;
    for(size_t i=0; i<cur_L; i++) APP[i] = exp(-score[4*i+1]) * denominator;
}

void print_matrix_int(const char* title, int** matrix, int rows, int cols) {
    printf("%s\n", title);
    if (rows > 10 || cols > 10) {
        printf("  (Matrix too large to print)\n\n");
        return;
    }
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) printf(" %d", matrix[i][j]);
        printf(" ]\n");
    }
    printf("\n");
}

int** create_int_matrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) matrix[i] = (int*)calloc(cols, sizeof(int));
    return matrix;
}
double** create_double_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) matrix[i] = (double*)calloc(cols, sizeof(double));
    return matrix;
}
void free_int_matrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) free(matrix[i]);
    free(matrix);
}
void free_double_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) free(matrix[i]);
    free(matrix);
}

double normal_dist_rand() {
    double u1 = (double)rand() / (RAND_MAX + 1.0);
    double u2 = (double)rand() / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
