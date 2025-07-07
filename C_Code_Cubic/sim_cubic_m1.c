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

typedef struct {
    int* data;
    int dim1, dim2, dim3;
} Tensor_i;

typedef struct {
    double* data;
    int dim1, dim2, dim3;
} Tensor_d;



Tensor_i create_tensor_i(int d1, int d2, int d3);
Tensor_d create_tensor_d(int d1, int d2, int d3);
void free_tensor_i(Tensor_i t);
void free_tensor_d(Tensor_d t);

int get_tensor_i(Tensor_i t, int i, int j, int k);
void set_tensor_i(Tensor_i t, int i, int j, int k, int val);
double get_tensor_d(Tensor_d t, int i, int j, int k);
void set_tensor_d(Tensor_d t, int i, int j, int k, double val);
double normal_dist_rand();


int* koopman2matlab(const char* k_poly, int* poly_len);
void getGH_sys_CRC(int n, int k, int** G, int** H);
int early_termination(Tensor_d L_APP, int** G, int n, int k);


void SOGRAND_bitSO(double* L_APP, double* L_E, int* N_guess, double* llr, int** H_matrix, int n, int k, int L, uint64_t Tmax, double thres, int even);
void sogrand_main_logic(double *chat, double *score, double *T, double *curL, double *pNL, double *APP, double *llr, uint8_t *H, uint64_t n, uint64_t s, int32_t IC, uint64_t L, uint64_t Tmax, double thres, uint8_t even);
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

int main() {

    const double EbN0dB[] = {1.0, 1.25, 1.5, 1.75};
    const int numSNR = sizeof(EbN0dB) / sizeof(double);
    const int NoErrors = 20;
    const int maxIt = 1000000;
    const int minIt = 100;

    const int n = 15;
    const int k = 10;

    int** G = (int**)malloc(k * sizeof(int*));
    for(int i=0; i<k; ++i) G[i] = (int*)malloc(n * sizeof(int));
    int** H = (int**)malloc((n-k) * sizeof(int*));
    for(int i=0; i<n-k; ++i) H[i] = (int*)malloc(n * sizeof(int));
    getGH_sys_CRC(n, k, G, H);

    const int L = 3;
    const int Imax = 30;
    const uint64_t Tmax = 0;
    const double p_ET = 1e-5;
    const double thres = 1.0 - p_ET;
    double alpha[100];
    for(int i=0; i<100; ++i) alpha[i] = 0.7;

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

    const double R = pow((double)k / n, 3.0);
    double EsN0dB[numSNR];
    for(int i=0; i<numSNR; ++i) EsN0dB[i] = EbN0dB[i] + 10 * log10(2 * R);

    printf("=========================================================\n");
    printf("     Cubic Code Simulation with SOGRAND Decoder in C     \n");
    printf("=========================================================\n");

    srand(time(NULL));

    for (int sp = 0; sp < numSNR; sp++) {
        long long BlockError = 0;
        long long BitError = 0;
        double n_iter_total = 0;
        long long ntx = 0;
        double sigma = 1.0 / sqrt(pow(10, EsN0dB[sp] / 10.0));

        printf("\n--- Eb/N0 = %.2f dB (sigma = %f) ---\n", EbN0dB[sp], sigma);

        while ((BlockError < NoErrors && ntx < maxIt) || ntx < minIt) {
            ntx++;

            Tensor_i u = create_tensor_i(k, k, k);
            for(int i=0; i<k*k*k; ++i) u.data[i] = rand() % 2;

            Tensor_i c = create_tensor_i(n, n, n);
            int temp_vec[n];
            int temp_vec_k[k];

            for (int slice = 0; slice < k; slice++) {
                for (int row = 0; row < k; row++) {
                    for(int i=0; i<k; ++i) temp_vec_k[i] = get_tensor_i(u, row, i, slice);
                    memset(temp_vec, 0, n * sizeof(int));
                    for (int col = 0; col < n; col++) {
                        for (int g_col = 0; g_col < k; g_col++) {
                            temp_vec[col] += temp_vec_k[g_col] * G[g_col][col];
                        }
                    }
                    for (int col = 0; col < n; col++) {
                        set_tensor_i(c, row, col, slice, temp_vec[col] % 2);
                    }
                }
                for (int col = 0; col < n; col++) {
                    for(int i=0; i<k; ++i) temp_vec_k[i] = get_tensor_i(c, i, col, slice);
                    memset(temp_vec, 0, n * sizeof(int));
                    for (int row = 0; row < n; row++) {
                        for (int g_col = 0; g_col < k; g_col++) {
                            temp_vec[row] += temp_vec_k[g_col] * G[g_col][row];
                        }
                    }
                    for (int row = 0; row < n; row++) {
                        set_tensor_i(c, row, col, slice, temp_vec[row] % 2);
                    }
                }
            }
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) {
                     for(int i=0; i<k; ++i) temp_vec_k[i] = get_tensor_i(c, row, col, i);
                     memset(temp_vec, 0, n * sizeof(int));
                     for (int slice = 0; slice < n; slice++) {
                         for (int g_col = 0; g_col < k; g_col++) {
                             temp_vec[slice] += temp_vec_k[g_col] * G[g_col][slice];
                         }
                     }
                     for (int slice = 0; slice < n; slice++) {
                        set_tensor_i(c, row, col, slice, temp_vec[slice] % 2);
                    }
                }
            }

            Tensor_d L_channel = create_tensor_d(n, n, n);
            for(int i=0; i<n*n*n; ++i) {
                double x = 1.0 - 2.0 * c.data[i];
                double y = x + sigma * normal_dist_rand();
                L_channel.data[i] = 2.0 * y / (sigma * sigma);
            }

            Tensor_d L_APP = create_tensor_d(n, n, n);
            Tensor_d L_E = create_tensor_d(n, n, n);
            Tensor_d L_A = create_tensor_d(n, n, n);
            Tensor_d input = create_tensor_d(n, n, n);

            double vec_in[n], vec_L_APP[n], vec_L_E[n];

            for (int i = 1; i <= Imax; i++) {
                n_iter_total += 0.5;
                for(int j=0; j<n*n*n; ++j) L_A.data[j] = alpha[2*i-2] * L_E.data[j];
                for(int j=0; j<n*n*n; ++j) input.data[j] = L_channel.data[j] + L_A.data[j];

                for (int slice = 0; slice < n; slice++) {
                    for (int col = 0; col < n; col++) {
                        for(int row=0; row<n; ++row) vec_in[row] = get_tensor_d(input, row, col, slice);
                        SOGRAND_bitSO(vec_L_APP, vec_L_E, NULL, vec_in, H, n, k, L, Tmax, thres, even);
                        for(int row=0; row<n; ++row) {
                            set_tensor_d(L_APP, row, col, slice, vec_L_APP[row]);
                            set_tensor_d(L_E, row, col, slice, vec_L_E[row]);
                        }
                    }
                }
                if (early_termination(L_APP, G, n, k)) break;

                n_iter_total += 0.5;
                for(int j=0; j<n*n*n; ++j) L_A.data[j] = alpha[2*i-2] * L_E.data[j];
                for(int j=0; j<n*n*n; ++j) input.data[j] = L_channel.data[j] + L_A.data[j];

                for (int slice = 0; slice < n; slice++) {
                    for (int row = 0; row < n; row++) {
                        for(int col=0; col<n; ++col) vec_in[col] = get_tensor_d(input, row, col, slice);
                        SOGRAND_bitSO(vec_L_APP, vec_L_E, NULL, vec_in, H, n, k, L, Tmax, thres, even);
                        for(int col=0; col<n; ++col) {
                            set_tensor_d(L_APP, row, col, slice, vec_L_APP[col]);
                            set_tensor_d(L_E, row, col, slice, vec_L_E[col]);
                        }
                    }
                }
                if (early_termination(L_APP, G, n, k)) break;

                n_iter_total += 0.5;
                for(int j=0; j<n*n*n; ++j) L_A.data[j] = alpha[2*i-2] * L_E.data[j]; 
                for(int j=0; j<n*n*n; ++j) input.data[j] = L_channel.data[j] + L_A.data[j];

                for (int row = 0; row < n; row++) {
                    for (int col = 0; col < n; col++) {
                        for(int slice=0; slice<n; ++slice) vec_in[slice] = get_tensor_d(input, row, col, slice);
                        SOGRAND_bitSO(vec_L_APP, vec_L_E, NULL, vec_in, H, n, k, L, Tmax, thres, even);
                        for(int slice=0; slice<n; ++slice) {
                            set_tensor_d(L_APP, row, col, slice, vec_L_APP[slice]);
                            set_tensor_d(L_E, row, col, slice, vec_L_E[slice]);
                        }
                    }
                }
                if (early_termination(L_APP, G, n, k)) break;
            }

            Tensor_i c_HD = create_tensor_i(n, n, n);
            for(int i=0; i<n*n*n; ++i) c_HD.data[i] = (L_APP.data[i] > 0) ? 0 : 1;

            int is_block_error = 0;
            for(int i=0; i<n*n*n; ++i) {
                if(c_HD.data[i] != c.data[i]) {
                    is_block_error = 1;
                    break;
                }
            }

            if (is_block_error) {
                BlockError++;
                Tensor_i uhat = create_tensor_i(k, k, k);
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        for (int l = 0; l < k; l++) {
                            int val = get_tensor_i(c_HD, i, j, l);
                            set_tensor_i(uhat, i, j, l, val);
                            if (val != get_tensor_i(u, i, j, l)) {
                                BitError++;
                            }
                        }
                    }
                }
                free_tensor_i(uhat);
            }

            free_tensor_i(u);
            free_tensor_i(c);
            free_tensor_i(c_HD);
            free_tensor_d(L_channel);
            free_tensor_d(L_APP);
            free_tensor_d(L_E);
            free_tensor_d(L_A);
            free_tensor_d(input);

            if (ntx % 10 == 0) {
                 printf("\rFrames: %lld, Block Errors: %lld", ntx, BlockError);
                 fflush(stdout);
            }
        }

        printf("\n");
        printf("BLER: %e\n", (double)BlockError / ntx);
        printf("BER: %e\n", (double)BitError / (ntx * k * k * k));
        printf("Avg Iterations: %.2f\n", n_iter_total / ntx);
    }

    for(int i=0; i<k; ++i) free(G[i]);
    free(G);
    for(int i=0; i<n-k; ++i) free(H[i]);
    free(H);

    return 0;
}


int early_termination(Tensor_d L_APP, int** G, int n, int k) {
    Tensor_i c_HD = create_tensor_i(n, n, n);
    for(int i=0; i<n*n*n; ++i) c_HD.data[i] = (L_APP.data[i] > 0) ? 0 : 1;

    Tensor_i c_test = create_tensor_i(n, n, n);
    int temp_vec[n];
    int temp_vec_k[k];

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                set_tensor_i(c_test, i, j, l, get_tensor_i(c_HD, i, j, l));
            }
        }
    }

    for (int slice = 0; slice < k; slice++) {
        for (int row = 0; row < k; row++) {
            for(int i=0; i<k; ++i) temp_vec_k[i] = get_tensor_i(c_test, row, i, slice);
            memset(temp_vec, 0, n * sizeof(int));
            for (int col = 0; col < n; col++) {
                for (int g_col = 0; g_col < k; g_col++) {
                    temp_vec[col] += temp_vec_k[g_col] * G[g_col][col];
                }
            }
            for (int col = 0; col < n; col++) {
                set_tensor_i(c_test, row, col, slice, temp_vec[col] % 2);
            }
        }
        for (int col = 0; col < n; col++) {
            for(int i=0; i<k; ++i) temp_vec_k[i] = get_tensor_i(c_test, i, col, slice);
            memset(temp_vec, 0, n * sizeof(int));
            for (int row = 0; row < n; row++) {
                for (int g_col = 0; g_col < k; g_col++) {
                    temp_vec[row] += temp_vec_k[g_col] * G[g_col][row];
                }
            }
            for (int row = 0; row < n; row++) {
                set_tensor_i(c_test, row, col, slice, temp_vec[row] % 2);
            }
        }
    }
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
             for(int i=0; i<k; ++i) temp_vec_k[i] = get_tensor_i(c_test, row, col, i);
             memset(temp_vec, 0, n * sizeof(int));
             for (int slice = 0; slice < n; slice++) {
                 for (int g_col = 0; g_col < k; g_col++) {
                     temp_vec[slice] += temp_vec_k[g_col] * G[g_col][slice];
                 }
             }
             for (int slice = 0; slice < n; slice++) {
                set_tensor_i(c_test, row, col, slice, temp_vec[slice] % 2);
            }
        }
    }

    int is_equal = 1;
    for(int i=0; i<n*n*n; ++i) {
        if (c_HD.data[i] != c_test.data[i]) {
            is_equal = 0;
            break;
        }
    }

    free_tensor_i(c_HD);
    free_tensor_i(c_test);

    return is_equal;
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

double normal_dist_rand() {
    double u1 = (double)rand() / (RAND_MAX + 1.0);
    double u2 = (double)rand() / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
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

void getGH_sys_CRC(int n, int k, int** G, int** H) {
    const char* hex_poly = NULL;
    int r = n - k;

    if (r == 3) hex_poly = "0x5";
    else if (r == 4) hex_poly = "0x9";
    else if (r == 5 && k <= 10) hex_poly = "0x15";
    else if (r == 6 && k <= 25) hex_poly = "0x23";
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
        G[i][i] = 1;
        for (int j = 0; j < r; j++) G[i][k + j] = P[i][j];
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < k; j++) H[i][j] = P[j][i];
        H[i][k + i] = 1;
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

    for (int i = 0; i < len; i++) {
        poly[i] = (dec_val >> (len - 1 - i)) & 1;
    }
    poly[len] = 1;

    return poly;
}

void sogrand_main_logic(double *chat, double *score, double *T, double *curL, double *pNL, double *APP, double *llr, uint8_t *H, uint64_t n, uint64_t s, int32_t IC, uint64_t L, uint64_t Tmax, double thres, uint8_t even){
    size_t *perm = calloc(n, sizeof(size_t));
    uint8_t *cHD = calloc(n, sizeof(uint8_t));
    uint8_t *TEP = calloc(n, sizeof(uint8_t));
    uint8_t *c   = calloc(n, sizeof(uint8_t));
    double *absL = calloc(n, sizeof(double));
    int32_t *u   = calloc(n, sizeof(int32_t));
    int32_t *d   = calloc(n, sizeof(int32_t));
    int32_t *D   = calloc(n, sizeof(int32_t));

    for(size_t i = 0; i < n; i++) perm[i] = i;
    for(size_t i = 0; i < 4*L; i++) score[i] = 0;
    for(size_t i = 0; i < L; i++) APP[i] = 0;

    uint64_t cur_L = 0;
    HardDec(cHD, llr, n);
    uint8_t parity_cHD = parity(cHD,n);
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

            } while (++mountain_iter_guard < n*n*n);

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
void HardDec(uint8_t *c, double *llr, uint64_t n) { for (size_t i = 0; i < n; i++) c[i] = (llr[i] > 0.0) ? 0 : 1; }
int parity(uint8_t array[], uint64_t n) { int sum = 0; for (uint64_t i = 0; i < n; i++) sum += array[i]; return sum % 2; }
double prob_parity(int parity_cHD, double *absL, uint64_t n) { double p_e = 1.0; for (uint64_t i = 0; i < n; i++) { p_e *= (1.0 - 2.0 * exp(-absL[i]) / (1.0 + exp(-absL[i]))); } p_e = 0.5 * (1.0 + p_e); return (parity_cHD == 0) ? p_e : 1.0 - p_e; }
void AddTEP(uint8_t *c, uint8_t *cHD, uint8_t *TEP, size_t *perm, uint64_t n) { for (size_t i = 0; i < n; i++) c[perm[i]] = cHD[perm[i]] ^ TEP[i]; }
double JacLog(double x) { if (x > 50) return x; if (x < -50) return 0.0; return log(1.0 + exp(x)); }
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
