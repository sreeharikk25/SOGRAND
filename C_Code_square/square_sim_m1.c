#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#define Inf 0x7fffffff


void print_matrix_int(const char* title, int** matrix, int rows, int cols);
int** create_int_matrix(int rows, int cols);
double** create_double_matrix(int rows, int cols);
void free_int_matrix(int** matrix, int rows);
void free_double_matrix(double** matrix, int rows);
double normal_dist_rand();


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
void sogrand_main_logic(double *chat, double *score, double *T, double *curL, double *pNL, double *APP, double *llr, uint8_t *H, uint64_t n, uint64_t s, int32_t IC, uint64_t L, uint64_t Tmax, double thres, uint8_t even);


int* koopman2matlab(const char* k_poly, int* poly_len);
void getGH_sys_CRC(int n, int k, int** G, int** H);
void SOGRAND_bitSO(double* L_APP, double* L_E, int* N_guess, double* llr, int** H_matrix, int n, int k, int L, uint64_t Tmax, double thres, int even);


int main() {
    const int num_simulations = 1000;
    const int k = 4;
    const int n = 7;

    printf("=========================================================\n");
    printf("        SOGRAND C Implementation - Full Simulation       \n");
    printf("=========================================================\n");
    printf("Starting %d simulations with (%d, %d) code...\n\n", num_simulations, n, k);

    srand(time(NULL));


    int successful_decodes = 0;
    long long total_bit_errors = 0;
    long long total_bits_simulated = 0;


    int** G = create_int_matrix(k, n);
    int** H = create_int_matrix(n - k, n);
    getGH_sys_CRC(n, k, G, H);

    double EbN0dB = 2.5;
    int L = 4;
    int Imax = 10;
    uint64_t Tmax = 0;
    double p_ET = 1e-5;
    double thres = 1.0 - p_ET;
    double alpha[50];
    for(int i = 0; i < 50; i++) alpha[i] = 0.5;

    int even = 0;

    double R = pow((double)k / n, 2.0);
    double EsN0dB = EbN0dB + 10 * log10(2 * R);
    double sigma = 1.0 / sqrt(pow(10, EsN0dB / 10.0));


    for (int sim_run = 1; sim_run <= num_simulations; sim_run++) {
        printf("\rRunning Simulation: %d / %d", sim_run, num_simulations);
        fflush(stdout);


        int** u = create_int_matrix(k, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                u[i][j] = rand() % 2;
            }
        }
        total_bits_simulated += k * k;


        int** c = create_int_matrix(n, n);
        int temp_row[n];

        for (int row = 0; row < k; row++) {
            for (int col = 0; col < n; col++) {
                c[row][col] = 0;
                for (int G_col = 0; G_col < k; G_col++) {
                    c[row][col] += u[row][G_col] * G[G_col][col];
                }
                c[row][col] %= 2;
            }
        }
        for (int row = k; row < n; row++) {
            for (int col = 0; col < n; col++) {
                c[row][col] = 0;
            }
        }
        for (int col = 0; col < n; col++) {
            int temp_c_col[k];
            for(int i = 0; i < k; i++) temp_c_col[i] = c[i][col];
            for (int row = 0; row < n; row++) {
                int val = 0;
                for (int G_col = 0; G_col < k; G_col++) {
                    val += temp_c_col[G_col] * G[G_col][row];
                }
                temp_row[row] = val % 2;
            }
            for(int row = 0; row < n; row++) c[row][col] = temp_row[row];
        }


        double** x = create_double_matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                x[i][j] = 1.0 - 2.0 * c[i][j];
            }
        }
        int flip_row = rand() % n;
        int flip_col = rand() % n;
        x[flip_row][flip_col] = -x[flip_row][flip_col];

        double** L_channel = create_double_matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double y = x[i][j] + sigma * normal_dist_rand();
                L_channel[i][j] = 2.0 * y / (sigma * sigma);
            }
        }


        double** L_APP = create_double_matrix(n, n);
        double** L_E = create_double_matrix(n, n);

        for (int i = 1; i <= Imax; i++) {
            double** input_row_matrix = create_double_matrix(n, n);
            for(int r=0; r<n; ++r) for(int cl=0; cl<n; ++cl) input_row_matrix[r][cl] = L_channel[r][cl] + alpha[2*i-2] * L_E[r][cl];
            for (int row = 0; row < n; row++) {
                SOGRAND_bitSO(&L_APP[row][0], &L_E[row][0], NULL, &input_row_matrix[row][0], H, n, k, L, Tmax, thres, even);
            }
            free_double_matrix(input_row_matrix, n);

            double** input_col_matrix = create_double_matrix(n, n);
            for(int r=0; r<n; ++r) for(int cl=0; cl<n; ++cl) input_col_matrix[r][cl] = L_channel[r][cl] + alpha[2*i-1] * L_E[r][cl];
            for (int col = 0; col < n; col++) {
                double input_col_vec[n], L_APP_col_vec[n], L_E_col_vec[n];
                for(int r=0; r<n; ++r) input_col_vec[r] = input_col_matrix[r][col];
                SOGRAND_bitSO(L_APP_col_vec, L_E_col_vec, NULL, input_col_vec, H, n, k, L, Tmax, thres, even);
                for(int r=0; r<n; ++r) L_APP[r][col] = L_APP_col_vec[r];
                for(int r=0; r<n; ++r) L_E[r][col] = L_E_col_vec[r];
            }
            free_double_matrix(input_col_matrix, n);

            int** c_HD = create_int_matrix(n, n);
            for(int r=0; r<n; ++r) for(int cl=0; cl<n; ++cl) c_HD[r][cl] = 0.5 * (1 - (L_APP[r][cl] > 0 ? 1 : -1));
            int s1_sum = 0;
            for(int r=0; r<n; ++r) {
                for(int s_row=0; s_row < n-k; ++s_row) {
                    int syn_bit = 0;
                    for(int c_col=0; c_col < n; ++c_col) {
                        syn_bit += c_HD[r][c_col] * H[s_row][c_col];
                    }
                    s1_sum += syn_bit % 2;
                }
            }
            free_int_matrix(c_HD, n);
            if (s1_sum == 0) break;
        }


        int** uhat = create_int_matrix(k, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                uhat[i][j] = 0.5 * (1.0 - (L_APP[i][j] > 0 ? 1.0 : -1.0));
            }
        }

        int errors = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (u[i][j] != uhat[i][j]) {
                    errors++;
                }
            }
        }

        if (errors == 0) {
            successful_decodes++;
        } else {
            total_bit_errors += errors;
        }


        free_int_matrix(u, k);
        free_int_matrix(uhat, k);
        free_int_matrix(c, n);
        free_double_matrix(x, n);
        free_double_matrix(L_channel, n);
        free_double_matrix(L_APP, n);
        free_double_matrix(L_E, n);
    }


    double accuracy = (double)successful_decodes / num_simulations * 100.0;
    double average_BER = (double)total_bit_errors / total_bits_simulated;

    printf("\n\n");
    printf("=============================================\n");
    printf("           FINAL SIMULATION REPORT           \n");
    printf("=============================================\n");
    printf("Total Simulations Run:      %d\n", num_simulations);
    printf("Successful Decodes:         %d\n", successful_decodes);
    printf("Failed Decodes:             %d\n", num_simulations - successful_decodes);
    printf("---------------------------------------------\n");
    printf("Overall Decoding Accuracy:  %.2f %%\n", accuracy);
    printf("Average Bit Error Rate (BER): %e\n", average_BER);
    printf("=============================================\n");


    free_int_matrix(G, k);
    free_int_matrix(H, n - k);

    return 0;
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

    int** P = create_int_matrix(k, r);
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
    free_int_matrix(P, k);
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
