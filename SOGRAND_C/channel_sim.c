#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// Helper to generate a standard normal random number
double normal_dist_rand() {
    double u1 = (double)rand() / (RAND_MAX + 1.0);
    double u2 = (double)rand() / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <Eb/N0_in_dB> <code_dimension>\n", argv[0]);
        fprintf(stderr, "  <code_dimension> should be 2 for square code, 3 for cubic code.\n");
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    double EbN0dB = atof(argv[3]);
    int code_dim = atoi(argv[4]);

    if (code_dim != 2 && code_dim != 3) {
        fprintf(stderr, "Error: Code dimension must be 2 or 3.\n");
        return 1;
    }

    // --- Code parameters for calculating noise variance ---
    const int n = 16;
    const int k = 8;
    const double R = pow((double)k / n, code_dim); // Rate depends on dimension

    // --- Calculate sigma from Eb/N0 ---
    double EsN0dB = EbN0dB + 10 * log10(2 * R);
    double sigma = 1.0 / sqrt(pow(10, EsN0dB / 10.0));
    double variance = sigma * sigma;

    FILE* fin = fopen(input_filename, "rb");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }
    FILE* fout = fopen(output_filename, "wb");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    printf("Simulating AWGN channel for Eb/N0 = %.2f dB (sigma = %f)...\n", EbN0dB, sigma);
    srand(time(NULL));

    // added by sarah 
    FILE* f_noise = fopen("noise_values.bin", "wb");
    FILE* f_y = fopen("y_values.bin", "wb");

    int byte_in;
    while ((byte_in = fgetc(fin)) != EOF) {
        for (int i = 7; i >= 0; i--) {
            int bit = (byte_in >> i) & 1;

            // 1. Modulate (BPSK)
            double x = (bit == 0) ? 1.0 : -1.0;

            // 2. Add noise
            //double y = x + sigma * normal_dist_rand();

            // 3. Calculate LLR
            // double llr = 2.0 * y / variance;

            double noise = sigma * normal_dist_rand();
            double y = x + noise; 

            fwrite(&noise, sizeof(double), 1, f_noise);
            fwrite(&y, sizeof(double), 1, f_y);
            
            double llr = 2.0 * y / variance;

            // added here by Sarah for debugging
            // printf("LLR: %.6f\n", llr);  // Adjust precision if needed


            // 4. Write LLR to output file
            fwrite(&llr, sizeof(double), 1, fout);
        }
    }


    printf("Channel simulation complete. Output file '%s' contains LLRs.\n", output_filename);

    fclose(fin);
    fclose(fout);

    return 0;
}
