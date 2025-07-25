#include <stdio.h>
#include <stdint.h>

// Original (incorrect) parity check
int parity_check_old(uint8_t *c, uint8_t *H, int n, int s) {
    for (int j = 0; j < s; j++) {
        uint8_t syndrome = 0;
        for (int i = 0; i < n; i++) {
            syndrome ^= (c[i] * H[j*n + i]);
        }
        if (syndrome != 0) return 0;  // WRONG: should be == 1
    }
    return 1;
}

// Fixed parity check (matches C implementation)
int parity_check_fixed(uint8_t *c, uint8_t *H, int n, int s) {
    for (int j = 0; j < s; j++) {
        uint8_t syndrome = 0;
        for (int i = 0; i < n; i++) {
            syndrome ^= (c[i] * H[j*n + i]);
        }
        if (syndrome == 1) return 0;  // CORRECT: matches C code
    }
    return 1;
}

int main() {
    // Test case: n=31, k=25, so s=6
    int n = 31, s = 6;
    
    // Create a simple parity check matrix H (6x31)
    uint8_t H[6*31] = {0};
    
    // Set up a simple systematic parity check matrix
    // H = [P^T | I_6] where P is 25x6
    for (int i = 0; i < 6; i++) {
        H[i*n + 25 + i] = 1;  // Identity part
    }
    
    printf("Testing parity check logic differences:\n\n");
    
    // Test with different syndrome values
    uint8_t c[31] = {0};
    
    // Test case 1: Syndrome = 0 (valid)
    printf("Test 1 - Syndrome = 0 (valid codeword):\n");
    printf("  Old check: %d\n", parity_check_old(c, H, n, s));
    printf("  Fixed check: %d\n", parity_check_fixed(c, H, n, s));
    printf("  Expected: 1 (valid)\n\n");
    
    // Test case 2: Syndrome = 1 (invalid)
    // Create a codeword that gives syndrome = 1
    uint8_t c2[31] = {0};
    c2[25] = 1;  // This will give syndrome = 1 for the first parity check
    printf("Test 2 - Syndrome = 1 (invalid codeword):\n");
    printf("  Old check: %d\n", parity_check_old(c2, H, n, s));
    printf("  Fixed check: %d\n", parity_check_fixed(c2, H, n, s));
    printf("  Expected: 0 (invalid)\n\n");
    
    // Test case 3: Syndrome = 2 (invalid)
    uint8_t c3[31] = {0};
    c3[26] = 1;  // This will give syndrome = 1 for the second parity check
    printf("Test 3 - Syndrome = 1 for second check (invalid codeword):\n");
    printf("  Old check: %d\n", parity_check_old(c3, H, n, s));
    printf("  Fixed check: %d\n", parity_check_fixed(c3, H, n, s));
    printf("  Expected: 0 (invalid)\n\n");
    
    // Test case 4: Syndrome = 3 (invalid)
    uint8_t c4[31] = {0};
    c4[25] = 1;
    c4[26] = 1;  // This will give syndrome = 1 for multiple checks
    printf("Test 4 - Multiple syndrome = 1 (invalid codeword):\n");
    printf("  Old check: %d\n", parity_check_old(c4, H, n, s));
    printf("  Fixed check: %d\n", parity_check_fixed(c4, H, n, s));
    printf("  Expected: 0 (invalid)\n\n");
    
    // Test case 5: Syndrome = 2 (invalid but old logic would accept)
    uint8_t c5[31] = {0};
    c5[0] = 1;  // This might give syndrome = 2 (not 1)
    printf("Test 5 - Syndrome = 2 (invalid codeword):\n");
    printf("  Old check: %d\n", parity_check_old(c5, H, n, s));
    printf("  Fixed check: %d\n", parity_check_fixed(c5, H, n, s));
    printf("  Note: Old logic accepts syndrome != 0, fixed logic only rejects syndrome == 1\n\n");
    
    printf("Key Difference:\n");
    printf("- Old logic: rejects if syndrome != 0 (rejects all errors)\n");
    printf("- Fixed logic: rejects if syndrome == 1 (rejects only odd-weight errors)\n");
    printf("- For binary codes, syndrome can only be 0 or 1, so they are equivalent!\n");
    printf("- The real issue is likely in the syndrome calculation or matrix setup.\n");
    
    return 0;
}