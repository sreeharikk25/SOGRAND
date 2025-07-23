#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Complete CRC polynomial selection from Koopman's database
const char* select_crc_polynomial(int n, int k) {
    int r = n - k;  // redundancy
    
    if (r == 3) {
        return "0x5";  // 3 bit HD3 max length 4 otherwise HD2
    }
    else if (r == 4) {
        return "0x9";  // 4 bit HD3 max length 11, otherwise HD2
    }
    else if (r == 5 && k <= 10) {
        return "0x15"; // 5 bit HD4 max length 10
    }
    else if (r == 5 && k <= 26) {
        return "0x12"; // 5 bit HD3 max length 26
    }
    else if (r == 6 && k <= 25) {
        return "0x23"; // 6 bit HD4 max length 25
    }
    else if (r == 6) {
        return "0x33"; // 6 bit HD3 max length 57 or HD2 thereafter
    }
    else if (r == 7 && k <= 4) {
        return "0x72"; // 7 bit HD5 max length 4
    }
    else if (r == 7 && k <= 56) {
        return "0x5b"; // 7 bit HD4 max length 56
    }
    else if (r == 7 && k <= 120) {
        return "0x65"; // 7 bit HD4 max length 120
    }
    else if (r == 8 && k <= 4) {
        return "0x9b"; // 8 bit HD6 max length 4
    }
    else if (r == 8 && k <= 9) {
        return "0xeb"; // 8 bit HD5 max length 9
    }
    else if (r == 8 && k <= 119) {
        return "0x83"; // 8 bit HD4 max length 119
    }
    else if (r == 8 && k <= 247) {
        return "0xe7"; // 8 bit HD3 max length 247
    }
    else if (r == 9 && k <= 8) {
        return "0x13c"; // 9 bit HD6 max length 8
    }
    else if (r == 9 && k <= 13) {
        return "0x185"; // 9 bit HD5 max length 13
    }
    else if (r == 9 && k <= 246) {
        return "0x17d"; // 9 bit HD4 max length 246
    }
    else if (r == 10 && k <= 5) {
        return "0x29b"; // 10 bit HD7 max length 5
    }
    else if (r == 10 && k <= 12) {
        return "0x28e"; // 10 bit HD6 max length 12
    }
    else if (r == 10 && k <= 26) {
        return "0x2b9"; // 10 bit HD5 max length 26
    }
    else if (r == 10 && k <= 501) {
        return "0x247"; // 10 bit HD4 max length 501
    }
    else if (r == 11 && k <= 4) {
        return "0x4f5"; // 11 bit HD8 max length 4
    }
    else if (r == 11 && k <= 12) {
        return "0x571"; // 11 bit HD7 max length 12
    }
    else if (r == 11 && k <= 22) {
        return "0x532"; // 11 bit HD6 max length 22
    }
    else if (r == 11 && k <= 26) {
        return "0x5d7"; // 11 bit HD5 max length 26
    }
    else if (r == 11 && k <= 1012) {
        return "0x583"; // 11 bit HD4 max length 1012
    }
    else if (r == 12 && k <= 11) {
        return "0xa4f"; // 12 bit HD7/8 max length 11
    }
    else if (r == 12 && k <= 27) {
        return "0xb41"; // 12 bit HD6 max length 27
    }
    else if (r == 12 && k <= 53) {
        return "0xbae"; // 12 bit HD5 max length 53
    }
    else if (r == 12 && k <= 2035) {
        return "0x8f3"; // 12 bit HD4 max length 2035
    }
    else if (r == 13 && k <= 11) {
        return "0x10b7"; // 13 bit HD8 max length 11
    }
    else if (r == 13 && k <= 12) {
        return "0x12a5"; // 13 bit HD7 max length 12
    }
    else if (r == 13 && k <= 52) {
        return "0x1e97"; // 13 bit HD6 max length 52
    }
    else if (r == 14 && k <= 11) {
        return "0x2371"; // 14 bit HD8 max length 11
    }
    else if (r == 14 && k <= 13) {
        return "0x28a9"; // 14 bit HD7 max length 13
    }
    else if (r == 14 && k <= 57) {
        return "0x372b"; // 14 bit HD6 max length 57
    }
    else if (r == 14 && k <= 113) {
        return "0x212d"; // 14 bit HD5 max length 113
    }
    else if (r == 15 && k <= 5) {
        return "0x5a47"; // 15 bit HD9 max length 5
    }
    else if (r == 15 && k <= 114) {
        return "0x573a"; // 15 bit HD6 max length 114
    }
    else if (r == 15 && k <= 136) {
        return "0x6a8d"; // 15 bit HD5 max length 136
    }
    else if (r == 15 && k <= 16368) {
        return "0x4306"; // 15 bit HD4 max length 16368
    }
    else if (r == 16 && k <= 5) {
        return "0xed2f"; // 16 bit HD10 max length 5
    }
    else if (r == 16 && k <= 19) {
        return "0x968b"; // 16 bit HD6 max length 19
    }
    else if (r == 16 && k <= 135) {
        return "0x9eb2"; // 16 bit HD6 max length 135
    }
    else if (r == 16 && k <= 241) {
        return "0xac9a"; // 16 bit HD5 max length 241
    }
    else if (r == 16 && k <= 32751) {
        return "0xd175"; // 16 bit HD4 max length 32751
    }
    else if (r == 18 && k <= 5) {
        return "0x26a3d"; // 18 bit HD11 max length 5
    }
    else if (r == 18 && k <= 8) {
        return "0x2e7de"; // 18 bit HD10 max length 8
    }
    else if (r == 18 && k <= 240) {
        return "0x32c69"; // 18 bit HD6 max length 240
    }
    else if (r == 19 && k <= 7) {
        return "0x6d133"; // 19 bit HD11 max length 7
    }
    else if (r == 19 && k <= 10) {
        return "0x44f75"; // 19 bit HD10 max length 10
    }
    else if (r == 19 && k <= 494) {
        return "0x5685a"; // 19 bit HD6 max length 494
    }
    else if (r == 20 && k <= 11) {
        return "0x8d3cc"; // 20 bit HD11 max length 11
    }
    else if (r == 21 && k <= 10) {
        return "0x165751"; // 21 bit HD12 max length 10
    }
    else if (r == 21 && k <= 106) {
        return "0x12faa5"; // 21 bit HD7 max length 106
    }
    else if (r == 22 && k <= 5) {
        return "0x25d467"; // 22 bit HD13 max length 5
    }
    else if (r == 22 && k <= 105) {
        return "0x289cfe"; // 22 bit HD8 max length 105
    }
    else if (r == 23 && k <= 105) {
        return "0x469d7c"; // 23 bit HD8 max length 105
    }
    else if (r == 29 && k <= 100) {
        return "0x1e150a87"; // 29 bit HD9 max length 100
    }
    
    // Default case - not supported
    return NULL;
}