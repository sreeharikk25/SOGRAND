# Fixes Applied to Match MATLAB Implementation

This document summarizes the fixes applied to the C implementation to match the MATLAB code for cubic and product (square) turbo decoding with SOGRAND.

## Major Fixes Applied

### 1. **Iterative Decoding Structure**
- Updated both cubic and square decoders to match MATLAB's iteration structure
- Each dimension (rows/columns/slices for cubic, rows/columns for square) is processed separately
- Iteration counter increments by 0.5 for each dimension (matching MATLAB's `n_iter = n_iter + 0.5`)

### 2. **Alpha Scaling**
- Fixed alpha scaling to use different values for each dimension within an iteration
- Cubic decoder: Uses `alpha[2*iter-2]` for columns, `alpha[2*iter-1]` for rows and slices
- Square decoder: Uses `alpha[2*iter-2]` for rows, `alpha[2*iter-1]` for columns
- Default alpha values set to 0.7 for cubic and 0.5 for square (matching MATLAB)

### 3. **Extrinsic LLR Calculation**
- Added proper L_A (a priori LLR) calculation: `L_A = alpha * L_E`
- Input to SOGRAND is now `L_channel + L_A` (not `L_channel + alpha * L_E` directly)
- L_E (extrinsic LLR) is properly initialized to zeros at the start

### 4. **NG (Number of Guesses) Tracking**
- Added tracking for both NG (total guesses) and NG_p (parallel guesses)
- NGmax tracks the maximum guesses per dimension for parallel counting
- Reports average NG per block and per information bit

### 5. **Early Termination**
- Implemented proper early termination matching MATLAB's logic
- Re-encodes the hard decision systematically and checks if it matches
- For cubic: encodes rows, then columns, then slices
- For square: encodes rows, then columns

### 6. **Code Parameter Support**
- Added support for multiple code parameters including [31,25] for square codes
- Updated getGH_sys_CRC to support various (n,k) pairs with proper CRC polynomials
- Added even code detection

### 7. **Hard Decision Extraction**
- Fixed hard decision to extract only information bits
- Cubic: Extracts bits from positions [0:k, 0:k, 0:k]
- Square: Extracts bits from positions [0:k, 0:k]

### 8. **SOGRAND Integration**
- Fixed N_guess tracking in SOGRAND_bitSO
- Properly integrated the C implementation of SOGRAND with the turbo decoder
- Added all necessary forward declarations

### 9. **Decoder Parameters**
- L (list size): 3 for cubic, 4 for square
- Imax (max iterations): 30 for cubic, 20 for square
- Tmax: Set to UINT64_MAX (effectively infinite)
- Threshold: 1 - 1e-5 for both

### 10. **Compilation Issues**
- Added -lm flag to link math library
- Fixed all missing declarations and includes
- Resolved start_time/end_time issues

## Key Differences from Original Implementation

1. **Iteration Structure**: Original code didn't separate dimensions properly
2. **Alpha Usage**: Original used same alpha for all dimensions
3. **NG Tracking**: Original didn't track number of guesses
4. **Early Termination**: Original had simplified termination logic
5. **Code Support**: Original only supported limited code parameters

## Performance Notes

The implementation now closely matches the MATLAB simulation structure. The C implementation should provide equivalent decoding performance with much faster execution time due to:
- Compiled C vs interpreted MATLAB
- Efficient memory management
- Direct SOGRAND C implementation

Note: The implementation does not exploit the parallelizability mentioned in the MATLAB comments, where all rows/columns/slices could be decoded in parallel. This could be a future optimization using OpenMP or similar parallel processing frameworks.