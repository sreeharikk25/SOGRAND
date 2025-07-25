# CUDA Encoding Bug Analysis and Fix

## Root Cause of High BER (~0.47-0.48)

The extremely high BER you're experiencing is due to a **critical bug in the CUDA cubic encoder's systematic bit mapping**.

## The Bug

In the CUDA cubic encoder (`cubic_encoder.cu`), the `copy_systematic_kernel` has incorrect bit-to-tensor mapping:

```cuda
// BUGGY CODE:
int slice = local_idx / (k * k);
int row = (local_idx % (k * k)) / k;
int col = local_idx % k;
```

This maps linear index to tensor as: `local_idx → (col, row, slice)` order.

However, the C code expects: `local_idx → (row, col, slice)` order, based on the tensor indexing formula:
```c
// C tensor indexing: index = k*dim1*dim2 + j*dim1 + i
// For position (i,j,k) = (row,col,slice)
```

## Why This Causes ~50% BER

With the wrong bit mapping:
- The decoder expects bit 0 at position (0,0,0)
- But the encoder put bit 0's data at a different position
- The systematic bits are scrambled, making the decoder essentially guess randomly
- Result: ~50% BER (random guessing)

## The Fix

The corrected systematic copy kernel should be:

```cuda
// FIXED CODE:
int row = local_idx % k;
int col = (local_idx / k) % k;
int slice = local_idx / (k * k);
```

This ensures:
- Bit 0 → position (0,0,0) → tensor index 0
- Bit 1 → position (1,0,0) → tensor index 1
- Bit k → position (0,1,0) → tensor index k
- etc.

## Testing the Fix

1. Build the fixed encoder:
```bash
cd SOGRAND_CUDA
make cubic_encoder_fixed
```

2. Run the complete test:
```bash
make test_cubic_fixed
```

This will:
- Use the fixed encoder (`cubic_encoder_fixed`)
- Pass through the channel simulator
- Decode with the fixed decoder (`cubic_decoder_fixed`)
- Compare results

## Expected Results

With the fix, you should see:
- BER drop from ~0.47 to near 0 at high SNR
- Performance matching the C implementation
- Proper error correction capability

## Additional Notes

1. The square encoder doesn't have this bug because it encodes directly from input to output without an intermediate systematic copy step.

2. The encoding kernels themselves (rows, columns, slices) are correct - only the initial systematic bit placement was wrong.

3. This bug would affect any decoder (original or fixed) because the encoded data itself was wrong.

## Verification

To verify the fix works:
1. Encode a known pattern (e.g., all zeros)
2. Check that the systematic part of the codeword matches the input
3. Run through decoder and verify output matches input