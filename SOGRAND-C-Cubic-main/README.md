# SOGRAND-C-Cubic

Subject to license: "**GRAND Codebase Non-Commercial Academic Research Use License 021722.pdf**"

Non-parallelized C implementation of Soft Input/ Soft Output (SISO) GRAND, **SOGRAND** (basic and 1-line SISO ORBGRAND), in mex format so it can be run from MATLAB.

To compile, in MATLAB: mex -O SOGRAND_mex.c

Two MATLAB sample simulations are included:

1) sim_product decodes product codes with SOGRAND as the component decoder. Rows and columns are processed in MATLAB and their decoding is *not parallelized*.
2) sim_cubic decodes cubic tensor product codes with SOGRAND as the component decoder. Rows, columns and slices are processed in MATLAB and their decoding is *not parallelized*.
Output is recored in the results directory.

The following should be cited in association with results from this code.

K. R. Duffy, J. Li, and M. Medard, "Capacity-achieving guessing random additive noise decoding," IEEE Trans. Inf. Theory, vol. 65, no. 7, pp. 4023–4040, 2019.

K. R. Duffy, “Ordered reliability bits guessing random additive noise decoding," Proceedings of IEEE ICASSP, 8268–8272, 2021.

K. R. Duffy, W. An, and M. Medard, "Ordered reliability bits guessing random additive noise decoding,” IEEE Trans. Signal Process., vol. 70, pp. 4528-4542, 2022.

K. Galligan, P. Yuan, M. Médard, K. R. Duffy. "Upgrade error detection to prediction with GRAND". IEEE Globecom, 1818-1823, 2023.

P. Yuan, M. Medard, K. Galligan, K. R. Duffy. "Soft-output (SO) GRAND and Iterative Decoding to Outperform LDPC Codes". IEEE Trans. Wireless Commun., 2025.

S. Khalifeh, K. R. Duffy, and M. Medard, "Turbo product decoding of cubic tensor codes", CISS, 2025.

Altough the OBRGRAND implementation provided here is **not parallelized**, the algorithm istefl is *highly parallelizable*. E.g. A. Riaz, Y. Alperen, F. Ercan, W. An, J. Ngo, K. Galligan, M. Medard, K. R. Duffy, R. Yazicigil. "A sub-0.8-pJ/bit universal soft detection decoder using ORBGRAND,” IEEE J. Solid-State Circuits, 2025.

Moreover, *cubic code decoding is highly parallelizable*, where all rows can be decoded in parallel, all columns can be decoded in parallel, and all slices can be decoded in paralle, but this is **not exploited here**.

For further details on GRAND, see: https://www.granddecoder.mit.edu/
