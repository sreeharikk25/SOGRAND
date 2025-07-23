### Simulation Pipelines

There are two primary simulation workflows available:

1.  **Square Code (2D)**: Simulates a `(225, 100)` product code constructed from two `(15, 10)` CRC codes.
2.  **Cubic Code (3D)**: Simulates a `(3375, 1000)` tensor product code constructed from three `(15, 10)` CRC codes.

***

### Execution Steps
1.  **To run the Square (2D) code simulation**, execute the following command in your terminal:
    ```bash
    ./square_flow.sh
    ```

2.  **To run the Cubic (3D) code simulation**, execute this command:
    ```bash
    ./cubic_flow.sh
    ```
***

### Changing the SNR
To change the $E_b/N_0$ value for a simulation, simply open the corresponding shell script (`square_flow.sh` or `cubic_flow.sh`) in a text editor and modify the `SNR_DB` variable at the top of the file.

For example, to set the SNR to **2.5 dB**:
```sh
# Inside square_flow.sh or cubic_flow.sh
SNR_DB="2.5"
```
