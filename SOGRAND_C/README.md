### Execution Steps
**To run the Cubic (3D) code simulation**, execute this command:
    ```bash
    ./cubic_flow.sh
    ```
***

### Changing the SNR
To change the $E_b/N_0$ value for a simulation, simply open the corresponding shell script (`square_flow.sh` or `cubic_flow.sh`) in a text editor and modify the `SNR_DB` variable at the top of the file.

For example, to set the SNR to **2.5 dB**:
```sh
# Inside cubic_flow.sh
SNR_DB="2.5"
```
