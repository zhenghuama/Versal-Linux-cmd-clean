# FPGA + AI Engine + Host Flow with Versal ACAP

To run hardware emulation and view the trace on Waiter,

Do this once per terminal (or add to `~/.bashrc`)

```
source /tools/Xilinx/Vivado/2024.1/settings64.sh && source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/xilinx/xrt/lib:/tools/Xilinx/Vitis/2024.1/aietools/lib/lnx64.o
```

Inside the repo,

```
cd gemm
make analyze
```
