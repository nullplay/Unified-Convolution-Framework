# Submanifold(Mask) Sparse Convolution Experimentation on CPU
This directory contains codes for experimenting with submanifold sparse convolution on CPU. 
To compile the generated UCF code, we recommend using Intel C++ Compiler Classic (`icc`, `icpc`) for better performance. You can download `icc` from [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.fyw7ne). Please note that you need to install Intel C++ Compiler Classic (`icc` and `icpc`), not a OneAPI compiler. 


## How to build 
To build, run the following commands:

    make gcc 
    make icc #if you have icc
    
## Testing on 3D point clouds
We have provided five real-world point clouds (`LIDAR1`, `LIDAR2`, `office`, `lobby`, and `conferenceRoom`).
To test submanifold sparse convolution (in_channel=out_channel=64), run the following command:

    ./3dcpu ../../Dataset/3D/<LIDAR1|LIDAR2|office|lobby|conferenceRoom>.tns 
        
If the elapsed time of each layer is displayed, the command ran successfully.


## Inspecting generated code by UCF
After running the command, you can find the generated submanifold sparse convolution code in the `./code/` directory.