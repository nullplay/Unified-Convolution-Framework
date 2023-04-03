# Submanifold(Mask) Sparse Convolution Experimentation on CPU
This directory contains codes for experimenting with submanifold sparse convolution on CPU. 
To compile the generated UCF code, we recommend using Intel C++ Compiler Classic (`icc`, `icpc`) for better performance. You can download `icc` from [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.fyw7ne). Please note that you need to install Intel C++ Compiler Classic, not a OneAPI compiler. 


## How to build 
To build, run the following commands:

    make gcc 
    make icc #if you have icc
    
## Testing on 3D point clouds
We have provided five real-world point clouds (`LIDAR1`, `LIDAR2`, `office`, `lobby`, and `conferenceRoom`).
To test submanifold sparse convolution (in_channel=out_channel=64), run the following command:

    ./3dcpu ../../Dataset/3D/<LIDAR1|LIDAR2|office|lobby|conferenceRoom>.tns dense 0

    #Example : office.tns
    ./3dcpu ../../Dataset/3D/office.tns dense 0
        
If the elapsed time of each layer is displayed, the command ran successfully.


## Testing Dual Sparse Submanifold Convolution
You can also test dual sparse submanifold convolution where filter is pruned(sparsified) and stored in R(U)-S(U)-T(C)-C(U)-M(U) format. We prune a filter's spatial dimension (R,S,T) and make channels to be dense for survived spatial dimension. 

To test dual sparse submanifold convolution, run the following command:

    ./3dcpu ../../Dataset/3D/<LIDAR1|LIDAR2|office|lobby|conferenceRoom>.tns sparse <FilterSparsity>

    #Example: running dual sparse convolution on office.tns with 75% Filter Sparsity(75% of filter is zero).
    ./3dcpu ../../Dataset/3D/office.tns sparse 75
 
If the elapsed time of each layer is displayed, the command ran successfully.

## Inspecting generated code by UCF
After running the command, you can find the generated submanifold sparse convolution code in the `./code/` directory.
