# (CPU) Dual Sparse Convolution Experimentation on Pruned ResNet50 

This directory contains codes for experimenting with dual sparse convolution on pruned ResNet50. To compile the generated UCF code, we recommend using Intel C++ Compiler Classic (`icc`, `icpc`) for better performance. You can download `icc` from [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.fyw7ne). Please note that you need to install Intel C++ Compiler Classic (`icc` and `icpc`), not a OneAPI compiler. 

## How to build 
To build, run the following commands:

    make gcc 
    make icc #if you have icc

## Testing on Pruned ResNet
We are using fixed format and schedule for dual sparse convolution; (a) sparse input activation is stored in N(U)-H(U)-W(U)-C(C) format and (b) sparse filter is stored in R(U)-S(U)-C(U)-M(C) format. We used  (1) loop order to be "npqrscm" (outer most to inner most), (2) fuse two loops p and q, and (3) parallelize a loop pq.

To test on pruned ResNet, run the following command:

    ./resnet.sh ../Dataset/ResNet50/unique<80|91|96|98>.cfg <Activation Sparsity>

    #Example: 96% Filter Sparsity and 70% Activation Sparisty. 
    ./resnet.sh ../Dataset/ResNet50/unique96.cfg 70


Here, `unique<80|91|96|98>.txt` includes layer shapes and number(80,91,96,and 98) meaning a different model sparsity. If the elapsed time of each layer is displayed, the script ran successfully.

## Inspecting generated code by UCF
After running the script, you can find the generated filter sparse convolution code for each layer in the `./code/` directory.
