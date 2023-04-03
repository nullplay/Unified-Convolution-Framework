# (GPU) Filter Sparse Convolution Experimentation on Pruned ResNet50 

This directory contains codes for experimenting with filter sparse convolution on pruned ResNet50 on GPU.  

## How to build 
To build, run the following commands:

    make  

## Testing on Pruned ResNet
We are using fixed format and schedule for GPU filter sparse convolution; (a) dense input activation is stored in N(U)-C(U)-H(U)-W(U) format and (b) sparse filter is stored in M(U)-R(U)-S(U)-C(C) format. We used  (1) loop order to be "nmpqrsc" (outer most to inner most), (2) fuse three loops m, p and q, (3) split mpq by 256, and (4) parallelize a loop mpq/256 to GPU block and 256 to GPU thread.

To test on pruned ResNet, run the following command:

    ./resnet.sh ../../Dataset/ResNet50/unique<80|91|96|98>.cfg

    #Example: 96% Filter Sparse Convolution 
    ./resnet.sh ../../Dataset/ResNet50/unique96.cfg


Here, `unique<80|91|96|98>.txt` includes layer shapes and number(80,91,96,and 98) meaning a different model sparsity. If the elapsed time of each layer is displayed, the script ran successfully.

## Inspecting generated code by UCF
After running the script, you can find the generated filter sparse convolution code for each layer in the `./code/` directory.
