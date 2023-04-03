# (CPU) Filter Sparse Convolution Experimentation on Pruned ResNet50

This directory contains codes for experimenting with filter sparse convolution on pruned ResNet50. To compile the generated UCF code, we recommend using Intel C++ Compiler Classic (`icc`, `icpc`) for better performance. You can download `icc` from [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.fyw7ne). Please note that you need to install Intel C++ Compiler Classic (`icc` and `icpc`), not a OneAPI compiler. 

## How to build 
To build, run the following commands:

    make gcc 
    make icc #if you have icc

## Testing on Pruned ResNet
In the `config` directory, we have provided the best format and schedule of each layer found by OpenTuner on **our system (Intel Xeon CPU E5-2680 v3 w/ icc)**. Please note that this configuration may not be optimal on a different system. Therefore, the user needs to run an auto-tuner to find the best format and schedule for their system. Nonetheless, this will give a reasonable performance.

To test on pruned ResNet, run the following command:

    ./resnet.sh config/ResNet<80|91|96|98>.txt 

Here, `config/ResNet<80|91|96|98>.txt` indicates a different model sparsity. If the elapsed time of each layer is displayed, the script ran successfully.

## Inspecting generated code by UCF
After running the script, you can find the generated filter sparse convolution code for each layer in the `./code/` directory.
