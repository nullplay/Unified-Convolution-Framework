# Submanifold(Mask) Sparse Convolution Experimentation on GPU
This directory contains codes for experimenting with submanifold sparse convolution on GPU. 

## Replace nvidia_cuda_arch with your CUDA

In [3dgpu.cpp:70](https://github.com/nullplay/Unified-Convolution-Framework/blob/d9fc39246fbe80b1a68e72d68b02cfb8863379fe/benchmark/submanifold_sparse_convolution/gpu/3dgpu.cpp#L70), please replace the number to your NVIDIA CUDA Architecture number. You can check your cuda number at [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

## How to build 
To build, run the following commands:

    make lidar # optimized version for LIDAR1 and LIDAR2
    make else  # optimized version for office, lobby, and conferenceRoom
    
## Testing on 3D point clouds
We have provided five real-world point clouds (`LIDAR1`, `LIDAR2`, `office`, `lobby`, and `conferenceRoom`).
To test submanifold sparse convolution (in_channel=out_channel=64), run the following command:

    ./3dgpu_lidar ../../Dataset/3D/LIDAR1.tns 
    ./3dgpu_lidar ../../Dataset/3D/LIDAR2.tns 
    
    ./3dgpu_else ../../Dataset/3D/office.tns 
    ./3dgpu_else ../../Dataset/3D/lobby.tns 
    ./3dgpu_else ../../Dataset/3D/conferenceRoom.tns 
    
If the elapsed time of each layer is displayed, the command ran successfully.

## Inspecting generated code by UCF
After running the command, you can find the generated submanifold sparse convolution code in the `./code/` directory.
