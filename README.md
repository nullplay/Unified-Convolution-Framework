This is a repository for an artifact evaluation of "Unified Convolution Framework: A compiler-based approach to support sparse convolutions".

Build TACO-UCF using CMake 3.4.0 or greater:

    cd <UCF-directory>
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    export LD_LIBRARY_PATH=$(pwd)/lib/:$LD_LIBRARY_PATH
    
## Building for CUDA
To build TACO-UCF for NVIDIA CUDA, add `-DCUDA=ON` to the cmake line above. For example:

    cmake -DCMAKE_BUILD_TYPE=Release -DCUDA=ON ..

Please also make sure that you have CUDA installed properly and that the following environment variables are set correctly:

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

The generated CUDA code will require compute capability 6.1 or higher to run.


## Running sparse convolutions
To run filter-sparse convolution:

    cd <taco-directory>/benchmark/filter_sparse_convolution

To run masked(submanifold) sparse convolution:

    cd <taco-directory>/benchmark/submanifold_sparse_convolution

