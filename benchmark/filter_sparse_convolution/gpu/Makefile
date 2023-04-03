GXXFLAG=-std=c++11 -O3 -march=native -mtune=native -fopenmp -g
INCLUDEPATH=-I../../../include/ -I../../../src/
LINKPATH=-L../../../build/lib/  
LINKFLAG=-ltaco -lm -ldl

gpu:
	g++ $(GXXFLAG) $(INCLUDEPATH) $(LINKPATH) filter_sparse_2dconv_gpu.cpp -o filter_sparse_2dconv_gpu	$(LINKFLAG)


