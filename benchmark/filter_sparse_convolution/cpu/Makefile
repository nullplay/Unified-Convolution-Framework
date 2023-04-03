CXXFLAG=-std=c++11 -O3 -march=native -mtune=native -qopenmp -g
GXXFLAG=-std=c++11 -O3 -march=native -mtune=native -fopenmp -g
INCLUDEPATH=-I../../../include/ -I../../../src/
LINKPATH=-L../../../build/lib/  
LINKFLAG=-ltaco -lm -ldl

icc:
	icpc -DICC $(CXXFLAG) $(INCLUDEPATH) $(LINKPATH) filter_sparse_2dconv.cpp -o filter_sparse_2dconv	$(LINKFLAG)

gcc:
	g++ -DGCC $(GXXFLAG) $(INCLUDEPATH) $(LINKPATH) filter_sparse_2dconv.cpp -o filter_sparse_2dconv	$(LINKFLAG)



