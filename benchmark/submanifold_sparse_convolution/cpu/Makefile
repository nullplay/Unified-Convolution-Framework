CXXFLAG=-std=c++11 -O3 -march=native -mtune=native -qopenmp -g
GXXFLAG=-std=c++11 -O3 -march=native -mtune=native -fopenmp -g
INCLUDEPATH=-I../../../include/ -I../../../src/
LINKPATH=-L../../../build/lib/  
LINKFLAG=-ltaco -lm -ldl

gcc:
		g++ -DGCC $(GXXFLAG) $(INCLUDEPATH) $(LINKPATH) 3dcpu.cpp -o 3dcpu $(LINKFLAG)

icc:
		icpc -DICC $(CXXFLAG) $(INCLUDEPATH) $(LINKPATH) 3dcpu.cpp -o 3dcpu $(LINKFLAG) 
