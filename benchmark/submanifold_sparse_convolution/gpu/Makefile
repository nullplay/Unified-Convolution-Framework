CXX=g++
CXXFLAG=-std=c++11 -O3 -march=native -mtune=native -fopenmp -g
INCLUDEPATH=-I../../../include/ -I../../../src/
LINKPATH=-L../../../build/lib/  
LINKFLAG=-ltaco -lm -ldl

lidar:
		$(CXX) -DLIDAR $(CXXFLAG) $(INCLUDEPATH) $(LINKPATH) 3dgpu.cpp -o 3dgpu_lidar $(LINKFLAG) 
else:
		$(CXX) -DELSE $(CXXFLAG) $(INCLUDEPATH) $(LINKPATH) 3dgpu.cpp -o 3dgpu_else $(LINKFLAG) 
