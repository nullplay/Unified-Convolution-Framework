#include "taco.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>

using namespace std;
using namespace taco;

void fillDense(Tensor<float>& T, float fill, Format format, vector<int> dimension) {
  T = taco::Tensor<float>("fill", dimension, format, (float)fill);
  T.pack();
  float* val = (float*)(T.getTacoTensorT()->vals);
  
  int fulldim = 1;
  for (auto d : dimension) { fulldim *= d;}
  for (int i = 0; i<fulldim; i++) { val[i] = fill; }
}


void fillSparseUniform(Tensor<float>& T, int input_sparsity, Format format, vector<int> dimension) {
  T = taco::Tensor<float>("fill", dimension, format);
  T.pack();
  float* val = (float*)(T.getTacoTensorT()->vals);
  assert(format.getOrder() == dimension.size());
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,99);
  if (dimension.size() == 4) {
    // Let's assume dimension is {N,C,H,W} for now.
    for (int n=0; n<dimension[0]; n++) {
      for (int c=0; c<dimension[1]; c++) {
        for (int h=0; h<dimension[2]; h++) {
          for (int w=0; w<dimension[3]; w++) {
            int number = distribution(generator);
            if (number >= input_sparsity) {
              T.insert({n,c,h,w}, (float)1.0);
            } else {
              T.insert({n,c,h,w}, (float)0.0);
            }
          }
        }
      }
    }
  }
  else if (dimension.size() == 5) {
    // Let's assume dimension is {N,CO,CI,H,W} for now.
    for (int n=0; n<dimension[0]; n++) {
      for (int co=0; co<dimension[1]; co++) {
        for (int ci=0; ci<dimension[2]; ci++) {
          for (int h=0; h<dimension[3]; h++) {
            for (int w=0; w<dimension[4]; w++) {
              int number = distribution(generator);
              if (number >= input_sparsity) {
                T.insert({n,co,ci,h,w}, (float)1.0);
              } else {
                T.insert({n,co,ci,h,w}, (float)0.0);
              }
            }
          }
        }
      }
    }
  }
  else { assert(false); }

  T.pack();
}


void readMCRS4DFromTNS(Tensor<float>& T, string filename, Format format, bool isPointConv) {
  ifstream f(filename);
  string line;
  getline(f, line); 
  
  //Get the dimension from the first line
  stringstream ss(line);
  int M, C, R, S;
  ss >> M;
  ss >> C;
  ss >> R;
  ss >> S;
  
  if (isPointConv) {
    R=S=1;
  }

  // Declare TACO Tensor
  assert(format.getOrder() == 4);
  T = taco::Tensor<float>("F", {M,C,R,S}, format);

  // Insert NNZ Coordinates into Tensor
  int nnz = 0;
  for (std::string line; std::getline(f, line); ) {
    stringstream ss(line);
    int m, c, r, s;
    ss >> m;
    ss >> c;
    ss >> r;
    ss >> s;
    if (isPointConv && !(r==0 && s==0)) continue; 
    T.insert({m,c,r,s}, (float)1.0);
    nnz++;
  }
  //cout << "Sparsity : " << 100.0 - ((float)nnz/(M*C*R*S))*100.0 << endl;
  T.pack();
}

void readMCRS4DFromSMTX(Tensor<float>& T, string filename, Format format, vector<int> MCRS) {
  ifstream f(filename);
  string line;
  getline(f, line); 
  int M = MCRS[0]; // Out Channel
  int C = MCRS[1]; // In Channel
  int R = MCRS[2]; // Filter Height
  int S = MCRS[3]; // Filter Width

  //Get the dimension from the first line
  stringstream ss(line);
  int M2, CRS;
  string M2s, CRSs;
  std::getline(ss, M2s, ',');
  M2 = stoi(M2s);
  std::getline(ss, CRSs, ',');
  CRS = stoi(CRSs);
  assert(M2==M && C*R*S==CRS);
  
  // Declare TACO Tensor
  assert(format.getOrder() == 4);
  T = Tensor<float>("F", {M,C,R,S}, format);
  
  // 1. Read pos array (second line)
  getline(f,line);
  stringstream ss2(line);
  int num;
  vector<int> pos_m;
  while (ss2 >> num) pos_m.push_back(num);
  assert(pos_m.size() == M+1);

  // 2. Read crd array (third line)
  getline(f,line);
  stringstream ss3(line);
  vector<int> crd_crs;
  while (ss3 >> num) crd_crs.push_back(num);

  // Insert
  for (int m = 0; m < M; m++) {
    for (int pos = pos_m[m]; pos<pos_m[m+1]; pos++) {
      int crs = crd_crs[pos]; // c*H*W + h*W + w
      int c = crs / (R*S);
      int r = (crs % (R*S)) / S;
      int s = (crs % (R*S)) % S;
      T.insert({m,c,r,s}, (float)1.0);
    }
  }
  T.pack();
}



void readMmCcRSFromTNS(Tensor<float>& T, string filename, Format format, int MI, int CI) {
  ifstream f(filename);
  string line;
  getline(f, line); 
  
  //Get the dimension from the first line
  stringstream ss(line);
  int M, C, R, S;
  ss >> M;
  ss >> C;
  ss >> R;
  ss >> S;
  
  int CO = (C+CI-1)/CI; 
  int MO = (M+MI-1)/MI;

  // Declare TACO Tensor
  assert(format.getOrder() == 6);
  T = taco::Tensor<float>("F", {MO,MI,CO,CI,R,S}, format);

  // Insert NNZ Coordinates into Tensor
  for (std::string line; std::getline(f, line); ) {
    stringstream ss(line);
    int m, c, r, s;
    ss >> m;
    ss >> c;
    ss >> r;
    ss >> s;
    T.insert({m/MI,m%MI,c/CI,c%CI,r,s}, (float)1.0);
  }
  T.pack();
}


void readMmCcRSFromSMTX(Tensor<float>& T, string filename, Format format, vector<int> MCRS, int MI, int CI) {
  ifstream f(filename);
  string line;
  getline(f, line); 
  int M = MCRS[0];
  int C = MCRS[1];
  int R = MCRS[2];
  int S = MCRS[3];

  //Get the dimension from the first line
  stringstream ss(line);
  int M2, CRS;
  string M2s, CRSs;
  std::getline(ss, M2s, ',');
  M2 = stoi(M2s);
  std::getline(ss, CRSs, ',');
  CRS = stoi(CRSs);
  assert(M2==M && C*R*S==CRS);
  
  // Declare TACO Tensor
  int CO = (C+CI-1)/CI; 
  int MO = (M+MI-1)/MI;
  assert(format.getOrder() == 6);
  T = Tensor<float>("F", {MO,MI,CO,CI,R,S}, format);
  
  // 1. Read pos array (second line)
  getline(f,line);
  stringstream ss2(line);
  int num;
  vector<int> pos_m;
  while (ss2 >> num) pos_m.push_back(num);
  assert(pos_m.size() == M+1);

  // 2. Read crd array (third line)
  getline(f,line);
  stringstream ss3(line);
  vector<int> crd_crs;
  while (ss3 >> num) crd_crs.push_back(num);

  // Insert
  for (int m = 0; m < M; m++) {
    for (int pos = pos_m[m]; pos<pos_m[m+1]; pos++) {
      int crs = crd_crs[pos]; // c*H*W + h*W + w
      int c = crs / (R*S);
      int r = (crs % (R*S)) / S;
      int s = (crs % (R*S)) % S;
      T.insert({m/MI,m%MI,c/CI,c%CI,r,s}, (float)1.0);
    }
  }
  T.pack();
}



