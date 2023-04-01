#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
#include <omp.h>
#include "time.hpp"
#include "taco.h"
#include "taco/ir/ir.h"
#include "taco/codegen/module.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen_cuda.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"
#include "taco/index_notation/transformations.h"
#include <regex>
using namespace taco;
using namespace std;
void* lib_handle = NULL;

typedef int (*compute)(taco_tensor_t* O, taco_tensor_t* Mask, taco_tensor_t* In, taco_tensor_t* F);
void* compileThenFunc(IndexStmt stmt) {
  string prefix_path="./code/";
  string c_file_name=prefix_path+"submanifold_cpu.c";
  string so_file_name=prefix_path+"submanifold_cpu.so";
 
  // Calling TACO-UCF to generate C code
  stringstream ss;
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(ss, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);

  // write generated C code into file
  // [TODO] Currently, regex handles sparse output as a patch to generated C code, but TACO-UCF should generate such code eventually. 
  string generated_code = std::regex_replace(ss.str(), std::regex("= uO "), "= uMask "); 
  ofstream source(c_file_name);
  source << generated_code ;
  source.close();

  // compile C code into shared library and load dynamically
#if defined(ICC)
  string gcc_command = "icc -march=native -mtune=native -O3 -qopenmp -ffast-math -fPIC -shared "+ c_file_name + " -o " + so_file_name + " -lm";
#elif defined(GCC)
  string gcc_command = "gcc -march=native -mtune=native -O3 -fopenmp -ffast-math -fPIC -shared "+ c_file_name + " -o " + so_file_name + " -lm";
#endif 
  system(gcc_command.c_str());
  if (lib_handle) {dlclose(lib_handle);}
  lib_handle = dlopen(so_file_name.c_str(), RTLD_NOW|RTLD_LOCAL);
  if (!lib_handle) {
    fprintf(stderr, "%s\n", dlerror());
    cout << "ERR" << endl;
  }
  return (dlsym(lib_handle, "compute"));
}

vector<vector<int>> fromFile(string filename) {
  ifstream f(filename);
  string line;
  getline(f, line); //Parsing the first line

  vector<vector<int>> coo;
  for (std::string line; std::getline(f, line); ) {
    stringstream ss(line);
    int crd1, crd2, crd3;
    ss >> crd1;
    ss >> crd2;
    ss >> crd3;
    coo.push_back({crd1, crd2, crd3});
  }
  return coo;
}

vector<int> shapeFromFile(string filename) {
  ifstream f(filename);
  string line;
  getline(f, line); //Parsing the first line
  stringstream ss(line);
  string sharp;
  ss >> sharp;
  int shape1, shape2, shape3;
  ss >> shape1;
  ss >> shape2;
  ss >> shape3;
  f.close();
  return {shape1, shape2, shape3};
}


//O[p,q,u,m] = Mask[p,q,u] * I[p+r,q+s,u+t,c] * F[m,r,s,t,c]
int main(int argc, char* argv[]) {
  string filename(argv[1]);
  // Input Shape
  int P = shapeFromFile(filename)[0];
  int Q = shapeFromFile(filename)[1];
  int U = shapeFromFile(filename)[2];
  // Filter Shape
  int R = 3;
  int S = 3;
  int T = 3;
  // In,Out Channel
  int C = 64;
  int M = 64;

  set_CUDA_codegen_enabled(false);
  set_CUDA_unified_memory_enabled(false);
  
  IndexVar p("p"), q("q"), r("r"), s("s"), t("t"), u("u"), c("c"), m("m"), b("b"), d("d"), e("e");
 
  auto QFormat = Sparse;
  auto UFormat = Sparse;
  Tensor<float> O("O", {P,Q,U,M}, Format{Dense,QFormat,UFormat,Dense});
  Tensor<float> Out("O", {P,Q,U,M}, Format{Dense,QFormat,UFormat,Dense});
  Tensor<float> I("In", {P,Q,U,C}, Format{Dense,QFormat,UFormat,Dense});
  Tensor<float> Mask("Mask", {P,Q,U}, Format{Dense,QFormat,UFormat});
  Tensor<float> F("F", {R,S,T,C,M}, Format{Dense,Dense,(argv[2]==std::string("dense"))?Dense:Sparse,Dense,Dense}); 
  Mask.setScalar();
  Tensor<float> B("b", {M}, Format{Dense});

  auto crds = fromFile(filename);
  for (auto p : crds) {
    //cout << p[0] << " " << p[1] << " " << p[2] << endl;
    for (int c = 0; c<1; c++) {
      I.insert({p[0], p[1], p[2],c}, (float)1.0);
    }
    for (int m = 0; m<1; m++) {
      O.insert({p[0], p[1], p[2], m}, (float)0.0);
    }
    Mask.insert({p[0], p[1], p[2]}, (float)1.0);
  }

  random_device rd;
  mt19937 gen;
  uniform_int_distribution<> dis(0,3);

  int cnt = 0;
  for (int r=0; r<R; r++) { 
    for (int s=0; s<S; s++) {
      for (int t=0; t<T; t++) {
        if (dis(gen)==0) { //75% Sparsity
          for (int c=0; c<C; c++) {
            for (int m=0; m<M; m++) {
                F.insert({r,s,t,c,m}, (float)1.0);
            }
          }
        }
      }
    }
  }

  F.pack();
  I.pack();
  Mask.pack();
  O.pack();
 

  IndexVar po("po"), pi("pi");
  IndexVar mo("mo"), mi("mi"), mib("mib"), co("co"), ci("ci"), cib("cib"), mb("mb"), cb("cb"),f("f"),fb("fb"); 
  
  Out(p,q,u,m) = Mask(p,q,u) * I(p+r-R/2, q+s-S/2, u+t-T/2, c) * F(r,s,t,c,m) ;
  IndexStmt stmt = Out.getAssignment().concretize()
                    .reorder({p,r,s,t,q,u,c,m})
                    .parallelize(p, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
                    //.reorder({p,q,r,s,t,u,c,m})
                    //.fuse(p,q,f)
                    //.bound(f,fb,P*Q,BoundType::MaxExact)
                    //.parallelize(fb, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);

  omp_set_num_threads(omp_get_max_threads());
  omp_set_schedule(omp_sched_dynamic, 1);
  
  compute func = (compute)compileThenFunc(stmt);
  func(O.getTacoTensorT(), Mask.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  func(O.getTacoTensorT(), Mask.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  func(O.getTacoTensorT(), Mask.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  auto start = Clock::now();
  for (int i = 0; i<30; i++){
    func(O.getTacoTensorT(), I.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  }
  auto end = Clock::now();
  cout << string(argv[1]) << " " << compute_clock(end, start) / (float)30 << " ms" <<endl;
  
  float* val = (float*)(O.getTacoTensorT()->vals);
  //for (int i =0; i<20; i++) {
  //  cout << val[i*M] << endl;
  //}

}
