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
  string prefix_path = "./code/";
  string sourcename = prefix_path+"submanifold_gpu.cu";
  string shimname = prefix_path+"submanifold_gpu_shim.cpp";
  string headername = prefix_path+"submanifold_gpu.h";
  string objectname = prefix_path+"submanifold_gpu.so";

  // Generated Cuda code using TACO-UCF
  stringstream ss;
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(ss, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);

  // sparse output patch using Regex
  string generated_code = std::regex_replace(ss.str(), std::regex("uO = uO \\+ 1;"), "");
  generated_code = std::regex_replace(generated_code, std::regex("= uO "), "= uMask ");
  generated_code = std::regex_replace(generated_code, std::regex("Mask_val \\* In_vals"), "In_vals");
  generated_code = std::regex_replace(generated_code, std::regex("O_vals\\[pO\\] = 0\\.0;"), "");
  
  // cuda kernel
  ofstream source_file;
  source_file.open(sourcename);
  source_file << generated_code;
  source_file.close();

  //header
  string header = "#ifndef TACO_C_HEADERS\n#define TACO_C_HEADERS\n#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <math.h>\n#include <thrust/complex.h>\n#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))\n#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)\n#ifndef TACO_TENSOR_T_DEFINED\n#define TACO_TENSOR_T_DEFINED\ntypedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;\ntypedef struct {\n  int32_t      order;         // tensor order (number of modes)\n  int32_t*     dimensions;    // tensor dimensions\n  int32_t      csize;         // component size\n  int32_t*     mode_ordering; // mode storage ordering\n  taco_mode_t* mode_types;    // mode storage types\n  uint8_t***   indices;       // tensor index data (per mode)\n  uint8_t*     vals;          // tensor values\n  int32_t      vals_size;     // values array size\n} taco_tensor_t;\n#endif\n#endif\n\n\n#ifndef TACO_GENERATED_compute\n#define TACO_GENERATED_compute\nint compute(taco_tensor_t *Out, taco_tensor_t *Mask, taco_tensor_t *In, taco_tensor_t *F);\n#endif";
  ofstream header_file;
  header_file.open(headername);
  header_file << header;
  header_file.close();

  //shim
  string shim = "\nextern \"C\" {\n  int _shim_compute(taco_tensor_t* Out, taco_tensor_t* Mask, taco_tensor_t* In, taco_tensor_t* F);\n}\n\nint _shim_compute(taco_tensor_t* Out, taco_tensor_t* Mask, taco_tensor_t* In, taco_tensor_t* F) {\n  return compute(Out,Mask,In,F);\n}";
  stringstream shims;
  ofstream shims_file;
  shims_file.open(shimname);
  shims_file << "#include \"" <<  "submanifold_gpu.h\"\n";
  shims_file << shim;
  shims_file.close();

  // compiling C code
  string cflag = " -maxrregcount=128  -w -O3 -Xcompiler \"-fPIC -shared -ffast-math -O3\" --generate-code arch=compute_70,code=sm_70 ";
  string compile_command = "nvcc" + cflag + sourcename + " " + shimname + " -o " + objectname + " -lm"; 
  int err = system(compile_command.c_str());
  if (err != 0) { throw std::invalid_argument("nvcc error"); }

  if (lib_handle) {dlclose(lib_handle);}
  lib_handle = dlopen(objectname.c_str(), RTLD_NOW|RTLD_LOCAL);
  if (!lib_handle) {
    fprintf(stderr, "%s\n", dlerror());
    cout << "ERR" << endl;
  }
  return (dlsym(lib_handle, "_shim_compute"));
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

  set_CUDA_codegen_enabled(true);
  set_CUDA_unified_memory_enabled(true);

#if defined LIDAR
  auto QFormat = Sparse; // For LIDAR1 LIDAR2   
#else
  auto QFormat = Dense; // For conferenceRoom, office, lobby
#endif
  
  auto UFormat = Sparse;

  Tensor<float> O("O", {P,Q,U,M}, Format({Dense,QFormat,UFormat,Dense}));
  Tensor<float> Mask("Mask", {P,Q,U}, Format({Dense,QFormat,UFormat}));
  Tensor<float> Out("O", {P,Q,U,M}, Format({Dense,QFormat,UFormat,Dense}));
  Tensor<float> I("In", {P,Q,U,C}, Format({Dense,QFormat,UFormat,Dense}));
  Tensor<float> F("F", {R,S,T,C,M}, Format({Dense,Dense,Dense,Dense,Dense}));

  Mask.setScalar();
  Tensor<float> B("b", {M}, Format{Dense});

  auto crds = fromFile(filename);
  for (auto p : crds) {
    I.insert({p[0], p[1], p[2], 0}, (float)1.0);
    O.insert({p[0], p[1], p[2], 0}, (float)0.0);
  }

  for (int r=0; r<R; r++)
    for (int s=0; s<S; s++)
      for (int t=0; t<T; t++)
        for (int m=0; m<M; m++)
          for (int c=0; c<1; c++)
            F.insert({r,s,t,c,m}, (float)1.0);

  I.pack();
  Mask.pack();
  F.pack();
  O.pack();

  //cout << filename << " " << P << "x" << Q << "x" << U <<  " (Nonzeros: " << crds.size() << ")" << I.getStorage().getSizeInBytes() << endl;
  cout << filename << " " << P << "x" << Q << "x" << U <<  " (Nonzeros: " << crds.size() << ")" << endl;
  IndexVar p("p"), q("q"), r("r"), s("s"), t("t"), u("u"), c("c"), m("m"), b("b"), d("d"), e("e"),f("f"),fb("fb"),ff("ff"), ffb("ffb"), fff("fff"), fffb("fffb"), pre_val("pre_val");
  IndexVar f1("f1"), f2("f2"), u1("u1"), u2("u2");
  IndexVar c1("c1"), c2("c2"), c3("c3");
  IndexVar m1("m1"), m2("m2"), m3("m3"), ffff("ffff");

  Out(p,q,u,m) = Mask(p,q,u) * I(p+r-R/2, q+s-S/2, u+t-T/2, c) * F(r,s,t,c,m);
 

#if defined LIDAR
  IndexStmt stmt = Out.getAssignment().concretize()
                    .reorder({p,r,s,t,q,u,m,c})
                    .fuse(p,r,f)
                    .fuse(f,s,ff)
                    .fuse(ff,t,fff)
                    .bound(fff,fffb,P*R*S*T,BoundType::MaxExact)
                    .parallelize(fffb, ParallelUnit::GPUBlock, OutputRaceStrategy::Atomics)
                    .parallelize(m, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);
#else  
  IndexStmt stmt = Out.getAssignment().concretize()
                    .reorder({p,q,r,s,t,u,c,m})
                    .fuse(p,q,f)
                    .bound(f,fb,P*Q,BoundType::MaxExact)
                    .parallelize(fb, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces)
                    .parallelize(m, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);
#endif


  compute func = (compute)compileThenFunc(stmt);

  func(O.getTacoTensorT(), I.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  func(O.getTacoTensorT(), I.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  func(O.getTacoTensorT(), I.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  
  auto start = Clock::now();
  for (int i = 0; i<10; i++) func(O.getTacoTensorT(), I.getTacoTensorT(), I.getTacoTensorT(), F.getTacoTensorT());
  auto end = Clock::now();
  cout << compute_clock(end, start)/10 <<  " ms" <<endl;

  float *val = (float*)(O.getTacoTensorT()->vals);
  //for (int i = 0; i<30; i++) {
  //  cout << val[i*M]/(rep*2) << endl;
  //}
  //float sum = 0;
  //for (int i = 0; i<crds.size(); i++) {
  //  sum+= val[i*M]/(rep*2);
  //}
  //cout << sum << endl;
}
