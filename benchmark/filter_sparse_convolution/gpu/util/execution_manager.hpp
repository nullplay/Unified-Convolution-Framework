#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
#include <omp.h>
#include <regex>
#include "taco.h"
#include "taco/ir/ir.h"
#include "taco/codegen/module.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"

using namespace taco;
using namespace std;
void* lib_handle = NULL;

typedef int (*compute_args2)(taco_tensor_t* arg1, taco_tensor_t* arg2);
typedef int (*compute_args3)(taco_tensor_t* arg1, taco_tensor_t* arg2, taco_tensor_t* arg3);
typedef int (*compute_args4)(taco_tensor_t* arg1, taco_tensor_t* arg2, taco_tensor_t* arg3, taco_tensor_t* arg4);


//////////////////
// GPU CodeGen  //
//////////////////
void codegenTACOGPU(IndexStmt stmt, string name, bool is_stdout) {
  string suffix = "";
  string prefix_path = "";
  string header = "#ifndef TACO_C_HEADERS\n#define TACO_C_HEADERS\n#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <math.h>\n#include <thrust/complex.h>\n#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))\n#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)\n#ifndef TACO_TENSOR_T_DEFINED\n#define TACO_TENSOR_T_DEFINED\ntypedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;\ntypedef struct {\n  int32_t      order;         // tensor order (number of modes)\n  int32_t*     dimensions;    // tensor dimensions\n  int32_t      csize;         // component size\n  int32_t*     mode_ordering; // mode storage ordering\n  taco_mode_t* mode_types;    // mode storage types\n  uint8_t***   indices;       // tensor index data (per mode)\n  uint8_t*     vals;          // tensor values\n  int32_t      vals_size;     // values array size\n} taco_tensor_t;\n#endif\n#endif\n\n\n#ifndef TACO_GENERATED_compute\n#define TACO_GENERATED_compute\nint compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B);\n#endif";
  //string shim = "\nextern \"C\" {\n  int _shim_compute(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B);\n}\n\nint _shim_compute(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B) {\n  return compute(C,A,B);\n}"; 

  string shim = "#include \""+ name + ".h\"" + "\n\nextern \"C\" {\n  int _shim_compute(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B);\n}\n\nint _shim_compute(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B) {\n  float ttot_ms = 0;\n  for (int32_t n = 0; n < 100; n++) {\n    float tot_ms;\n    cudaEvent_t event1, event2;\n    cudaEventCreate(&event1);\n    cudaEventCreate(&event2);\n    cudaDeviceSynchronize();\n    cudaEventRecord(event1,0);\n\n    compute(C,A,B);\n\n    cudaEventRecord(event2,0);\n    cudaEventSynchronize(event1);\n    cudaEventSynchronize(event2);\n    cudaEventElapsedTime(&tot_ms, event1, event2);\n    ttot_ms += tot_ms;\n  }\n  printf(\"%f ms\\n\", ttot_ms/100);\n  return 0;\n}";

  stringstream ss;
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(ss, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);
  if (is_stdout) { cout << ss.str() << endl; return; } 

  // Source Code 
  ofstream source("./code/" + name + ".cu");
  string generated_code = std::regex_replace(ss.str(), std::regex("int32_t &"), "int32_t "); 
  generated_code = std::regex_replace(generated_code, std::regex("Out_vals\\[pOut\\] = 0\\.0;"), ""); 
  source << generated_code ;
  source.close();

  // Header Code
  ofstream header_file;
  header_file.open("./code/"+ name + ".h");
  header_file << header;
  header_file.close();

  // shim Code
  stringstream shims;
  ofstream shims_file;
  shims_file.open("./code/"+ name +"_shim.cpp");
  shims_file << "#include \"" << name << ".h\"\n";
  shims_file << shim;
  shims_file.close();

  return;
}

void compileKernelGPU(string name) {
  string cflag = " -w -O3 -Xcompiler \"-fPIC -shared -ffast-math -O3\" --generate-code arch=compute_70,code=sm_70 ";
  string sourcename = "./code/" + name +".cu ";
  string shimname = "./code/"+ name +"_shim.cpp";
  string objectname = "./code/"+ name + ".so";
  string compile_command = "nvcc" + cflag + sourcename + shimname + " -o " + objectname + " -lm";
  int err = system(compile_command.c_str());
  assert(err == 0);
}


void* loadLibraryGPU(string name) {
  // Call shared library
  if (lib_handle) {dlclose(lib_handle);}
  string library_name = "./code/" + name + ".so";
  lib_handle = dlopen(library_name.c_str(), RTLD_NOW|RTLD_LOCAL);
  if (!lib_handle) {
    fprintf(stderr, "%s\n", dlerror());
    cout << "LoadLibrary - ERR" << endl;
    return nullptr;
  }

  return dlsym(lib_handle, "_shim_compute");
}
