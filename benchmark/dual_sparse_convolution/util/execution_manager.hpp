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
// CPU CodeGen  //
//////////////////
void codegenTACO(IndexStmt stmt, string name, bool is_stdout) {
  stringstream ss;
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(ss, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);
  if (is_stdout) { cout << ss.str() << endl; return; } 

  // Write C code into file
  string full_path = "./code/" + name + ".c";
  ofstream source(full_path);
  source << (ss.str()) ;
  source.close();

  return;
}

void compileKernel(string name) {
#if defined(ICC)
  string cxx = "icc";
  string cflags = " -march=native -mtune=native -O3 -qopenmp -ffast-math -fPIC -shared";
#elif defined(GCC)
  string cxx = "gcc"; 
  string cflags = " -march=native -mtune=native -O3 -fopenmp -ffast-math -fPIC -shared";
#endif
  string cfile_name = " ./code/" + name + ".c";
  string library_name = " -o ./code/" + name + ".so";
  string link = " -lm";

  string compile_command = cxx + cflags + cfile_name + library_name + link;
  int ret = system(compile_command.c_str());
  assert(ret == 0);
}

void* loadLibrary(string name) {
  // Call shared library
  if (lib_handle) {dlclose(lib_handle);}
  string library_name = "./code/" + name + ".so";
  lib_handle = dlopen(library_name.c_str(), RTLD_NOW|RTLD_LOCAL);
  if (!lib_handle) {
    fprintf(stderr, "%s\n", dlerror());
    cout << "LoadLibrary - ERR" << endl;
    return nullptr;
  }

  return dlsym(lib_handle, "compute");
}


