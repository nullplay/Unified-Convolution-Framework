#include <iostream>
#include <thread>
#include <omp.h>
#include "util/time.hpp"
#include "util/tensor.hpp"
#include "util/execution_manager.hpp"
#include "include/parser.hpp"
#include "include/2dconv_direct.hpp"
using namespace taco;
using namespace std;

#define S_ Sparse
#define D_ Dense

int main(int argc, char* argv[]) {
  set_CUDA_codegen_enabled(true);
  set_CUDA_unified_memory_enabled(true);

  string filename(argv[1]);
  int N   = atoi(argv[2]);
  int H   = atoi(argv[3]);
  int W   = atoi(argv[4]);
  int R   = atoi(argv[5]);
  int S   = atoi(argv[6]);
  int STR = atoi(argv[7]);
  int PAD = atoi(argv[8]);
  int C   = atoi(argv[9]);
  int M   = atoi(argv[10]);
  int input_sparsity = atoi(argv[11]);
  string OutputLayout(argv[12]);
  string InputLayout(argv[13]);
  string InputLvlFormat(argv[14]);
  string FilterLayout(argv[15]);
  string FilterLvlFormat(argv[16]);
  string LoopOrder = "nmpqrsc";
  string isVerify(argv[17]); // "verify" : verifying correctness
  string label(argv[18]);    // name for file of generated code.

  //Padded Input and Output Shape
  int HPAD = H+2*PAD;
  int WPAD = W+2*PAD;
  int P = (H+2*PAD-R)/STR+1;
  int Q = (W+2*PAD-S)/STR+1;

  IndexStmt stmt;
  compute_args3 func;
 
  // Prepare Format 
  Format FilterFormat = Format(getModeFormat(FilterLvlFormat), getPermutation("MCRS",FilterLayout));
  Format InputFormat  = Format(getModeFormat(InputLvlFormat),  getPermutation("NCHW",InputLayout));
  Format OutputFormat = MakeDense(getPermutation("NMPQ",OutputLayout));

  /// Defining Algorithm 
  stmt = conv2d_direct_gpu(N, H, W, R, S, STR, PAD, C, M,            //Algorithm
                           InputFormat, FilterFormat, OutputFormat,  //Data Representation
                           strToIndexVars(LoopOrder));               //Schedule

  /// Codegen -> Compile -> Load 
  codegenTACOGPU(stmt, label, false);
  compileKernelGPU(label);
  func = (compute_args3)loadLibraryGPU(label);

  /// Prepare Data
  Tensor<float> FilterData, InputData, InputData_sparse, OutputData, FilterDense;
  if (filename[filename.size()-1] == 's')       //.tns file
    readMCRS4DFromTNS(FilterData, filename, FilterFormat, R==1);
  else if (filename[filename.size()-1] == 'x')  //.smtx file
    readMCRS4DFromSMTX(FilterData, filename, FilterFormat, {M,C,R,S});
  
  // For Verify in the future
  if (isVerify=="verify") readMCRS4DFromSMTX(FilterDense, filename, MakeDense(4), {M,C,R,S});

  fillSparseUniform(InputData, input_sparsity, MakeDense(4), {N,C,H,W}); //UUUU Format after ReLU
  fillDense(OutputData, 0, OutputFormat, {N,M,P,Q});

  InputData_sparse = Tensor<float>("inx", {N,C,HPAD,WPAD}, InputFormat); //Sparse Format
  float* in = (float*)(InputData.getTacoTensorT()->vals);
  for (int n=0; n<N; n++) {
    for (int c=0; c<C; c++) {
      for (int h=0; h<H; h++) {
        for (int w=0; w<W; w++) {
          float in_chw = in[n*C*H*W+c*H*W+h*W+w];
          if (in_chw != 0) {
            InputData_sparse.insert({n,c,h+PAD,w+PAD}, (float)in_chw);
          }
        }
      }
    }
  }
  InputData_sparse.pack();  
  
  // Measuring time inside func (see shim code in ./util/execution_manager.hpp)
  cout << "Filter Memory Saving: " << M*C*R*S*4/1000.0 << " -> " << FilterData.getStorage().getSizeInBytes()/1000.0 << "(KB) / "; 
  func(OutputData.getTacoTensorT(), InputData_sparse.getTacoTensorT(), FilterData.getTacoTensorT());

  /// Verify
  if (isVerify=="verify") {
    vector<float> refv(N*M*P*Q);
    float* ref = refv.data();
    float* in = (float*)(InputData.getTacoTensorT()->vals);
    float* flt = (float*)(FilterDense.getTacoTensorT()->vals);
    if (OutputLayout == "NMPQ"){
    for (int n=0;n<N;n++){
      for (int m=0;m<M;m++){
        for (int c=0;c<C;c++) {
          for (int r=0;r<R;r++){
            for (int s=0;s<S;s++){
              for (int p=0;p<P;p++){
                for (int q=0;q<Q;q++){
                  int h = STR*p+r-PAD;
                  int w = STR*q+s-PAD;
                  if (0<=h && h<H && 0<=w&&w<W){
                      ref[n*M*P*Q + m*P*Q + p*Q + q] += in[n*C*H*W + c*H*W + h*W + w] * flt[m*C*R*S+c*R*S+r*S+s];
                  } 
                }}}}}}}}

    else if (OutputLayout == "NPQM"){
    for (int n=0;n<N;n++){
      for (int m=0;m<M;m++){
        for (int c=0;c<C;c++) {
          for (int r=0;r<R;r++){
            for (int s=0;s<S;s++){
              for (int p=0;p<P;p++){
                for (int q=0;q<Q;q++){
                  int h = STR*p+r-PAD;
                  int w = STR*q+s-PAD;
                  if (0<=h && h<H && 0<=w&&w<W){
                      ref[n*P*Q*M+p*Q*M+q*M+m] += in[n*C*H*W + c*H*W + h*W + w] * flt[m*C*R*S+c*R*S+r*S+s];
                  } 
                }}}}}}}}
      
    float* out = (float*)(OutputData.getTacoTensorT()->vals);
    for (int i=0; i<N*M*P*Q; i++) {
      if (ref[i]*100 != out[i]) { cout << "WrongResult - " << i << " " << ref[i] << " " << out[i] << endl; break; }
    }
  }


  //NCHW -> NHWC Test
  //ResNet50 Test
}
