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
  set_CUDA_codegen_enabled(false);
  set_CUDA_unified_memory_enabled(false);

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
  string OutputLayout(argv[11]);
  string InputLayout(argv[12]);
  string FilterLayout(argv[13]);
  string FilterLvlFormat(argv[14]);
  string LoopOrder(argv[15]);
  string ParallelVars(argv[16]);
  int NTHREADS = atoi(argv[17]); 
  NTHREADS = omp_get_max_threads()/2; //to avoid hyperthreading
  int CHUNK    = atoi(argv[18]);
  string isVerify(argv[19]);
  string label(argv[20]);

  //Padded Input and Output Shape
  int HPAD = H+2*PAD;
  int WPAD = W+2*PAD;
  int P = (H+2*PAD-R)/STR+1;
  int Q = (W+2*PAD-S)/STR+1;

  IndexStmt stmt;
  compute_args3 func;
 
  // Prepare Format 
  Format FilterFormat = Format(getModeFormat(FilterLvlFormat), getPermutation("MCRS",FilterLayout));
  Format InputFormat  = MakeDense(getPermutation("NCHW",InputLayout));
  Format OutputFormat = MakeDense(getPermutation("NMPQ",OutputLayout));

  /// Defining Algorithm 
  stmt = conv2d_direct(N,H, W, R, S, STR, PAD, C, M,                                               //Algorithm
                       InputFormat, FilterFormat, OutputFormat,                                    //Data Representation
                       strToIndexVars(LoopOrder), strToIndexVars(ParallelVars), NTHREADS, CHUNK);  //Schedule

  /// Codegen -> Compile -> Load 
  codegenTACO(stmt, label, false);
  compileKernel(label);
  func = (compute_args3)loadLibrary(label);


  /// Prepare Data
  Tensor<float> FilterData, InputData, InputData_padded, OutputData;
  if (filename[filename.size()-1] == 's')       //.tns file
    readMCRS4DFromTNS(FilterData, filename, FilterFormat, R==1);
  else if (filename[filename.size()-1] == 'x')  //.smtx file
    readMCRS4DFromSMTX(FilterData, filename, FilterFormat, {M,C,R,S});
  
  fillDense(InputData,  1, InputFormat, {N,C,H,W});
  fillDense(OutputData, 0, OutputFormat, {N,M,P,Q});

  /// Padding Input to avoid boundary checking
  if (PAD == 0) {InputData_padded = InputData;}
  else {
    fillDense(InputData_padded,  0, InputFormat, {N,C,HPAD,WPAD});
    if (isVerify=="verify") {
      float* in = (float*)(InputData.getTacoTensorT()->vals);
      float* inpad = (float*)(InputData_padded.getTacoTensorT()->vals);
      for (int c = 0; c<C; c++)  
        for (int h = 0; h<H; h++) 
          for (int w=0; w<W; w++) 
            inpad[c*HPAD*WPAD + (h+PAD)*WPAD + (PAD)+w] = in[c*H*W + h*W+w];
    }
  }
  
  /// Benchmark - median in (ms)
  vector<float> elapsed;
  for (int r=0; r<20; r++) {
    auto t1 = Clock::now();
    for(int k=0; k<30; k++) 
      func(OutputData.getTacoTensorT(), InputData_padded.getTacoTensorT(), FilterData.getTacoTensorT());
    auto t2 = Clock::now();
    elapsed.push_back(compute_clock(t2,t1)/30.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::sort(elapsed.begin(), elapsed.end());
  cout << elapsed[elapsed.size()/2] << " ms" << endl;

  /// Verify
  if (isVerify=="verify") {
    Tensor<float> FilterDense;
    if (filename[filename.size()-1] == 's') { //tns
      readMCRS4DFromTNS(FilterDense, filename, MakeDense(4), R==1);
    }
    else if (filename[filename.size()-1] == 'x') { //mtx
      readMCRS4DFromSMTX(FilterDense, filename, MakeDense(4), {M,C,R,S});
    }
    vector<float> refv(N*M*P*Q), inv(N*C*H*W,1);
    float* ref = refv.data();
    float* in  = inv.data();
    float* flt = (float*)(FilterDense.getTacoTensorT()->vals);
    float* inpad = (float*)(InputData_padded.getTacoTensorT()->vals);
    for (int n=0;n<N;n++){
      for (int m=0;m<M;m++){
        for (int c=0;c<C;c++) {
          for (int r=0;r<R;r++){
            for (int s=0;s<S;s++){
              for (int p=0;p<P;p++){
                for (int q=0;q<Q;q++){
                  int h = p+r-PAD;
                  int w = q+s-PAD;
                  if (0<=h && h<H && 0<=w&&w<W){
                    ref[n*M*P*Q + m*P*Q + p*Q + q] += in[n*C*H*W + c*H*W + h*W + w] * flt[m*C*R*S+c*R*S+r*S+s];
                  } 
                }}}}}}}

      
    float* out = (float*)(OutputData.getTacoTensorT()->vals);
    for (int i=0; i<N*M*P*Q; i++) {
      if (ref[i] != out[i]) { cout << "WrongResult - " << i << " " << ref[i] << " " << out[i] << endl; break; }
    }
  }


  //NCHW -> NHWC Test
  //ResNet50 Test

}
