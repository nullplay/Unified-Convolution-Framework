#include "taco_factory.hpp"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation.h"

//O[N,M,P,Q]
//I[N,C,H,W]
//F[M,C,R,S]
//O(n,m,p,q) = I(n,c,p+r,q+s) * F(m,c,r,s)
IndexStmt conv2d_direct_gpu(int N,
                           int H,  
                           int W,  
                           int R, 
                           int S, 
                           int STR, 
                           int PAD, 
                           int C,   
                           int M,   
                           Format input_format,
                           Format filter_format,
                           Format output_format,
                           vector<IndexVar> loop_order
                           ) {
  // Padding input to avoid boundary check
  int H_PAD = H+2*PAD;
  int W_PAD = W+2*PAD;

  // Output Shape
  int P = (H+2*PAD-R)/STR+1;
  int Q = (W+2*PAD-S)/STR+1;
  
  // Declare TACO Tensors
  Tensor<float> Input("In", {N,C,H_PAD,W_PAD}, input_format); 
  Tensor<float> Output("Out", {N,M,P,Q}, output_format);
  Tensor<float> Filter("F", {M,C,R,S}, filter_format);

  Output(n,m,p,q) = Input(n,c,p*STR+r,q*STR+s) * Filter(m,c,r,s);
 
  IndexStmt stmt = Output.getAssignment().concretize();

  // Fixed Schedule
  stmt = stmt.reorder({n,m,p,q,r,s,c});
  stmt = stmt.fuse(m,p,f);
  stmt = stmt.fuse(f,q,ff);
  stmt = stmt.bound(ff,ffb,P*Q*M,BoundType::MaxExact);
  stmt = stmt.split(ffb, io, ii, 256);
  stmt = stmt.parallelize(io, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces);
  stmt = stmt.parallelize(ii, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);

  return stmt;
}


