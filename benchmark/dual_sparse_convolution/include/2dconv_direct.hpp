#include "taco_factory.hpp"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation.h"




//O[N,M,P,Q]
//I[N,C,H,W]
//F[M,C,R,S]
//O(n,m,p,q) = I(n,c,p+r,q+s) * F(m,c,r,s)
IndexStmt conv2d_direct(int N,
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
                       vector<IndexVar> loop_order,
                       vector<IndexVar> parallel_vars,
                       int n_threads,
                       int chunk_size
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

  // First result
  //stmt = stmt.reorder({m,c,r,s,p,q})
  //           .parallelize(m, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);

  stmt = stmt.reorder(loop_order);
  //TensorVar workspace("w", Type(Float32, {P,Q}), Format({Dense,Dense}));
  //stmt = stmt.precompute(precomputed, {p,q}, {p,q}, workspace);
  

  // 2. Decide loop fusion
  if (parallel_vars.size() == 1) {
    stmt = stmt.parallelize(parallel_vars[0], ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  } else {
    vector<IndexVar> orderedFuse;
    vector<int> fuseLvls;
    for (int currLvl = 0; currLvl<loop_order.size(); currLvl++) {
      IndexVar loopVar = loop_order[currLvl];
      if (find(parallel_vars.begin(), parallel_vars.end(), loopVar) != parallel_vars.end()) { 
        orderedFuse.push_back(loopVar);
        fuseLvls.push_back(currLvl);
      }
    }
    
    // if try to fuse separate lvls
    for (int idx = 1; idx < fuseLvls.size(); idx++) {
      assert(fuseLvls[idx] == fuseLvls[idx-1]+1);
    }

    int boundDim = 1;
    for (auto fuseVar : orderedFuse) {
      if (fuseVar == n) { boundDim *= N;}
      else if (fuseVar == p) { boundDim *= P;}
      else if (fuseVar == q) { boundDim *= Q;}
      else if (fuseVar == m) { boundDim *= M;}
    }

    if (orderedFuse.size() == 2) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f); 
      stmt = stmt.bound(f,fb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(fb,ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    } else if (orderedFuse.size() == 3) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f);
      stmt = stmt.fuse(f, orderedFuse[2], ff);
      stmt = stmt.bound(ff,ffb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(ffb,ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    } else {
      assert(false);
    }
  }

  omp_set_num_threads(n_threads);
  omp_set_schedule(omp_sched_dynamic,chunk_size);
  //omp_set_schedule(omp_sched_static,chunk_size);

  return stmt;
}


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
                           vector<IndexVar> loop_order,
                           vector<IndexVar> parallel_vars_blocks, 
                           vector<IndexVar> parallel_vars_threads
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
  stmt = stmt.reorder(loop_order);
  
  stmt = stmt.fuse(m,p,f);
  stmt = stmt.fuse(f,q,ff);
  stmt = stmt.bound(ff,ffb,P*Q*M,BoundType::MaxExact);
  stmt = stmt.split(ffb, io, ii, 256);
  stmt = stmt.parallelize(io, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces);
  stmt = stmt.parallelize(ii, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);

  /*
  // 2. Decide loop fusion and parallelize GPUBlock
  if (parallel_vars_blocks.size() == 1) {
    stmt = stmt.parallelize(parallel_vars_blocks[0], ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces);
  } else {
    vector<IndexVar> orderedFuse;
    vector<int> fuseLvls;
    for (int currLvl = 0; currLvl<loop_order.size(); currLvl++) {
      IndexVar loopVar = loop_order[currLvl];
      if (find(parallel_vars_blocks.begin(), parallel_vars_blocks.end(), loopVar) != parallel_vars_blocks.end()) { 
        orderedFuse.push_back(loopVar);
        fuseLvls.push_back(currLvl);
      }
    }
    
    // if try to fuse separate lvls
    for (int idx = 1; idx < fuseLvls.size(); idx++) {
      assert(fuseLvls[idx] == fuseLvls[idx-1]+1);
    }

    int boundDim = 1;
    for (auto fuseVar : orderedFuse) {
      if (fuseVar == n) { boundDim *= N;}
      else if (fuseVar == p) { boundDim *= P;}
      else if (fuseVar == q) { boundDim *= Q;}
      else if (fuseVar == m) { boundDim *= M;}
    }

    if (orderedFuse.size() == 2) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f); 
      stmt = stmt.bound(f,fb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(fb,ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces);
    } else if (orderedFuse.size() == 3) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f);
      stmt = stmt.fuse(f, orderedFuse[2], ff);
      stmt = stmt.bound(ff,ffb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(ffb,ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces);
    } else {
      assert(false);
    }
  }


  // 3. Decide loop fusion for parallelizing GPUThread
  auto strategy = find(parallel_vars_threads.begin(), parallel_vars_threads.end(), c) != parallel_vars_threads.end() ? OutputRaceStrategy::Atomics : OutputRaceStrategy::NoRaces;
  if (parallel_vars_threads.size() == 1) {
    stmt = stmt.parallelize(parallel_vars_threads[0], ParallelUnit::GPUThread, strategy);
  } else {
    vector<IndexVar> orderedFuse;
    vector<int> fuseLvls;
    for (int currLvl = 0; currLvl<loop_order.size(); currLvl++) {
      IndexVar loopVar = loop_order[currLvl];
      if (find(parallel_vars_threads.begin(), parallel_vars_threads.end(), loopVar) != parallel_vars_threads.end()) { 
        orderedFuse.push_back(loopVar);
        fuseLvls.push_back(currLvl);
      }
    }
    
    // if try to fuse separate lvls
    for (int idx = 1; idx < fuseLvls.size(); idx++) {
      assert(fuseLvls[idx] == fuseLvls[idx-1]+1);
    }

    int boundDim = 1;
    for (auto fuseVar : orderedFuse) {
      if (fuseVar == n) { boundDim *= N;}
      else if (fuseVar == p) { boundDim *= P;}
      else if (fuseVar == q) { boundDim *= Q;}
      else if (fuseVar == m) { boundDim *= M;}
    }

    if (orderedFuse.size() == 2) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f); 
      stmt = stmt.bound(f,fb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(fb,ParallelUnit::GPUThread, strategy);
    } else if (orderedFuse.size() == 3) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f);
      stmt = stmt.fuse(f, orderedFuse[2], ff);
      stmt = stmt.bound(ff,ffb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(ffb,ParallelUnit::GPUThread, strategy);
    } else {
      assert(false);
    }
  }
*/

  return stmt;
}



//O[N,M,P,Q]
//I[N,C,H,W]
//F[M,C,R,S]
//O(n,m,p,q) = I(n,c,p+r,q+s) * F(m,c,r,s)
IndexStmt conv2d_direct_nopad(int N,
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
                             vector<IndexVar> loop_order,
                             vector<IndexVar> parallel_vars,
                             int n_threads,
                             int chunk_size
                             ) {
  // Output Shape
  int P = (H+2*PAD-R)/STR+1;
  int Q = (W+2*PAD-S)/STR+1;
  
  // Declare TACO Tensors
  Tensor<float> Input("In", {N,C,H,W}, input_format); 
  Tensor<float> Output("Out", {N,M,P,Q}, output_format);
  Tensor<float> Filter("F", {M,C,R,S}, filter_format);

  Output(n,m,p,q) = Input(n,c,p*STR+r-PAD,q*STR+s-PAD) * Filter(m,c,r,s); 

  IndexStmt stmt = Output.getAssignment().concretize();

  // First result
  //stmt = stmt.reorder({m,c,r,s,p,q})
  //           .parallelize(m, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);

  stmt = stmt.reorder(loop_order);
  
  // 2. Decide loop fusion
  if (parallel_vars.size() == 1) {
    stmt = stmt.parallelize(parallel_vars[0], ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  } else {
    vector<IndexVar> orderedFuse;
    vector<int> fuseLvls;
    for (int currLvl = 0; currLvl<loop_order.size(); currLvl++) {
      IndexVar loopVar = loop_order[currLvl];
      if (find(parallel_vars.begin(), parallel_vars.end(), loopVar) != parallel_vars.end()) { 
        orderedFuse.push_back(loopVar);
        fuseLvls.push_back(currLvl);
      }
    }
    
    // if try to fuse separate lvls
    for (int idx = 1; idx < fuseLvls.size(); idx++) {
      assert(fuseLvls[idx] == fuseLvls[idx-1]+1);
    }

    int boundDim = 1;
    for (auto fuseVar : orderedFuse) {
      if (fuseVar == n) { boundDim *= N;}
      else if (fuseVar == p) { boundDim *= P;}
      else if (fuseVar == q) { boundDim *= Q;}
      else if (fuseVar == m) { boundDim *= M;}
    }

    if (orderedFuse.size() == 2) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f); 
      stmt = stmt.bound(f,fb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(fb,ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    } else if (orderedFuse.size() == 3) {
      stmt = stmt.fuse(orderedFuse[0], orderedFuse[1], f);
      stmt = stmt.fuse(f, orderedFuse[2], ff);
      stmt = stmt.bound(ff,ffb,boundDim, BoundType::MaxExact);
      stmt = stmt.parallelize(ffb,ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    } else {
      assert(false);
    }
  }

  omp_set_num_threads(n_threads);
  omp_set_schedule(omp_sched_dynamic,chunk_size);

  return stmt;
}


