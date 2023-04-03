#include "taco_factory.hpp"


//O[N,M,P,Q]
//I[N,C,H,W]
//F[M,C,R,S]
//O(n,mo,mi,p,q) = I(n,co,ci,p*STR+r,q*STR+s) * F(mo,mi,co,ci,r,s); 
IndexStmt conv2d_direct_tile(int N,
                           int H,  
                           int W,  
                           int R, 
                           int S, 
                           int STR, 
                           int PAD, 
                           int C,   
                           int M,   
                           int C_inner,
                           int M_inner,
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
 
  // Tiled Shape
  int CO = (C+C_inner-1)/C_inner; 
  int MO = (M+M_inner-1)/M_inner;
  int CI = C_inner;
  int MI = M_inner;

  // Declare TACO Tensors
  Tensor<float> Input("In", {N,CO,CI,H_PAD,W_PAD}, input_format); 
  Tensor<float> Output("Out", {N,MO,MI,P,Q}, output_format);
  Tensor<float> Filter("F", {MO,MI,CO,CI,R,S}, filter_format);

  Output(n,mo,mi,p,q) = Input(n,co,ci,p*STR+r,q*STR+s) * Filter(mo,mi,co,ci,r,s); 

  IndexStmt stmt = Output.getAssignment().concretize();
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
      else if (fuseVar == mi) { boundDim *= MI;}
      else if (fuseVar == mo) { boundDim *= MO;}
      else if (fuseVar == ci) { boundDim *= CI;}
      else if (fuseVar == co) { boundDim *= CO;}
      else { assert(false); }
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

