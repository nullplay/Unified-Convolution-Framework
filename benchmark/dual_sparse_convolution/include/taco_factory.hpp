#pragma once 
#include "taco.h"
#include "taco/index_notation/transformations.h"
using namespace taco;
using namespace std;

IndexVar p("p"), q("q"), r("r"), s("s"), c("c"), n("n"), m("m");
IndexVar pp("pp"), qp("qp"), rp("rp"), sp("sp"), cp("cp"), np("np"), mp("mp");
IndexVar pb("pb"), qb("qb"), rb("rb"), sb("sb"), cb("cb"), nb("nb"), mb("mb");
IndexVar po("po"), pi("pi"), qo("qo"), qi("qi"), ro("ro"), ri("ri"), mo("mo");
IndexVar so("so"), si("si"), co("co"), ci("ci"), no("no"), ni("ni"), mi("mi");
IndexVar f("f"), ff("ff"), fff("fff");
IndexVar fb("fb"), ffb("ffb"), fffb("fffb");

IndexVar i("i"), k("k"), j("j");
IndexVar io("io"), ii("ii"), ko("ko"), ki("ki"), jo("jo"), ji("ji");


Format MakeDense(vector<int> perm) {
  vector<ModeFormatPack> f(perm.size(), Dense);
  return Format(f, perm);
}
Format MakeDense(int dim) {
  vector<ModeFormatPack> f(dim, Dense);
  return Format(f);
}


/* Sparse Filter 4D */
// N : out_channel
// C : in_channel
// H : filter_height
// W : filter_weight

// All dimensions are sparse (Unstructured)
Format nchw() {
  return Format({Sparse,Sparse,Sparse,Sparse});
}

// 1D Dense Block
Format nchW() {
  return Format({Sparse,Sparse,Sparse,Dense});
}

// 2D Dense Block
Format ncHW() {
  return Format({Sparse,Sparse,Dense,Dense});
}

// 3D Dense Block
Format nCHW() {
  return Format({Sparse,Dense,Dense,Dense});
}

