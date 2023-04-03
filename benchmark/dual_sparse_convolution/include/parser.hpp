#include <string>
#include <vector>
#include "taco_factory.hpp"
using namespace std;

// getPermutation(NCHW, NHWC) = {0,2,3,1}
vector<int> getPermutation(string before, string after) {
  vector<int> perm;
  for (int i = 0; i < after.size(); i++) {
    string after_i;
    after_i.push_back(after[i]);
    size_t found = before.find(after_i);
    perm.push_back(found);
  }
  return perm;
}

// DSDD -> {Dense, Compressed, Dense, Dense}
vector<ModeFormatPack> getModeFormat(string desc) {
  vector<ModeFormatPack> result;
  for (int i = 0; i<desc.size(); i++) {
    string lvlfmt;
    lvlfmt.push_back(desc[i]);
    if (lvlfmt == "U" || lvlfmt == "D") 
      result.push_back(Dense);
    else if (lvlfmt == "C" || lvlfmt == "S")
      result.push_back(Sparse);
  }
  return result;
}


vector<IndexVar> strToIndexVars(string arg) {
  vector<IndexVar> result;
  for (int i = 0; i<arg.size(); i++) {
    string ch;
    ch.push_back(arg[i]);
    if (ch == "p") result.push_back(p);
    if (ch == "q") result.push_back(q);
    if (ch == "r") result.push_back(r);
    if (ch == "s") result.push_back(s);
    if (ch == "c") result.push_back(c);
    if (ch == "n") result.push_back(n);
    if (ch == "m") result.push_back(m);
  }
  return result;
}

// nMmCcrspq
vector<IndexVar> strToTiledIndexVars(string arg) {
  vector<IndexVar> result;
  for (int i = 0; i<arg.size(); i++) {
    string ch;
    ch.push_back(arg[i]);
    if (ch == "p") result.push_back(p);
    if (ch == "q") result.push_back(q);
    if (ch == "r") result.push_back(r);
    if (ch == "s") result.push_back(s);
    if (ch == "c") result.push_back(ci);
    if (ch == "C") result.push_back(co);
    if (ch == "n") result.push_back(n);
    if (ch == "m") result.push_back(mi);
    if (ch == "M") result.push_back(mo);
  }
  return result;
}
