#include <string>
#include <vector>
#include "taco.h"
using namespace std;

// getPermutation(NCHW, NHWC) = {0,2,3,1}
vector<int> getPermutation(string before, string after) {
  vector<int> perm;
  for (int i = 0; i < after.size(); i++) {
    string after_i = after.charAt(i)+"";
    size_t found = before.find(after_i);
    perm.push_back(found);
  }
  return perm;
}

// DSDD -> {Dense, Compressed, Dense, Dense}
vector<ModeFormatPack> getModeFormat(string desc) {
  vector<ModeFormatPack> result;
  for (int i = 0; i<desc.size(); i++) {
    string lvlfmt = desc.charAt(i)+"";
    if (lvlfmt == "U" || lvlfmt == "D") 
      result.push_back(Dense);
    else if (lvlfmt == "C" || lvlfmt == "S")
      result.push_back(Sparse);
  }
  return result;
}


