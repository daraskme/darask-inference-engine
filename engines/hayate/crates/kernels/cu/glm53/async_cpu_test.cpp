#include "async_math.cuh"
#include <vector>
extern "C" unsigned glm53_async_sparse(unsigned* ids,unsigned* counts,unsigned* sorted,const int* selected,unsigned rows) {
  for(unsigned r=0;r<rows;++r) {
    unsigned* values=sorted+r*4096;
    for(unsigned i=0;i<4096;++i)values[i]=i<2051&&selected[r*2051+i]>=0?unsigned(selected[r*2051+i]):~0u;
    for(unsigned width=2;width<=4096;width*=2)for(unsigned stride=width/2;stride;stride/=2)
      for(unsigned i=0;i<4096;++i)glm53::sort_pair(values,i,stride,width);
  }
  return glm53::merge_queries(ids,counts,sorted,rows);
}
extern "C" unsigned glm53_async_groups(unsigned* descriptors,const int* ids,unsigned rows) {
  const void* table[288]={};
  glm53::AsyncGroup groups[32];
  unsigned n=glm53::async_groups(groups,ids,table,rows);
  for(unsigned g=0;g<n;++g){descriptors[g*5]=groups[g].count;for(unsigned j=0;j<4;++j)descriptors[g*5+1+j]=groups[g].assignments[j];}
  return n;
}
