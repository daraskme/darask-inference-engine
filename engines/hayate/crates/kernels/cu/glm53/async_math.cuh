#pragma once
#ifdef __CUDACC__
#define GLM_ASYNC_HD __host__ __device__
#else
#define GLM_ASYNC_HD
#endif
namespace glm53 {
struct AsyncGroup { const void* image; unsigned count, assignments[4]; };
GLM_ASYNC_HD inline unsigned async_groups(AsyncGroup* groups,const int* ids,const void* const* table,unsigned rows) {
  unsigned n=0;
  for(unsigned expert=0;expert<288;++expert) {
    AsyncGroup group{};group.image=table[expert];
    for(unsigned a=0;a<rows*8;++a)if(ids[a]==int(expert)&&group.count<4)group.assignments[group.count++]=a;
    if(group.count)groups[n++]=group;
  }
  return n;
}
GLM_ASYNC_HD inline void sort_pair(unsigned* values,unsigned i,unsigned stride,unsigned width) {
  unsigned j=i^stride;
  if(j>i){unsigned a=values[i],b=values[j];if((a>b)==((i&width)==0)){values[i]=b;values[j]=a;}}
}
GLM_ASYNC_HD inline unsigned merge_queries(unsigned* ids,unsigned* counts,const unsigned* sorted,unsigned rows) {
  unsigned positions[4]={},n=0;
  for(;;) {
    unsigned best=~0u;
    for(unsigned r=0;r<rows;++r){unsigned id=sorted[r*4096+positions[r]];if(id<best)best=id;}
    if(best==~0u)break;
    ids[n++]=best;
    for(unsigned r=0;r<rows;++r)if(sorted[r*4096+positions[r]]==best)++positions[r];
  }
  counts[0]=n;for(unsigned r=0;r<rows;++r)counts[1+r]=positions[r];
  return n;
}
}
#undef GLM_ASYNC_HD
