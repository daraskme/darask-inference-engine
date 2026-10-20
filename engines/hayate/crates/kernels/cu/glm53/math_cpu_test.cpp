// CPU validation of the arithmetic shared with the CUDA kernels.
#include "math.cuh"
#include <cassert>
#include <cstdio>
void kda_step(float* state,const float* q,const float* k,const float* v,const float* g,float beta,float* out,unsigned dim) {
  for (unsigned j=0; j<dim; ++j) out[j]=glm53::kda_column(state,q,k,v[j],g,beta,dim,j);
}
int main() {
  assert(glm53::fp8(0)==0);assert(glm53::fp8(1)==1.f/512);
  assert(glm53::fp8(0x7e)==448);assert(glm53::fp8(0xfe)==-448);
  assert(glm53::forget(0,0,0)==-2.5f);
  float logits[24]={},base[24]={},scale[3]={1,1,1},map[24];
  glm53::hc_weights(map,logits,base,scale);
  for(unsigned i=0;i<4;++i) {
    assert(fabsf(map[i]-0.500001f)<1.e-7f);assert(map[4+i]==1);
    float sum=0;for(unsigned j=0;j<4;++j)sum+=map[8+i*4+j];
    assert(fabsf(sum-1)<2.e-6f);
  }
  float state[4]={},q[2]={1,0},k[2]={0,1},v[2]={2,-1},g[2]={0,0},out[2];
  kda_step(state,q,k,v,g,0.5f,out,2);
  assert(state[0]==0 && state[1]==0 && state[2]==1 && state[3]==-0.5f);
  assert(out[0]==0 && out[1]==0);
  kda_step(state,k,k,v,g,1,out,2);
  assert(fabsf(out[0]-2/sqrtf(2))<1.e-6f);assert(fabsf(out[1]+1/sqrtf(2))<1.e-6f);
  puts("GLM shared math: FP8, decay, mHC and two KDA steps passed");
}
extern "C" void glm53_cpu_hc(float* out,const float* logits,const float* base,const float* scale) {
  glm53::hc_weights(out,logits,base,scale);
}
extern "C" void glm53_cpu_step(float* state,const float* q,const float* k,const float* v,const float* g,float beta,float* out,unsigned dim) {
  kda_step(state,q,k,v,g,beta,out,dim);
}

extern "C" void glm53_cpu_fp8(float* out,const float* x,const unsigned char* w,const float* scale,unsigned n,unsigned k) {
 for(unsigned row=0;row<n;++row){float sum=0;for(unsigned c=0;c<k;++c)sum+=x[c]*glm53::scaled_fp8(w[row*k+c],scale[(row/128)*((k+127)/128)+c/128]);out[row]=glm53::bf16(sum);}
}
extern "C" void glm53_cpu_conv(float* out,float* history,const float* x,const float* w,unsigned cols) {
 for(unsigned c=0;c<cols;++c){float past[3],weight[4];for(unsigned j=0;j<3;++j)past[j]=history[j*cols+c];for(unsigned j=0;j<4;++j)weight[j]=w[c*4+j];float sum=glm53::conv_sum(past,x[c],weight);out[c]=glm53::bf16(sum*glm53::sigmoid(sum));for(unsigned j=0;j<2;++j)history[j*cols+c]=history[(j+1)*cols+c];history[2*cols+c]=x[c];}
}

extern "C" void glm53_cpu_nvfp4(float* out,const unsigned char* image,unsigned n,unsigned k) {
 for(unsigned r=0;r<n;++r)for(unsigned c=0;c<k;++c)out[size_t(r)*k+c]=glm53::nvfp4(image,r,c,n,k);
}
extern "C" void glm53_cpu_hadamard(float* out,const float* x,unsigned count,unsigned block) {
 for(unsigned offset=0;offset<count;offset+=block) {
   float z[256];for(unsigned c=0;c<block;++c)z[c]=x[offset+c];
   for(unsigned stride=1;stride<block;stride*=2) {
     float next[256];
     for(unsigned c=0;c<block;++c)next[c]=glm53::hadamard_value(z[c&~stride],z[c|stride],c,stride);
     for(unsigned c=0;c<block;++c)z[c]=next[c];
   }
   for(unsigned c=0;c<block;++c)out[offset+c]=glm53::bf16(z[c]*(1.f/sqrtf(float(block))));
 }
}
extern "C" void glm53_cpu_dense(float* out,const float* x,const unsigned char* w,const float* scales,unsigned n,unsigned k,unsigned format) {
  for(unsigned r=0;r<n;++r){double sum=0;for(unsigned c=0;c<k;++c){
    float v=format==3?glm53::nvfp4(w,r,c,n,k):glm53::fp8(w[size_t(r)*k+c])*(format==1?scales[(r/128)*((k+127)/128)+c/128]:*reinterpret_cast<const float*>(w+size_t(n)*k));
    sum+=double(x[c])*glm53::bf16(v);
  }out[r]=sum;}
}
extern "C" float glm53_cpu_swiglu(float gate,float up) {return glm53::swiglu(gate,up);}
extern "C" float glm53_cpu_reduce(const float* values,const int* ids,const float* weights,float shared) {
  int order[8];glm53::route_order(order,ids);float sum=0;
  for(unsigned g=0;g<8;++g){unsigned i=order[g];sum=glm53::expert_add(sum,values[i],weights[i]);}
  return glm53::bf16(sum+shared);
}
