#pragma once
#include <cmath>
#include <cstddef>
#ifdef __CUDACC__
#define GLM_HD __host__ __device__
#else
#define GLM_HD
#endif
namespace glm53 {
GLM_HD inline float hadamard_value(float a,float b,unsigned lane,unsigned stride) {
  return lane&stride?a-b:a+b;
}
GLM_HD inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
GLM_HD inline float forget(float projection, float bias, float log_a) {
  return -5.f * sigmoid(expf(log_a) * (projection + bias));
}
GLM_HD inline float fp8(unsigned char bits) {
  unsigned e = (bits >> 3) & 15, m = bits & 7;
  if (e == 15 && m == 7) return NAN;
  float v = e ? ldexpf(1.f + m / 8.f, int(e) - 7) : ldexpf(float(m), -9);
  return bits & 128 ? -v : v;
}
GLM_HD inline float bf16(float x) {
#ifdef __CUDA_ARCH__
  return __bfloat162float(__float2bfloat16_rn(x));
#else
  union { float value; unsigned bits; } u;
  u.value=x; u.bits=(u.bits+0x7fff+((u.bits>>16)&1))&0xffff0000; return u.value;
#endif
}
GLM_HD inline float swiglu(float gate,float up) {
  gate=fminf(gate,10.f);up=fmaxf(-10.f,fminf(10.f,up));
  return bf16(bf16(gate*sigmoid(gate))*up);
}
GLM_HD inline void route_order(int* order,const int* ids) {
  for(int i=0;i<8;++i)order[i]=i;
  for(int i=1;i<8;++i){int v=order[i],j=i;while(j&&ids[order[j-1]]>ids[v]){order[j]=order[j-1];--j;}order[j]=v;}
}
GLM_HD inline float expert_add(float sum,float value,float weight) { return bf16(sum+bf16(value*weight)); }
GLM_HD inline float scaled_fp8(unsigned char value,float scale) { return bf16(fp8(value)*scale); }
GLM_HD inline float nvfp4(const unsigned char* image,unsigned r,unsigned c,unsigned n,unsigned k,bool scaled=true) {
  const float values[]={0,.5f,1,1.5f,2,3,4,6};
  unsigned code=(image[(size_t(r)*k+c)/2]>>(4*(c%2)))&15;
  unsigned cols=(k/16+3)/4*4,scol=c/16;
  size_t tile=((r/128)*(cols/4)+scol/4)*512+(r%32)*16+((r%128)/32)*4+scol%4;
  const unsigned char* scales=image+size_t(n)*k/2;
  float global=*reinterpret_cast<const float*>(scales+size_t((n+127)/128*128)*cols);
  return values[code&7]*(code&8?-1.f:1.f)*fp8(scales[tile])*(scaled?global:1.f);
}
GLM_HD inline float conv_sum(const float* history,float current,const float* weights) {
  float sum=0;for(unsigned j=0;j<3;++j)sum+=history[j]*weights[j];return bf16(sum+current*weights[3]);
}
// One value column of row-major [key,value] state. CUDA lanes own disjoint
// columns; the CPU oracle calls exactly this arithmetic for every column.
GLM_HD inline float kda_column(float* state, const float* q, const float* k,
                              float v, const float* log_decay, float beta,
                              unsigned dim, unsigned j) {
  float memory=0;
  for (unsigned i=0; i<dim; ++i) {
    state[i*dim+j] *= expf(log_decay[i]);
    memory += state[i*dim+j]*k[i];
  }
  float delta=(v-memory)*beta, sum=0;
  for (unsigned i=0; i<dim; ++i) {
    state[i*dim+j] += k[i]*delta;
    sum += state[i*dim+j]*q[i];
  }
  return sum/sqrtf(float(dim));
}
GLM_HD inline void hc_weights(float* out, const float* logits, const float* base,
                              const float* scale) {
  constexpr float eps=1.e-6f;
  for (unsigned i=0; i<4; ++i) {
    out[i]=sigmoid(logits[i]*scale[0]+base[i])+eps;
    out[4+i]=2.f*sigmoid(logits[4+i]*scale[1]+base[4+i]);
    float max=-INFINITY;
    for (unsigned j=0; j<4; ++j) max=fmaxf(max,logits[8+i*4+j]*scale[2]+base[8+i*4+j]);
    float sum=0;
    for (unsigned j=0; j<4; ++j) { out[8+i*4+j]=expf(logits[8+i*4+j]*scale[2]+base[8+i*4+j]-max); sum+=out[8+i*4+j]; }
    for (unsigned j=0; j<4; ++j) out[8+i*4+j]=out[8+i*4+j]/sum+eps;
  }
  for (unsigned iter=0; iter<20; ++iter) {
    if (iter) for (unsigned i=0; i<4; ++i) {
      float sum=eps; for (unsigned j=0; j<4; ++j) sum+=out[8+i*4+j];
      for (unsigned j=0; j<4; ++j) out[8+i*4+j]/=sum;
    }
    for (unsigned j=0; j<4; ++j) {
      float sum=eps; for (unsigned i=0; i<4; ++i) sum+=out[8+i*4+j];
      for (unsigned i=0; i<4; ++i) out[8+i*4+j]/=sum;
    }
  }
}
}
