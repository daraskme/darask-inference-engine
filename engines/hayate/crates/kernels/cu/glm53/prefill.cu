#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "math.cuh"
namespace {
using B=__nv_bfloat16;
__device__ float f(B x){return __bfloat162float(x);}
__device__ B b(float x){return __float2bfloat16_rn(x);}
__global__ void embed(B* out,const B* table,const unsigned* ids,unsigned rows){
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if(i<size_t(rows)*16384)out[i]=table[size_t(ids[i/16384])*4096+i%4096];
}
__global__ void conv(B* out,const B* history,const B* x,const B* w,unsigned rows,unsigned cols){
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(rows)*cols)return;
  unsigned t=i/cols,c=i%cols;float sum=0;
  for(unsigned j=0;j<4;++j){int r=int(t)+int(j)-3;
    sum+=f(r<0?history[(r+3)*cols+c]:x[size_t(r)*cols+c])*f(w[c*4+j]);}
  sum=glm53::bf16(sum);out[i]=b(sum*glm53::sigmoid(sum));
}
__global__ void conv_tail(B* history,const B* x,unsigned rows,unsigned cols){
  unsigned c=blockIdx.x*256+threadIdx.x;if(c>=cols)return;
  B tail[3];for(unsigned j=0;j<3;++j){int r=int(rows)+int(j)-3;
    tail[j]=r<0?history[(r+3)*cols+c]:x[size_t(r)*cols+c];}
  for(unsigned j=0;j<3;++j)history[j*cols+c]=tail[j];
}
__global__ void pools(B* out,const B* keys,const B* gates,const B* oldk,const B* oldg,
    const B* ape,unsigned start,unsigned rows){
  unsigned i=blockIdx.x*256+threadIdx.x,first=start/4,last=(start+rows)/4;
  if(i>=(last-first)*128)return;unsigned p=first+i/128,c=i%128;
  float logits[4],m=-INFINITY,den=0;
  for(unsigned j=0;j<4;++j){int r=int(p*4+j)-int(start);
    logits[j]=f(r<0?oldg[j*128+c]:gates[size_t(r)*128+c])+f(ape[j*128+c]);m=fmaxf(m,logits[j]);}
  for(unsigned j=0;j<4;++j){logits[j]=expf(logits[j]-m);den+=logits[j];}
  float sum=0;for(unsigned j=0;j<4;++j){int r=int(p*4+j)-int(start);
    float key=f(r<0?oldk[j*128+c]:keys[size_t(r)*128+c]);
    sum+=glm53::bf16(glm53::bf16(logits[j]/den)*key);}
  out[p*128+c]=b(sum);
}
__global__ void pool_tail(B* oldk,B* oldg,const B* keys,const B* gates,unsigned start,unsigned rows){
  unsigned c=threadIdx.x,begin=(start+rows)/4*4;
  // Complete pools no longer need their tail. Partial pools retain old slots
  // preceding this chunk; decode subsequently overwrites unused slots.
  for(unsigned pos=begin;pos<start+rows;++pos)if(pos>=start){
    oldk[(pos%4)*128+c]=keys[size_t(pos-start)*128+c];
    oldg[(pos%4)*128+c]=gates[size_t(pos-start)*128+c];}
}
__global__ void select(int* ids,float* scratch,const B* q,const B* pools,const B* weights,
    unsigned start,unsigned rows){
  unsigned t=blockIdx.x,length=start+t+1,np=length/4,stride=(start+rows)/4;
  float* scores=scratch+size_t(t)*stride;q+=size_t(t)*4096;weights+=t*32;ids+=size_t(t)*2051;
  for(unsigned p=threadIdx.x;p<np;p+=256){float sum=0;
    for(unsigned h=0;h<32;++h){float dot=0;for(unsigned c=0;c<128;++c)dot+=f(q[h*128+c])*f(pools[p*128+c]);
      sum+=fmaxf(0.f,dot/sqrtf(128.f))*f(weights[h])/sqrtf(32.f);}scores[p]=sum;}
  __syncthreads();if(threadIdx.x)return;
  for(unsigned j=0;j<2051;++j)ids[j]=-1;
  unsigned count=0;for(unsigned j=0;j<512&&j<np;++j){unsigned best=0;
    for(unsigned p=1;p<np;++p)if(scores[p]>scores[best])best=p;
    for(unsigned c=0;c<4;++c)ids[count++]=best*4+c;scores[best]=-INFINITY;}
  for(unsigned c=np*4;c<length;++c)ids[count++]=c;
}
__device__ float reduce(float v,float* s){s[threadIdx.x]=v;__syncthreads();
  for(unsigned j=128;j;j/=2){if(threadIdx.x<j)s[threadIdx.x]+=s[threadIdx.x+j];__syncthreads();}return s[0];}
__global__ void attention(B* out,const B* q,const B* packed,const unsigned* indices,const unsigned* counts){
  __shared__ B prob[2051];__shared__ float s[256];
  unsigned row=blockIdx.x,h=blockIdx.y,count=counts[row];indices+=size_t(row)*2051;q+=(size_t(row)*64+h)*256;
  for(unsigned j=threadIdx.x;j<count;j+=256){float dot=0;auto key=packed+(size_t(indices[j])*64+h)*512;
    for(unsigned c=0;c<256;++c)dot+=f(q[c])*f(key[c]);prob[j]=b(glm53::bf16(dot)/16.f);}
  __syncthreads();float m=-INFINITY;for(unsigned j=threadIdx.x;j<count;j+=256)m=fmaxf(m,f(prob[j]));
  s[threadIdx.x]=m;__syncthreads();for(unsigned j=128;j;j/=2){if(threadIdx.x<j)s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+j]);__syncthreads();}
  m=s[0];__syncthreads();float den=0;for(unsigned j=threadIdx.x;j<count;j+=256)den+=expf(f(prob[j])-m);
  den=reduce(den,s);for(unsigned j=threadIdx.x;j<count;j+=256)prob[j]=b(expf(f(prob[j])-m)/den);
  __syncthreads();float dot=0;unsigned c=threadIdx.x;
  for(unsigned j=0;j<count;++j)dot+=f(prob[j])*f(packed[(size_t(indices[j])*64+h)*512+256+c]);
  out[(size_t(row)*64+h)*256+c]=b(dot);
}
}
extern "C" int hayate_glm_embed_batch(void* out,const void* table,const void* ids,unsigned rows,cudaStream_t st){
  if(!out||!table||!ids||!rows)return cudaErrorInvalidValue;
  embed<<<(size_t(rows)*16384+255)/256,256,0,st>>>((B*)out,(const B*)table,(const unsigned*)ids,rows);return cudaGetLastError();
}
extern "C" int hayate_glm_conv_chunk(void* out,void* history,const void* x,const void* w,unsigned rows,unsigned cols,cudaStream_t st){
  if(!out||!history||!x||!w||!rows||!cols)return cudaErrorInvalidValue;
  conv<<<(size_t(rows)*cols+255)/256,256,0,st>>>((B*)out,(const B*)history,(const B*)x,(const B*)w,rows,cols);
  auto e=cudaGetLastError();if(e)return e;
  conv_tail<<<(cols+255)/256,256,0,st>>>((B*)history,(const B*)x,rows,cols);return cudaGetLastError();
}
extern "C" int hayate_glm_pool_chunk(void* out,void* oldk,void* oldg,const void* keys,const void* gates,const void* ape,unsigned start,unsigned rows,cudaStream_t st){
  if(!out||!oldk||!oldg||!keys||!gates||!ape||!rows)return cudaErrorInvalidValue;
  unsigned count=((start+rows)/4-start/4)*128;
  if(count){pools<<<(count+255)/256,256,0,st>>>((B*)out,(const B*)keys,(const B*)gates,(const B*)oldk,(const B*)oldg,(const B*)ape,start,rows);
    auto e=cudaGetLastError();if(e)return e;}
  pool_tail<<<1,128,0,st>>>((B*)oldk,(B*)oldg,(const B*)keys,(const B*)gates,start,rows);return cudaGetLastError();
}
extern "C" int hayate_glm_select_chunk(void* ids,void* scores,const void* q,const void* pools,const void* weights,unsigned start,unsigned rows,cudaStream_t st){
  if(!ids||!scores||!q||!pools||!weights||!rows)return cudaErrorInvalidValue;
  select<<<rows,256,0,st>>>((int*)ids,(float*)scores,(const B*)q,(const B*)pools,(const B*)weights,start,rows);return cudaGetLastError();
}
extern "C" int hayate_glm_attention_indexed(void* out,const void* q,const void* packed,const void* indices,const void* counts,unsigned rows,cudaStream_t st){
  if(!out||!q||!packed||!indices||!counts||!rows)return cudaErrorInvalidValue;
  attention<<<dim3(rows,64),256,0,st>>>((B*)out,(const B*)q,(const B*)packed,(const unsigned*)indices,(const unsigned*)counts);return cudaGetLastError();
}
extern "C" int hayate_glm_prefill_latent(void* host,const void* latent,size_t bytes,cudaStream_t st){
  if(!host||!latent||!bytes)return cudaErrorInvalidValue;return cudaMemcpyAsync(host,latent,bytes,cudaMemcpyDeviceToHost,st);
}

namespace {
constexpr unsigned KC=16, KD=128;
__global__ void kda_normalize(float* nq,float* nk,const B* q,const B* k,unsigned heads,unsigned dim){
  __shared__ float qs[128],ks[128];unsigned c=threadIdx.x;
  size_t p=(size_t(blockIdx.x)*heads+blockIdx.y)*dim;
  float a=f(q[p+c]),z=f(k[p+c]);qs[c]=a*a;ks[c]=z*z;__syncthreads();
  for(unsigned s=dim/2;s;s/=2){if(c<s){qs[c]+=qs[c+s];ks[c]+=ks[c+s];}__syncthreads();}
  nq[p+c]=a/sqrtf(qs[0]+1.e-6f);nk[p+c]=z/sqrtf(ks[0]+1.e-6f);
}
// A rank-one delta update becomes a unit lower-triangular system per chunk.
// Invert it by doubling (I-L)^-1 = (I+L)(I+L^2)(I+L^4)... .
// Value columns are independent given the chunk's (I-L)^-1 and q.k terms, so
// each block owns JS state columns of one head in shared memory and the grid
// is heads x dim/JS. Per-element arithmetic and order match the one-block-
// per-head version (exp(g) and exp(g_last - g_i) k are only hoisted).
constexpr unsigned JS=16;
__global__ void __launch_bounds__(256) kda_chunk(B* out,float* state,const float* nq,const float* nk,const B* v,
    const float* decay,const B* beta,unsigned tokens,unsigned heads,unsigned dim){
  __shared__ float q[KC*KD],k[KC*KD],g[KC*KD],ke[KC*KD];
  __shared__ float s[KD*JS],u[KC*JS],delta[KC*JS];
  __shared__ float inv[KC*KC],power[KC*KC],ni[KC*KC],np[KC*KC],qk[KC*KC];
  const unsigned lane=threadIdx.x,h=blockIdx.x,js=dim/gridDim.y,j0=blockIdx.y*js;state+=size_t(h)*dim*dim;
  for(unsigned x=lane;x<dim*js;x+=256){unsigned c=x/js,j=x%js;s[x]=state[c*dim+j0+j];}
  for(unsigned start=0;start<tokens;start+=KC){
    unsigned n=min(KC,tokens-start);
    for(unsigned x=lane;x<n*dim;x+=256){unsigned t=x/dim,c=x%dim;
      size_t p=(size_t(start+t)*heads+h)*dim+c;
      q[x]=nq[p];k[x]=nk[p];float sum=0;
      for(unsigned i=0;i<=t;++i)sum+=decay[(size_t(start+i)*heads+h)*dim+c];g[x]=sum;
    }__syncthreads();
    if(lane<KC*KC){unsigned t=lane/KC,i=lane%KC;float a=0,z=0;
      if(t<n&&i<=t)for(unsigned c=0;c<dim;++c){float d=expf(g[t*dim+c]-g[i*dim+c])*k[i*dim+c];
        a+=k[t*dim+c]*d;z+=q[t*dim+c]*d;}
      float l=t<n&&i<t?-f(beta[(start+t)*heads+h])*a:0;
      power[lane]=l;inv[lane]=(t==i?1.f:0.f)+l;qk[lane]=z;
    }__syncthreads();
    for(unsigned stride=2;stride<KC;stride*=2){
      if(lane<KC*KC){unsigned t=lane/KC,i=lane%KC;float p=0;
        for(unsigned j=0;j<KC;++j)p+=power[t*KC+j]*power[j*KC+i];np[lane]=p;
      }__syncthreads();
      if(lane<KC*KC){unsigned t=lane/KC,i=lane%KC;float a=inv[lane];
        for(unsigned j=0;j<KC;++j)a+=np[t*KC+j]*inv[j*KC+i];ni[lane]=a;
      }__syncthreads();
      if(lane<KC*KC){power[lane]=np[lane];inv[lane]=ni[lane];}__syncthreads();
    }
    // q becomes q exp(g) once the q.k terms are done; ke = k exp(g).
    for(unsigned x=lane;x<n*dim;x+=256){float e=expf(g[x]);q[x]=q[x]*e;ke[x]=k[x]*e;}
    __syncthreads();
    if(lane<n*js){unsigned t=lane/js,j=lane%js;float memory=0;
      for(unsigned c=0;c<dim;++c)memory+=ke[t*dim+c]*s[c*js+j];
      u[lane]=f(beta[(start+t)*heads+h])*(f(v[(size_t(start+t)*heads+h)*dim+j0+j])-memory);
    }__syncthreads();
    if(lane<n*js){unsigned t=lane/js,j=lane%js;float z=0;
      for(unsigned i=0;i<=t;++i)z+=inv[t*KC+i]*u[i*js+j];delta[lane]=z;
    }__syncthreads();
    if(lane<n*js){unsigned t=lane/js,j=lane%js;float a=0;
      for(unsigned c=0;c<dim;++c)a+=q[t*dim+c]*s[c*js+j];
      for(unsigned i=0;i<=t;++i)a+=qk[t*KC+i]*delta[i*js+j];
      out[(size_t(start+t)*heads+h)*dim+j0+j]=b(a/sqrtf(float(dim)));
    }
    // Reuse ke as w[i][c] = exp(g_last[c] - g_i[c]) k_i[c].
    __syncthreads();
    for(unsigned x=lane;x<n*dim;x+=256){unsigned i=x/dim,c=x%dim;ke[x]=expf(g[(n-1)*dim+c]-g[x])*k[x];}
    __syncthreads();
    for(unsigned x=lane;x<dim*js;x+=256){unsigned c=x/js,j=x%js;
      float a=expf(g[(n-1)*dim+c])*s[x];
      for(unsigned i=0;i<n;++i)a+=ke[i*dim+c]*delta[i*js+j];
      s[x]=a;
    }__syncthreads();
  }
  for(unsigned x=lane;x<dim*js;x+=256){unsigned c=x/js,j=x%js;state[c*dim+j0+j]=s[x];}
}
}
extern "C" int hayate_glm_kda_chunk(void* out,void* state,void* nq,void* nk,const void* q,
    const void* k,const void* v,const void* decay,const void* beta,unsigned tokens,unsigned heads,unsigned dim,cudaStream_t st){
  if(!out||!state||!nq||!nk||!q||!k||!v||!decay||!beta||!tokens||!heads||dim<2||dim>128||(dim&(dim-1)))return cudaErrorInvalidValue;
  kda_normalize<<<dim3(tokens,heads),dim,0,st>>>((float*)nq,(float*)nk,(const B*)q,(const B*)k,heads,dim);
  auto e=cudaGetLastError();if(e)return e;
  kda_chunk<<<dim3(heads,dim/min(JS,dim)),256,0,st>>>((B*)out,(float*)state,(const float*)nq,(const float*)nk,(const B*)v,(const float*)decay,(const B*)beta,tokens,heads,dim);return cudaGetLastError();
}
