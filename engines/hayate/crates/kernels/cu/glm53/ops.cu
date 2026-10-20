#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "math.cuh"
using B=__nv_bfloat16;
__device__ float f(B x) { return __bfloat162float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ float rnd(float x) { return f(b(x)); }
__device__ float reduce(float x, float* s) {
  __syncthreads(); s[threadIdx.x]=x; __syncthreads();
  for (unsigned i=blockDim.x/2;i;i/=2) { if(threadIdx.x<i)s[threadIdx.x]+=s[threadIdx.x+i]; __syncthreads(); }
  return s[0];
}
__global__ void restore(B* out,const unsigned char* w,const float* scale,unsigned n,unsigned k) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if(i<size_t(n)*k)out[i]=b(glm53::fp8(w[i])*scale[(i/k/128)*((k+127)/128)+(i%k)/128]);
}
__global__ void norm(B* out,const B* x,const B* w,const B* bias,unsigned cols,int mode) {
  __shared__ float s[256]; unsigned row=blockIdx.x; float mean=0, sq=0;
  if(mode==1) { for(unsigned i=threadIdx.x;i<cols;i+=256)mean+=f(x[size_t(row)*cols+i]); mean=reduce(mean,s)/cols; }
  for(unsigned i=threadIdx.x;i<cols;i+=256) { float v=f(x[size_t(row)*cols+i])-mean; sq+=v*v; }
  float inv=rsqrtf(reduce(sq,s)/cols+(mode==1?1.e-6f:1.e-5f));
  for(unsigned i=threadIdx.x;i<cols;i+=256) {
    float v=(f(x[size_t(row)*cols+i])-mean)*inv;
    out[size_t(row)*cols+i]=mode==1?b(v*f(w[i])+f(bias[i])):b(rnd(v)*f(w[i]));
  }
}
__global__ void point(B* out,const B* x,const B* y,size_t count,int mode) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if(i>=count)return;
  float a=f(x[i]),c=y?f(y[i]):0;
  if(mode==0)out[i]=b(glm53::sigmoid(a));
  if(mode==1) { a=fminf(a,10.f); c=fmaxf(-10.f,fminf(10.f,c)); out[i]=b(rnd(a*glm53::sigmoid(a))*c); }
  if(mode==2)out[i]=b(a+c);
}
__global__ void conv(B* out,const B* x,const B* w,unsigned tokens,unsigned cols) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if(i>=size_t(tokens)*cols)return;
  unsigned t=i/cols,c=i%cols; float sum=0;
  for(unsigned j=0;j<4;++j)if(t+j>=3)sum+=f(x[size_t(t+j-3)*cols+c])*f(w[c*4+j]);
  sum=rnd(sum); out[i]=b(sum*glm53::sigmoid(sum));
}
__global__ void hc_map(float* map,const B* x,const B* fn,const float* base,const float* scale,unsigned hidden) {
  __shared__ float s[256],logits[24]; unsigned t=blockIdx.x; float sq=0; unsigned cols=hidden*4;
  for(unsigned i=threadIdx.x;i<cols;i+=256) { float v=f(x[size_t(t)*cols+i]); sq+=v*v; }
  float inv=rsqrtf(reduce(sq,s)/cols+1.e-5f);
  for(unsigned j=0;j<24;++j) {
    float sum=0; for(unsigned i=threadIdx.x;i<cols;i+=256)sum+=f(x[size_t(t)*cols+i])*inv*f(fn[size_t(j)*cols+i]);
    sum=reduce(sum,s); if(threadIdx.x==0)logits[j]=sum;
    __syncthreads();
  }
  if(threadIdx.x==0)glm53::hc_weights(map+size_t(t)*24,logits,base,scale);
}
__global__ void hc_collapse(B* out,const B* x,const float* map,unsigned tokens,unsigned hidden) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x; if(i>=size_t(tokens)*hidden)return;
  unsigned t=i/hidden,c=i%hidden; float sum=0;
  for(unsigned j=0;j<4;++j)sum+=map[t*24+j]*f(x[(size_t(t)*4+j)*hidden+c]);
  out[i]=b(sum);
}
// Collapse retains its BF16 boundary before the following RMSNorm.
__global__ void hc_collapse_norm(B* collapsed,B* out,const B* x,const float* map,const B* w,unsigned hidden) {
  __shared__ float s[256];float sq=0;
  for(unsigned c=threadIdx.x;c<hidden;c+=256){float sum=0;for(unsigned h=0;h<4;++h)sum+=map[h]*f(x[h*hidden+c]);collapsed[c]=b(sum);sq+=rnd(sum)*rnd(sum);}
  float inv=rsqrtf(reduce(sq,s)/hidden+1.e-5f);
  for(unsigned c=threadIdx.x;c<hidden;c+=256)out[c]=b(rnd(f(collapsed[c])*inv)*f(w[c]));
}
__global__ void kda_gated(B* core,B* out,float* state,const B* q,const B* k,const B* v,const float* decay,const B* beta,const B* gate,const B* w) {
  __shared__ float qs[128],ks[128],s[128];unsigned h=blockIdx.x,j=threadIdx.x;size_t p=size_t(h)*128;
  float qv=f(q[p+j]),kv=f(k[p+j]);float nq=reduce(qv*qv,s),nk=reduce(kv*kv,s);
  qs[j]=qv/sqrtf(nq+1.e-6f);ks[j]=kv/sqrtf(nk+1.e-6f);__syncthreads();
  float value=rnd(glm53::kda_column(state+size_t(h)*128*128,qs,ks,f(v[p+j]),decay+p,f(beta[h]),128,j));
  core[p+j]=b(value);
  float inv=rsqrtf(reduce(value*value,s)/128+1.e-5f);
  out[p+j]=b(value*inv*f(w[j])*glm53::sigmoid(f(gate[p+j])));
}
__global__ void hc_expand(B* out,const B* x,const B* branch,const float* map,unsigned tokens,unsigned hidden) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x; if(i>=size_t(tokens)*4*hidden)return;
  unsigned t=i/(4*hidden),j=(i/hidden)%4,c=i%hidden; float mix=0;
  for(unsigned h=0;h<4;++h)mix+=rnd(map[t*24+8+h*4+j])*f(x[(size_t(t)*4+h)*hidden+c]);
  float product=rnd(rnd(map[t*24+4+j])*f(branch[size_t(t)*hidden+c]));
  out[i]=b(product+rnd(mix));
}
__global__ void decay(float* out,const B* projection,const float* bias,const float* alog,unsigned count,unsigned dim,unsigned heads) {
  unsigned i=blockIdx.x*256+threadIdx.x; if(i<count)out[i]=glm53::forget(f(projection[i]),bias[i%(dim*heads)],alog[(i/dim)%heads]);
}
__global__ void kda(B* out,float* state,const B* q,const B* k,const B* v,const float* g,const B* beta,unsigned tokens,unsigned heads,unsigned dim) {
  __shared__ float qs[128],ks[128],s[128]; unsigned h=blockIdx.x,j=threadIdx.x;
  for(unsigned t=0;t<tokens;++t) {
    size_t p=(size_t(t)*heads+h)*dim; float qv=f(q[p+j]),kv=f(k[p+j]);
    float nq=reduce(qv*qv,s),nk=reduce(kv*kv,s);
    qs[j]=qv/sqrtf(nq+1.e-6f); ks[j]=kv/sqrtf(nk+1.e-6f); __syncthreads();
    out[p+j]=b(glm53::kda_column(state+size_t(h)*dim*dim,qs,ks,f(v[p+j]),
                               g+p,f(beta[t*heads+h]),dim,j)); __syncthreads();
  }
}
__global__ void gated_norm(B* out,const B* x,const B* gate,const B* w,unsigned dim) {
  __shared__ float s[128]; unsigned i=blockIdx.x*dim+threadIdx.x; float v=f(x[i]);
  float inv=rsqrtf(reduce(v*v,s)/dim+1.e-5f);
  out[i]=b(v*inv*f(w[threadIdx.x])*glm53::sigmoid(f(gate[i])));
}
__global__ void route(float* logits,float* weights,int* ids,const B* x,const B* w,const float* bias,unsigned hidden,unsigned experts,unsigned topk) {
  __shared__ float s[256],scores[288],choices[288]; unsigned t=blockIdx.x;
  for(unsigned e=0;e<experts;++e) {
    float sum=0;for(unsigned c=threadIdx.x;c<hidden;c+=256)sum+=f(x[size_t(t)*hidden+c])*f(w[size_t(e)*hidden+c]);
    sum=reduce(sum,s);if(threadIdx.x==0) { logits[t*experts+e]=sum; scores[e]=glm53::sigmoid(sum);choices[e]=scores[e]+bias[e]; } __syncthreads();
  }
  if(threadIdx.x==0) {
    float sum=0;
    for(unsigned j=0;j<topk;++j) {
      unsigned best=0;for(unsigned e=1;e<experts;++e)if(choices[e]>choices[best])best=e;
      ids[t*topk+j]=best;weights[t*topk+j]=scores[best];sum+=scores[best];choices[best]=-INFINITY;
    }
    for(unsigned j=0;j<topk;++j)weights[t*topk+j]=weights[t*topk+j]/(sum+1.e-20f)*2.5f;
  }
}
// Entire-token expert evaluation is deliberate stage-1 simplicity. Only selected
// experts run, but gather/grouped GEMM and cache placement belong to stage 2.
__global__ void accumulate(B* out,const B* expert,const int* ids,const float* weights,unsigned tokens,unsigned hidden,unsigned topk,unsigned e) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(tokens)*hidden)return;
  unsigned t=i/hidden;
  for(unsigned j=0;j<topk;++j)if(ids[t*topk+j]==int(e))out[i]=b(f(out[i])+rnd(f(expert[i])*weights[t*topk+j]));
}
__global__ void pool(B* out,const B* keys,const B* gates,const B* ape,unsigned pools,unsigned dim) {
  unsigned i=blockIdx.x*256+threadIdx.x;if(i>=pools*dim)return;
  unsigned p=i/dim,c=i%dim;float m=-INFINITY,den=0,logits[4];
  for(unsigned j=0;j<4;++j) { logits[j]=f(gates[(p*4+j)*dim+c])+f(ape[j*dim+c]); m=fmaxf(m,logits[j]); }
  for(unsigned j=0;j<4;++j) { logits[j]=expf(logits[j]-m);den+=logits[j]; }
  float sum=0;for(unsigned j=0;j<4;++j)sum+=rnd(rnd(logits[j]/den)*f(keys[(p*4+j)*dim+c]));
  out[i]=b(sum);
}
__global__ void select(int* ids,float* scratch,const B* q,const B* pools,const B* weights,unsigned tokens,unsigned heads,unsigned dim,unsigned budget) {
  unsigned t=blockIdx.x;unsigned np=tokens/4;float* scores=scratch+size_t(t)*np;
  for(unsigned p=threadIdx.x;p<np;p+=256) {
    float sum=0;
    for(unsigned h=0;h<heads;++h) {
      float dot=0;for(unsigned c=0;c<dim;++c)dot+=f(q[(size_t(t)*heads+h)*dim+c])*f(pools[p*dim+c]);
      sum+=fmaxf(0.f,dot/sqrtf(float(dim)))*f(weights[t*heads+h])/sqrtf(float(heads));
    }
    scores[p]=p*4+3<=t?sum:-INFINITY;
  }
  __syncthreads();if(threadIdx.x)return;
  unsigned width=budget+3;for(unsigned j=0;j<width;++j)ids[size_t(t)*width+j]=-1;
  unsigned n=0;
  for(unsigned j=0;j<budget/4 && j<np;++j) {
    unsigned best=0;for(unsigned p=1;p<np;++p)if(scores[p]>scores[best])best=p;
    if(scores[best]==-INFINITY)break;
    for(unsigned c=0;c<4;++c)ids[size_t(t)*width+n++]=best*4+c;
    scores[best]=-INFINITY;
  }
  for(unsigned c=((t+1)/4)*4;c<=t;++c)ids[size_t(t)*width+n++]=c;
}
__global__ void split_kv(B* k,B* v,const B* packed,unsigned tokens,unsigned heads,unsigned dim) {
  unsigned i=blockIdx.x*256+threadIdx.x;if(i>=tokens*heads*dim)return;
  unsigned row=i/dim,c=i%dim;k[i]=packed[size_t(row)*dim*2+c];v[i]=packed[size_t(row)*dim*2+dim+c];
}
__global__ void attention(B* out,B* probabilities,const B* q,const B* k,const B* v,const int* ids,unsigned tokens,unsigned heads,unsigned dim,unsigned width) {
  __shared__ float s[256];unsigned t=blockIdx.x,h=blockIdx.y;B* prob=probabilities+(size_t(t)*heads+h)*tokens;
  for(unsigned j=threadIdx.x;j<tokens;j+=256) {
    bool visible=false;for(unsigned a=0;a<width;++a)if(ids[size_t(t)*width+a]==int(j))visible=true;
    float dot=0;for(unsigned c=0;c<dim;++c)dot+=f(q[(size_t(t)*heads+h)*dim+c])*f(k[(size_t(j)*heads+h)*dim+c]);
    prob[j]=b(visible?rnd(rnd(dot)/sqrtf(float(dim))):-3.38953139e38f);
  }
  __syncthreads();float max=-INFINITY;
  for(unsigned j=threadIdx.x;j<tokens;j+=256)max=fmaxf(max,f(prob[j]));
  s[threadIdx.x]=max;__syncthreads();
  for(unsigned i=128;i;i/=2) { if(threadIdx.x<i)s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+i]);__syncthreads(); }
  max=s[0];__syncthreads();float sum=0;
  for(unsigned j=threadIdx.x;j<tokens;j+=256)sum+=expf(f(prob[j])-max);
  float den=reduce(sum,s);
  for(unsigned j=threadIdx.x;j<tokens;j+=256)prob[j]=b(expf(f(prob[j])-max)/den);
  __syncthreads();
  for(unsigned c=threadIdx.x;c<dim;c+=256) {
    float dot=0;for(unsigned j=0;j<tokens;++j)dot+=f(prob[j])*f(v[(size_t(j)*heads+h)*dim+c]);
    out[(size_t(t)*heads+h)*dim+c]=b(dot);
  }
}
// Decode reads block FP8 directly and rounds each scaled value as the BF16 oracle.
__global__ void fp8_gemv(B* out,const B* x,const unsigned char* w,const float* scale,unsigned n,unsigned k) {
  __shared__ float s[256];unsigned row=blockIdx.x;float sum=0;
  for(unsigned c=threadIdx.x;c<k;c+=256)
    sum+=f(x[c])*glm53::scaled_fp8(w[size_t(row)*k+c],scale[(row/128)*((k+127)/128)+c/128]);
  sum=reduce(sum,s);if(!threadIdx.x)out[row]=b(sum);
}
__global__ void conv_step(B* out,B* history,const B* x,const B* w,unsigned cols) {
  unsigned c=blockIdx.x*256+threadIdx.x;if(c>=cols)return;
  float past[3],weight[4];
  for(unsigned j=0;j<3;++j)past[j]=f(history[j*cols+c]);
  for(unsigned j=0;j<4;++j)weight[j]=f(w[c*4+j]);
  float sum=glm53::conv_sum(past,f(x[c]),weight);
  for(unsigned j=0;j<2;++j)history[j*cols+c]=history[(j+1)*cols+c];
  history[2*cols+c]=x[c];sum=rnd(sum);out[c]=b(sum*glm53::sigmoid(sum));
}
__global__ void select_step(int* ids,float* scores,const B* q,const B* pools,const B* weights,unsigned length) {
  unsigned np=length/4;
  for(unsigned p=threadIdx.x;p<np;p+=256) {
    float sum=0;for(unsigned h=0;h<32;++h) {
      float dot=0;for(unsigned c=0;c<128;++c)dot+=f(q[h*128+c])*f(pools[p*128+c]);
      sum+=fmaxf(0.f,dot/sqrtf(128.f))*f(weights[h])/sqrtf(32.f);
    }scores[p]=sum;
  }
  __syncthreads();if(threadIdx.x)return;
  for(unsigned j=0;j<2051;++j)ids[j]=-1;
  unsigned n=0;for(unsigned j=0;j<512&&j<np;++j) {
    unsigned best=0;for(unsigned p=1;p<np;++p)if(scores[p]>scores[best])best=p;
    for(unsigned c=0;c<4;++c)ids[n++]=best*4+c;scores[best]=-INFINITY;
  }
  for(unsigned c=np*4;c<length;++c)ids[n++]=c;
}
__global__ void attention_step(B* out,B* probs,const B* q,const B* k,const B* v,unsigned count) {
  __shared__ float s[256];unsigned h=blockIdx.x;B* prob=probs+h*count;
  for(unsigned j=threadIdx.x;j<count;j+=256) {
    float dot=0;for(unsigned c=0;c<256;++c)dot+=f(q[h*256+c])*f(k[(size_t(j)*64+h)*256+c]);
    prob[j]=b(rnd(dot)/16.f);
  }
  __syncthreads();float max=-INFINITY;
  for(unsigned j=threadIdx.x;j<count;j+=256)max=fmaxf(max,f(prob[j]));
  s[threadIdx.x]=max;__syncthreads();
  for(unsigned i=128;i;i/=2){if(threadIdx.x<i)s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+i]);__syncthreads();}
  max=s[0];__syncthreads();float sum=0;
  for(unsigned j=threadIdx.x;j<count;j+=256)sum+=expf(f(prob[j])-max);
  float den=reduce(sum,s);
  for(unsigned j=threadIdx.x;j<count;j+=256)prob[j]=b(expf(f(prob[j])-max)/den);
  __syncthreads();unsigned c=threadIdx.x;float dot=0;
  for(unsigned j=0;j<count;++j)dot+=f(prob[j])*f(v[(size_t(j)*64+h)*256+c]);
  out[h*256+c]=b(dot);
}
__global__ void embed(B* out,const B* table,unsigned token,unsigned hidden) {
 unsigned i=blockIdx.x*256+threadIdx.x;if(i<4*hidden)out[i]=table[size_t(token)*hidden+i%hidden];
}
__global__ void head_mean(B* out,const B* x,unsigned hidden) {
  unsigned c=blockIdx.x*256+threadIdx.x;if(c<hidden){float sum=0;for(unsigned j=0;j<4;++j)sum+=f(x[j*hidden+c]);out[c]=b(sum/4.f);}
}
__global__ void greedy(unsigned* id,const B* logits,unsigned n) {
  __shared__ float values[256];__shared__ unsigned ids[256];
  float best=-INFINITY;unsigned arg=0;
  for(unsigned i=threadIdx.x;i<n;i+=256)if(f(logits[i])>best){best=f(logits[i]);arg=i;}
  values[threadIdx.x]=best;ids[threadIdx.x]=arg;__syncthreads();
  for(unsigned s=128;s;s/=2){if(threadIdx.x<s&&(values[threadIdx.x+s]>values[threadIdx.x]||(values[threadIdx.x+s]==values[threadIdx.x]&&ids[threadIdx.x+s]<ids[threadIdx.x]))){values[threadIdx.x]=values[threadIdx.x+s];ids[threadIdx.x]=ids[threadIdx.x+s];}__syncthreads();}
  if(!threadIdx.x)*id=ids[0];
}
#define END return cudaGetLastError()
extern "C" {
int hayate_glm_embed(void* out,const void* table,unsigned token,unsigned hidden,cudaStream_t st) {
 if(!out||!table||!hidden)return cudaErrorInvalidValue;
 embed<<<(hidden*4+255)/256,256,0,st>>>((B*)out,(const B*)table,token,hidden);END;
}
int hayate_glm_fp8_gemv(void* out,const void* x,const void* w,const void* scale,unsigned n,unsigned k,cudaStream_t st) {
 if(!out||!x||!w||!scale||!n||!k)return cudaErrorInvalidValue;
 fp8_gemv<<<n,256,0,st>>>((B*)out,(const B*)x,(const unsigned char*)w,(const float*)scale,n,k);END;
}
int hayate_glm_conv_step(void* out,void* history,const void* x,const void* w,unsigned cols,cudaStream_t st) {
 if(!out||!history||!x||!w||!cols)return cudaErrorInvalidValue;
 conv_step<<<(cols+255)/256,256,0,st>>>((B*)out,(B*)history,(const B*)x,(const B*)w,cols);END;
}
int hayate_glm_select_step(void* ids,void* scores,const void* q,const void* pools,const void* weights,unsigned length,cudaStream_t st) {
 if(!ids||!scores||!q||!pools||!weights||!length)return cudaErrorInvalidValue;
 select_step<<<1,256,0,st>>>((int*)ids,(float*)scores,(const B*)q,(const B*)pools,(const B*)weights,length);END;
}
int hayate_glm_attention_step(void* out,void* probs,const void* q,const void* k,const void* v,unsigned count,cudaStream_t st) {
 if(!out||!probs||!q||!k||!v||!count||count>2051)return cudaErrorInvalidValue;
 attention_step<<<64,256,0,st>>>((B*)out,(B*)probs,(const B*)q,(const B*)k,(const B*)v,count);END;
}
int hayate_glm_head_mean(void* out,const void* x,unsigned hidden,cudaStream_t st) {
 if(!out||!x||!hidden)return cudaErrorInvalidValue;
 head_mean<<<(hidden+255)/256,256,0,st>>>((B*)out,(const B*)x,hidden);END;
}
int hayate_glm_greedy(void* id,const void* logits,unsigned n,cudaStream_t st) {
 if(!id||!logits||!n)return cudaErrorInvalidValue;
 greedy<<<1,256,0,st>>>((unsigned*)id,(const B*)logits,n);END;
}

int hayate_glm_restore(void* out,const void* w,const void* scale,unsigned n,unsigned k,cudaStream_t st) {
 if(!out||!w||!scale||!n||!k)return cudaErrorInvalidValue;
 restore<<<(size_t(n)*k+255)/256,256,0,st>>>((B*)out,(const unsigned char*)w,(const float*)scale,n,k);END;
}
int hayate_glm_norm(void* out,const void* x,const void* w,const void* bias,unsigned rows,unsigned cols,int mode,cudaStream_t st) {
 if(!out||!x||!w||!rows||!cols||mode<0||mode>1||(mode==1&&!bias))return cudaErrorInvalidValue;
 norm<<<rows,256,0,st>>>((B*)out,(const B*)x,(const B*)w,(const B*)bias,cols,mode);END;
}
int hayate_glm_point(void* out,const void* x,const void* y,size_t count,int mode,cudaStream_t st) {
 if(!out||!x||!count||mode<0||mode>2||(mode>0&&!y))return cudaErrorInvalidValue;
 point<<<(count+255)/256,256,0,st>>>((B*)out,(const B*)x,(const B*)y,count,mode);END;
}
int hayate_glm_conv(void* out,const void* x,const void* w,unsigned tokens,unsigned cols,cudaStream_t st) {
 if(!out||!x||!w||!tokens||!cols)return cudaErrorInvalidValue;
 conv<<<(size_t(tokens)*cols+255)/256,256,0,st>>>((B*)out,(const B*)x,(const B*)w,tokens,cols);END;
}
int hayate_glm_hc(void* map,void* out,const void* x,const void* fn,const void* base,const void* scale,unsigned tokens,unsigned hidden,cudaStream_t st) {
 if(!map||!out||!x||!fn||!base||!scale||!tokens||!hidden)return cudaErrorInvalidValue;
 hc_map<<<tokens,256,0,st>>>((float*)map,(const B*)x,(const B*)fn,(const float*)base,(const float*)scale,hidden);
 hc_collapse<<<(size_t(tokens)*hidden+255)/256,256,0,st>>>((B*)out,(const B*)x,(const float*)map,tokens,hidden);END;
}
int hayate_glm_expand(void* out,const void* x,const void* branch,const void* map,unsigned tokens,unsigned hidden,cudaStream_t st) {
 if(!out||!x||!branch||!map||!tokens||!hidden)return cudaErrorInvalidValue;
 hc_expand<<<(size_t(tokens)*hidden*4+255)/256,256,0,st>>>((B*)out,(const B*)x,(const B*)branch,(const float*)map,tokens,hidden);END;
}
int hayate_glm_decay(void* out,const void* x,const void* bias,const void* alog,unsigned tokens,unsigned heads,unsigned dim,cudaStream_t st) {
 if(!out||!x||!bias||!alog||!tokens||!heads||!dim)return cudaErrorInvalidValue;
 unsigned count=tokens*heads*dim;decay<<<(count+255)/256,256,0,st>>>((float*)out,(const B*)x,(const float*)bias,(const float*)alog,count,dim,heads);END;
}
int hayate_glm_kda(void* out,void* state,const void* q,const void* k,const void* v,const void* g,const void* beta,unsigned tokens,unsigned heads,unsigned dim,cudaStream_t st) {
 if(!out||!state||!q||!k||!v||!g||!beta||!tokens||!heads||dim<2||dim>128||(dim&(dim-1)))return cudaErrorInvalidValue;
 kda<<<heads,dim,0,st>>>((B*)out,(float*)state,(const B*)q,(const B*)k,(const B*)v,(const float*)g,(const B*)beta,tokens,heads,dim);END;
}
int hayate_glm_gated(void* out,const void* x,const void* gate,const void* w,unsigned rows,unsigned dim,cudaStream_t st) {
 if(!out||!x||!gate||!w||!rows||dim<2||dim>128||(dim&(dim-1)))return cudaErrorInvalidValue;
 gated_norm<<<rows,dim,0,st>>>((B*)out,(const B*)x,(const B*)gate,(const B*)w,dim);END;
}
int hayate_glm_route(void* logits,void* weights,void* ids,const void* x,const void* w,const void* bias,unsigned tokens,unsigned hidden,unsigned experts,unsigned topk,cudaStream_t st) {
 if(!logits||!weights||!ids||!x||!w||!bias||!tokens||!hidden||!experts||experts>288||!topk||topk>experts)return cudaErrorInvalidValue;
 route<<<tokens,256,0,st>>>((float*)logits,(float*)weights,(int*)ids,(const B*)x,(const B*)w,(const float*)bias,hidden,experts,topk);END;
}
int hayate_glm_accumulate(void* out,const void* x,const void* ids,const void* weights,unsigned tokens,unsigned hidden,unsigned topk,unsigned expert,cudaStream_t st) {
 if(!out||!x||!ids||!weights||!tokens||!hidden||!topk)return cudaErrorInvalidValue;
 accumulate<<<(size_t(tokens)*hidden+255)/256,256,0,st>>>((B*)out,(const B*)x,(const int*)ids,(const float*)weights,tokens,hidden,topk,expert);END;
}
int hayate_glm_pool(void* out,const void* keys,const void* gates,const void* ape,unsigned pools,unsigned dim,cudaStream_t st) {
 if(!out||!keys||!gates||!ape||!dim)return cudaErrorInvalidValue;
 if(pools)pool<<<(pools*dim+255)/256,256,0,st>>>((B*)out,(const B*)keys,(const B*)gates,(const B*)ape,pools,dim);END;
}
int hayate_glm_select(void* ids,void* scratch,const void* q,const void* pools,const void* weights,unsigned tokens,unsigned heads,unsigned dim,unsigned budget,cudaStream_t st) {
 if(!ids||!scratch||!q||!pools||!weights||!tokens||!heads||!dim||!budget||budget%4)return cudaErrorInvalidValue;
 select<<<tokens,256,0,st>>>((int*)ids,(float*)scratch,(const B*)q,(const B*)pools,(const B*)weights,tokens,heads,dim,budget);END;
}
int hayate_glm_split(void* k,void* v,const void* packed,unsigned tokens,unsigned heads,unsigned dim,cudaStream_t st) {
 if(!k||!v||!packed||!tokens||!heads||!dim)return cudaErrorInvalidValue;
 split_kv<<<(tokens*heads*dim+255)/256,256,0,st>>>((B*)k,(B*)v,(const B*)packed,tokens,heads,dim);END;
}
int hayate_glm_attention(void* out,void* probs,const void* q,const void* k,const void* v,const void* ids,unsigned tokens,unsigned heads,unsigned dim,unsigned width,cudaStream_t st) {
 if(!out||!probs||!q||!k||!v||!ids||!tokens||!heads||!dim||!width)return cudaErrorInvalidValue;
 attention<<<dim3(tokens,heads),256,0,st>>>((B*)out,(B*)probs,(const B*)q,(const B*)k,(const B*)v,(const int*)ids,tokens,heads,dim,width);END;
}
}

extern "C" int hayate_glm_hc_norm(void* map,void* collapsed,void* out,const void* x,const void* fn,const void* base,const void* scale,const void* w,cudaStream_t st) {
  if(!map||!collapsed||!out||!x||!fn||!base||!scale||!w)return cudaErrorInvalidValue;
  hc_map<<<1,256,0,st>>>((float*)map,(const B*)x,(const B*)fn,(const float*)base,(const float*)scale,4096);
  auto e=cudaGetLastError();if(e)return e;
  hc_collapse_norm<<<1,256,0,st>>>((B*)collapsed,(B*)out,(const B*)x,(const float*)map,(const B*)w,4096);return cudaGetLastError();
}
extern "C" int hayate_glm_kda_gated(void* core,void* out,void* state,const void* q,const void* k,const void* v,const void* decay,const void* beta,const void* gate,const void* w,cudaStream_t st) {
  if(!core||!out||!state||!q||!k||!v||!decay||!beta||!gate||!w)return cudaErrorInvalidValue;
  kda_gated<<<64,128,0,st>>>((B*)core,(B*)out,(float*)state,(const B*)q,(const B*)k,(const B*)v,(const float*)decay,(const B*)beta,(const B*)gate,(const B*)w);return cudaGetLastError();
}
