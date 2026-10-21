// Native vision operations. Tensor layout and math follow Hugging Face Transformers
// Qwen3_5MoeVisionModel / Glm5NextVisionModel (Apache-2.0); see NOTICE.
#include <cuda_runtime.h>
#include <math.h>
__device__ float reduce(float x) {
  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);
  __shared__ float sm[8];int t=threadIdx.x;if((t&31)==0)sm[t/32]=x;__syncthreads();
  x=t<8?sm[t]:0;for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);
  __shared__ float total;if(t==0)total=x;__syncthreads();return total;
}
extern "C" __global__ void vision_norm(const float*x,float*y,const float*w,const float*b,int h,float eps,int rms){
 int r=blockIdx.x,t=threadIdx.x;float sum=0,sq=0;
 for(int j=t;j<h;j+=256){float v=x[r*h+j];sum+=v;sq+=v*v;}
 float mean=rms?0:reduce(sum)/h;__syncthreads();float sc=rsqrtf(fmaxf(0.f,reduce(sq)/h-mean*mean)+eps);
 for(int j=t;j<h;j+=256)y[r*h+j]=(x[r*h+j]-mean)*sc*w[j]+(b?b[j]:0);
}
// act: 0 identity, 1 tanh GELU, 2 exact GELU. Add bias before activation.
extern "C" __global__ void bias_act(float*x,const float*b,int h,int n,int act){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;float v=x[i]+(b?b[i%h]:0);
 if(act==1)v=0.5f*v*(1+tanhf(0.7978845608028654f*(v+0.044715f*v*v*v)));
 if(act==2)v=0.5f*v*(1+erff(v*0.7071067811865475f));x[i]=v;
}
extern "C" __global__ void add(float*x,const float*y,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]+=y[i];}
extern "C" __global__ void swiglu(float*g,const float*u,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n){float a=fminf(g[i],10.f),b=fminf(10.f,fmaxf(-10.f,u[i]));g[i]=a/(1+expf(-a))*b;}}
// Pack token-major QKV into head-major Q/K/V; QK normalisation is GLM-specific.
extern "C" __global__ void qkv_rope(const float*x,float*y,const float*qn,const float*kn,const int*coords,int n,int h,int heads,float eps){
 int token=blockIdx.x,head=blockIdx.y,kind=blockIdx.z,d=h/heads,t=threadIdx.x;
 __shared__ float v[128];float a=t<d?x[token*3*h+kind*h+head*d+t]:0;
 float scale=1;
 if(qn && kind<2)scale=rsqrtf(reduce(a*a)/d+eps);
 if(t<d)v[t]=a*scale*(qn && kind<2?(kind==0?qn[t]:kn[t]):1);__syncthreads();
 if(t<d){a=v[t];if(kind<2){int j=t%(d/2),axis=j/(d/4);float angle=coords[token*2+axis]*powf(10000.f,-float(j%(d/4))/float(d/4));float sn,cs;sincosf(angle,&sn,&cs);a=t<d/2?v[j]*cs-v[j+d/2]*sn:v[j]*sn+v[j+d/2]*cs;}y[((kind*heads+head)*n+token)*d+t]=a;}
}
extern "C" __global__ void softmax(float*x,int n){
 int row=blockIdx.x,t=threadIdx.x;float mx=-INFINITY;
 for(int j=t;j<n;j+=256)mx=fmaxf(mx,x[row*n+j]);
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 __shared__ float sm[8],m;if((t&31)==0)sm[t/32]=mx;__syncthreads();
 if(t==0){m=sm[0];for(int j=1;j<8;j++)m=fmaxf(m,sm[j]);}__syncthreads();
 float sum=0;for(int j=t;j<n;j+=256)sum+=expf(x[row*n+j]-m);float z=reduce(sum);
 for(int j=t;j<n;j+=256)x[row*n+j]=expf(x[row*n+j]-m)/z;
}
extern "C" __global__ void unpack(const float*x,float*y,int n,int h,int heads){int i=blockIdx.x*256+threadIdx.x;if(i<n*h){int d=h/heads,token=i/h,j=i%h;y[i]=x[((j/d)*n+token)*d+j%d];}}
extern "C" __global__ void downsample_pack(const float*x,float*y,int tokens,int h){int i=blockIdx.x*256+threadIdx.x;if(i<tokens*h*4){int token=i/(h*4),j=i%(h*4);y[i]=x[(token*4+j%4)*h+j/4];}}
