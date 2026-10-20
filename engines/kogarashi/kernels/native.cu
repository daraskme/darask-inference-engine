// Native Ornith decoder. GGML Q4_K/Q6_K layouts follow llama.cpp (MIT, see licenses).
// All arithmetic/launches here belong to Kogarashi; no llama.cpp code is linked.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float sum(float x) {
    for(int d=16;d;d>>=1) x+=__shfl_xor_sync(0xffffffff,x,d);
    return x;
}
__device__ __forceinline__ float blocksum(float x) {
    __shared__ float p[32];
    x=sum(x); if((threadIdx.x&31)==0)p[threadIdx.x/32]=x;
    __syncthreads(); x=threadIdx.x<blockDim.x/32?p[threadIdx.x]:0;
    x=sum(x); if(threadIdx.x==0)p[0]=x; __syncthreads(); return p[0];
}
__device__ __forceinline__ float sigmoid(float x){return 1.f/(1.f+expf(-x));}
__device__ __forceinline__ float silu(float x){return x*sigmoid(x);}
__device__ __forceinline__ float half_at(const uint8_t*p){return __half2float(*reinterpret_cast<const __half*>(p));}
__device__ __forceinline__ float weight(const uint8_t *w,int ty,size_t i){
    if(ty==0)return reinterpret_cast<const float*>(w)[i];
    if(ty==1)return __half2float(reinterpret_cast<const __half*>(w)[i]);
    if(ty==30)return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(w)[i]);
    const int j=i&255;
    if(ty==12){
        const uint8_t*b=w+(i>>8)*144; const uint8_t*s=b+4;
        const int g=j>>5;
        int sc,mi;
        if(g<4){sc=s[g]&63;mi=s[g+4]&63;}
        else{sc=(s[g+4]&15)|((s[g-4]>>6)<<4);mi=(s[g+4]>>4)|((s[g]>>6)<<4);}
        int q=(b[16+(j/64)*32+(j&31)]>>((g&1)*4))&15;
        return __fsub_rn(__fmul_rn(__fmul_rn(half_at(b),float(sc)),float(q)),__fmul_rn(half_at(b+2),float(mi)));
    }
    const uint8_t*b=w+(i>>8)*210;
    const int n=j/128,l=j&31,g=(j%128)/32;
    int q=(b[n*64+l+(g&1)*32]>>((g/2)*4))&15;
    q|=((b[128+n*32+l]>>(g*2))&3)<<4;
    int sc=reinterpret_cast<const int8_t*>(b+192)[n*8+l/16+g*2];
    return __fmul_rn(__fmul_rn(half_at(b+208),float(sc)),float(q-32));
}
extern "C" __global__ void embed(const uint8_t*w,int ty,float*x,const int*token,int h){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<h)x[i]=weight(w,ty,size_t(*token)*h+i);
}
// Four independent rows per block; selected experts share the input and stay on the device.
extern "C" __global__ void gemv(const uint8_t*w,int ty,const float*x,float*y,int k,int rows,const int*ids,int selected){
    int r=blockIdx.x*4+threadIdx.x/32,l=threadIdx.x&31,s=blockIdx.y;
    if(r>=rows)return;
    size_t row=size_t(ids?ids[s]:0)*rows+r;
    const float*in=x+(selected?size_t(s)*k:0);
    float v=0;
    if(ty==12){
        const uint8_t*wr=w+row*(k/256)*144;
        for(int b=0;b<k/256;b++){
            const uint8_t*bl=wr+b*144,*sc=bl+4;
            float d=half_at(bl),dm=half_at(bl+2);
            #pragma unroll
            for(int g=0;g<8;g++){
                int scale=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
                int mi=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
                int q=(bl[16+(g/2)*32+l]>>((g&1)*4))&15;
                float val=__fsub_rn(__fmul_rn(__fmul_rn(d,float(scale)),float(q)),__fmul_rn(dm,float(mi)));
                v=fmaf(val,in[b*256+g*32+l],v);
            }
        }
    }else if(ty==14){
        const uint8_t*wr=w+row*(k/256)*210;
        for(int b=0;b<k/256;b++){
            const uint8_t*bl=wr+b*210;float d=half_at(bl+208);
            #pragma unroll
            for(int n=0;n<2;n++){
                int lo0=bl[n*64+l],lo1=bl[n*64+l+32],hi=bl[128+n*32+l];
                #pragma unroll
                for(int g=0;g<4;g++){
                    int lo=g&1?lo1:lo0;
                    int q=((lo>>((g/2)*4))&15)|(((hi>>(g*2))&3)<<4);
                    int scale=reinterpret_cast<const int8_t*>(bl+192)[n*8+l/16+g*2];
                    float val=__fmul_rn(__fmul_rn(d,float(scale)),float(q-32));
                    v=fmaf(val,in[b*256+n*128+g*32+l],v);
                }
            }
        }
    }else{
        for(int i=l;i<k;i+=32)v=fmaf(weight(w,ty,row*k+i),in[i],v);
    }

    v=sum(v);if(l==0)y[size_t(s)*rows+r]=v;
}
extern "C" __global__ void rms_native(const float*x,const float*w,float*y,int n,float eps){
    float v=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)v=fmaf(x[i],x[i],v);
    float scale=rsqrtf(blocksum(v)/n+eps);
    for(int i=threadIdx.x;i<n;i+=blockDim.x)y[i]=x[i]*scale*w[i];
}
extern "C" __global__ void residual(float*x,const float*y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=y[i];
}
extern "C" __global__ void activate(const float*g,const float*u,float*y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=silu(g[i])*u[i];
}
extern "C" __global__ void route(const float*logits,int*ids,float*prob){
    // Only 256 experts. One warp selects eight with a deterministic lower-id tie break.
    int l=threadIdx.x; float vals[8];for(int i=0;i<8;i++)vals[i]=logits[l+i*32];
    for(int s=0;s<8;s++){
        float best=-INFINITY;int id=0;
        for(int i=0;i<8;i++)if(vals[i]>best){best=vals[i];id=l+i*32;}
        for(int d=16;d;d>>=1){float b=__shfl_xor_sync(0xffffffff,best,d);int j=__shfl_xor_sync(0xffffffff,id,d);if(b>best||(b==best&&j<id)){best=b;id=j;}}
        if(l==0){ids[s]=id;prob[s]=best;}
        if((id&31)==l)vals[id/32]=-INFINITY;
    }
    __syncwarp();float mx=prob[0],z=0;if(l<8)z=expf(prob[l]-mx);z=sum(z);
    if(l<8)prob[l]=expf(prob[l]-mx)/z;
}
extern "C" __global__ void combine(float*x,const float*y,const float*p,const float*shared,const float*sg,int h){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=h)return;
    float v=0;for(int s=0;s<8;s++)v=fmaf(p[s],y[s*h+i],v);
    x[i]+=v+sigmoid(sg[0])*shared[i];
}
extern "C" __global__ void conv(float*qkv,const float*w,float*history,int c){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c)return;
    float*hs=history+3*i;const float*ws=w+4*i;float x=qkv[i];
    float v=fmaf(ws[3],x,fmaf(ws[2],hs[2],fmaf(ws[1],hs[1],ws[0]*hs[0])));
    hs[0]=hs[1];hs[1]=hs[2];hs[2]=x;qkv[i]=silu(v);
}
extern "C" __global__ void qk_l2(float*qkv,int d,float eps){
    int h=blockIdx.x;float*x=qkv+h*d;int i=threadIdx.x;
    float v=x[i];float sc=rsqrtf(blocksum(v*v)+eps);x[i]=v*sc;
}
// One warp owns a value column; state is transposed [head][value][key].
extern "C" __global__ void delta(const float*qkv,const float*alpha,const float*beta,const float*a,const float*dt,float*state,float*out){
    int h=blockIdx.x/32,col=(blockIdx.x%32)*4+threadIdx.x/32,l=threadIdx.x&31;
    int kh=h%16;
    float decay=expf(a[h]*(fmaxf(alpha[h]+dt[h],0.f)+log1pf(expf(-fabsf(alpha[h]+dt[h])))));
    const float*k=qkv+2048+kh*128,*q=qkv+kh*128;
    float*s=state+(size_t(h)*128+col)*128;
    float v[4],kv=0;
    for(int r=0;r<4;r++){int j=r*32+l;v[r]=s[j];kv=fmaf(v[r],k[j],kv);}
    float diff=(qkv[4096+h*128+col]-decay*sum(kv))*sigmoid(beta[h]);
    float y=0;for(int r=0;r<4;r++){int j=r*32+l;v[r]=fmaf(k[j],diff,decay*v[r]);s[j]=v[r];y=fmaf(v[r],q[j],y);}
    y=sum(y);if(l==0)out[h*128+col]=y*rsqrtf(128.f);
}
extern "C" __global__ void gated_norm(float*y,const float*z,const float*w,float eps){
    int h=blockIdx.x,j=threadIdx.x;float v=y[h*128+j];float sc=rsqrtf(blocksum(v*v)/128+eps);
    y[h*128+j]=v*sc*w[j]*silu(z[h*128+j]);
}
extern "C" __global__ void qk_rope(float*qg,float*k,const float*v,const float*qn,const float*kn,float*kc,float*vc,const int*pos,int ctx,float theta,float eps){
    int h=blockIdx.x,j=threadIdx.x;bool isq=h<32;int head=isq?h:h-32;
    float*x=isq?qg+head*512:k+head*256;const float*w=isq?qn:kn;
    float a=x[j];float sc=rsqrtf(blocksum(a*a)/256+eps);
    __shared__ float tmp[256];tmp[j]=a*sc*w[j];__syncthreads();a=tmp[j];
    if(j<64){int p=j%32;float an=(*pos)*powf(theta,-2.f*p/64);float sn,cs;sincosf(an,&sn,&cs);a=j<32?tmp[p]*cs-tmp[p+32]*sn:tmp[p]*sn+tmp[p+32]*cs;}
    x[j]=a;
    if(!isq){kc[(size_t(head)*ctx+*pos)*256+j]=a;vc[(size_t(head)*ctx+*pos)*256+j]=v[head*256+j];}
}
extern "C" __global__ void scores(const float*qg,const float*kc,float*score,const int*pos,int ctx){
    int head=blockIdx.y,t=blockIdx.x*4+threadIdx.x/32,l=threadIdx.x&31;
    for(;t<=*pos;t+=gridDim.x*4){
        float v=0;for(int i=l;i<256;i+=32)v=fmaf(qg[head*512+i],kc[(size_t(head/16)*ctx+t)*256+i],v);
        v=sum(v)*0.0625f;if(l==0)score[head*ctx+t]=v;
    }
}
extern "C" __global__ void softmax(float*score,const int*pos,int ctx){
    int h=blockIdx.x,l=threadIdx.x;float mx=-INFINITY;
    for(int i=l;i<=*pos;i+=256)mx=fmaxf(mx,score[h*ctx+i]);
    __shared__ float part[8];for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    if((l&31)==0)part[l/32]=mx;__syncthreads();mx=l<8?part[l]:-INFINITY;
    for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    __shared__ float maxv;if(l==0)maxv=mx;__syncthreads();mx=maxv;
    float z=0;for(int i=l;i<=*pos;i+=256)z+=expf(score[h*ctx+i]-mx);z=blocksum(z);
    for(int i=l;i<=*pos;i+=256)score[h*ctx+i]=expf(score[h*ctx+i]-mx)/z;
}
extern "C" __global__ void attention(const float*score,const float*vc,const float*qg,float*out,const int*pos,int ctx){
    int h=blockIdx.x,j=threadIdx.x;float y=0;
    for(int i=0;i<=*pos;i++)y=fmaf(score[h*ctx+i],vc[(size_t(h/16)*ctx+i)*256+j],y);
    out[h*256+j]=y*sigmoid(qg[h*512+256+j]);
}
extern "C" __global__ void greedy(const float*logits,int*token,int*pos,int n){
    int l=threadIdx.x;float best=-INFINITY;int id=0;
    for(int i=l;i<n;i+=256)if(logits[i]>best){best=logits[i];id=i;}
    for(int d=16;d;d>>=1){float b=__shfl_xor_sync(0xffffffff,best,d);int j=__shfl_xor_sync(0xffffffff,id,d);if(b>best||(b==best&&j<id)){best=b;id=j;}}
    __shared__ float bs[8];__shared__ int ids[8];if((l&31)==0){bs[l/32]=best;ids[l/32]=id;}__syncthreads();
    if(l==0){best=bs[0];id=ids[0];for(int i=1;i<8;i++)if(bs[i]>best||(bs[i]==best&&ids[i]<id)){best=bs[i];id=ids[i];}*token=id;(*pos)++;}
}
