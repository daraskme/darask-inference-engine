#pragma once
#include "math.cuh"
namespace glm53::lowbit {
// Adjacent to the eight image pointers; no header is added to HYT payloads.
struct MatrixFormat { unsigned char format,grid,group,reserved; };
struct ExpertFormat { MatrixFormat gate_up,down; };
GLM_HD constexpr unsigned bits(unsigned grid) { return grid==2||grid==3?3:2; }
GLM_HD inline size_t scale_start(unsigned n,unsigned k,unsigned grid) { return size_t(n)*k*bits(grid)/8; }
GLM_HD inline size_t storage(unsigned n,unsigned k,MatrixFormat f) {
  return f.format?scale_start(n,k,f.grid)+size_t(n)*k/f.group+4
      :size_t(n)*k/2+size_t((n+127)/128*128)*(k/16)+4;
}
GLM_HD inline size_t stride(unsigned n,unsigned k,MatrixFormat f) { return (storage(n,k,f)+255)/256*256; }
GLM_HD inline unsigned nibble(unsigned grid,unsigned code) {
  // Eight nibbles in registers, indexed with a variable shift (no constant-memory lookup).
  unsigned lut=grid==0?0x000052ad: grid==1?0x000020ac: grid==2?0x5420acdeu:
      grid==3?0x65319bdeu:0x000062aeu;
  return (lut>>(code*4))&15;
}
GLM_HD inline unsigned packed_nibble(unsigned lo,unsigned hi,unsigned strip,unsigned j,unsigned grid,unsigned high_strip) {
  unsigned v=(lo>>(j*8+strip*2))&3;
  if(bits(grid)==3)v|=((hi>>(j*8+high_strip+strip))&1)<<2;
  return nibble(grid,v);
}
GLM_HD inline unsigned packed8(unsigned lo0,unsigned lo1,unsigned hi0,unsigned hi1,unsigned strip,unsigned grid,unsigned high_strip) {
  unsigned out=0;
  #ifdef __CUDACC__
  #pragma unroll
  #endif
  for(unsigned j=0;j<8;++j)
    out|=packed_nibble(j<4?lo0:lo1,j<4?hi0:hi1,strip,j%4,grid,high_strip)<<(j*4);
  return out;
}
GLM_HD inline unsigned code(const unsigned char* image,unsigned r,unsigned c,unsigned n,unsigned k,unsigned grid) {
  unsigned v=(image[size_t(r)*(k/4)+(c/128)*32+c%32]>>(2*(c%128/32)))&3;
  if(bits(grid)==3)v|=((image[size_t(n)*k/4+size_t(r)*(k/8)+(c/256)*32+c%32]>>(c%256/32))&1)<<2;
  return v;
}
// One destination byte, including zero padding and the unmodified FP32 tensor scale.
GLM_HD inline unsigned char convert_byte(const unsigned char* image,size_t i,unsigned n,unsigned k,unsigned grid,unsigned group) {
  size_t payload=size_t(n)*k/2,scales=size_t((n+127)/128*128)*(k/16),start=scale_start(n,k,grid);
  if(i<payload) {
    unsigned r=i/(k/2),c=i%(k/2)*2;
    return nibble(grid,code(image,r,c,n,k,grid))|(nibble(grid,code(image,r,c+1,n,k,grid))<<4);
  }
  i-=payload;
  if(i<scales) {
    unsigned tile=i/512,inner=i%512;
    unsigned r=(tile/(k/64))*128+(inner%16/4)*32+inner/16,c=(tile%(k/64))*4+inner%4;
    return r<n?image[start+size_t(r)*(k/group)+c/(group/16)]:0;
  }
  return image[start+size_t(n)*k/group+i-scales];
}
// H and I are multiples of 256, so all three destination NVFP4 strides agree.
GLM_HD inline unsigned char convert_expert_byte(const unsigned char* image,size_t i,unsigned part,ExpertFormat formats,unsigned hidden,unsigned intermediate) {
  unsigned n=part==2?hidden:intermediate,k=part==2?intermediate:hidden;
  MatrixFormat f=part==2?formats.down:formats.gate_up;
  image+=part*stride(intermediate,hidden,formats.gate_up);
  if(i>=storage(n,k,{0,0,0,0}))return 0;
  return f.format?convert_byte(image,i,n,k,f.grid,f.group):image[i];
}
}
