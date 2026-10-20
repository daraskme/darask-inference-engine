"""Checkpoint-free checks of the actual CUDA primitives; run explicitly on the main GPU."""
import ctypes
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import torch
from hayate_tools.glm53_source import dequantize
from tests.test_glm53 import config

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA GPU')
class GLMGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        source=ROOT/'crates/kernels/cu/glm53/ops.cu'
        out=Path(cls.temp.name)/'glm.so'
        subprocess.run(['nvcc','-shared','-O3','--use_fast_math','-std=c++20',
            '-gencode=arch=compute_120a,code=sm_120a','-Xcompiler=-fPIC',str(source),'-o',str(out)],check=True)
        cls.lib=ctypes.CDLL(str(out))
        for name,args in re.findall(r'int (hayate_glm_\w+)\((.*?)\) \{',source.read_text()):
            fn=getattr(cls.lib,name)
            fn.argtypes=[{'void*':ctypes.c_void_p,'const void*':ctypes.c_void_p,'unsigned':ctypes.c_uint,
                'size_t':ctypes.c_size_t,'int':ctypes.c_int,'cudaStream_t':ctypes.c_void_p}[arg.rsplit(' ',1)[0]] for arg in args.split(',')]
            fn.restype=ctypes.c_int

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        torch.manual_seed(53)
        torch.backends.cuda.matmul.allow_tf32=False

    def call(self,name,*args):
        args=[a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args]
        code=getattr(self.lib,'hayate_glm_'+name)(*args,torch.cuda.current_stream().cuda_stream)
        self.assertEqual(code,0,name)
        torch.cuda.synchronize()

    def close(self,actual,expected):
        self.assertEqual(actual.shape,expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        energy=expected.float().norm()
        error=float((actual.float()-expected.float()).norm()/energy) if energy else float((actual!=expected).any())
        print('GLM GPU relative L2:',error)
        self.assertLess(error,0.01)

    def test_native_fp8_gemv_and_decode_conv(self):
        w=torch.randn(129,257,device='cuda').to(torch.float8_e4m3fn)
        scales=torch.rand(2,3,device='cuda')+0.1;x=torch.randn(257,device='cuda').bfloat16()
        out=torch.empty(129,device='cuda',dtype=torch.bfloat16)
        self.call('fp8_gemv',out,x,w,scales,129,257)
        self.close(out,torch.nn.functional.linear(x,dequantize(w,scales)))
        x=torch.randn(17,16,device='cuda').bfloat16();weight=torch.randn(16,4,device='cuda').bfloat16()
        history=torch.zeros(3,16,device='cuda',dtype=torch.bfloat16);rows=[]
        for row in x:
            out=torch.empty_like(row);self.call('conv_step',out,history,row,weight,16);rows.append(out)
        full=torch.empty_like(x);self.call('conv',full,x,weight,17,16)
        self.close(torch.stack(rows),full)
        self.assertTrue(torch.equal(history,x[-3:]))

    def test_decode_selection_and_attention(self):
        length=2057;np=length//4
        pools=torch.randn(np,128,device='cuda').bfloat16()
        q=torch.randn(32,128,device='cuda').bfloat16();weights=torch.randn(32,device='cuda').bfloat16()
        ids=torch.empty(2051,device='cuda',dtype=torch.int32);scores=torch.empty(np,device='cuda')
        self.call('select_step',ids,scores,q,pools,weights,length)
        score=torch.relu(q.float()@pools.float().T/(128**0.5))*weights.float()[:,None]/(32**0.5)
        chosen=score.sum(0).topk(512).indices
        expected=torch.cat([(chosen[:,None]*4+torch.arange(4,device='cuda')).flatten(),torch.arange(np*4,length,device='cuda')])
        self.assertEqual(set(ids[ids>=0].tolist()),set(expected.tolist()))
        count=7;q=torch.randn(64,256,device='cuda').bfloat16()
        k,v=[torch.randn(count,64,256,device='cuda').bfloat16() for _ in range(2)]
        out=torch.empty_like(q);probs=torch.empty(64,count,device='cuda',dtype=torch.bfloat16)
        self.call('attention_step',out,probs,q,k,v,count)
        expected_prob=torch.softmax(((q[:,None,:]@k.transpose(0,1).transpose(1,2))/16).float(),dim=-1).bfloat16()
        expected=(expected_prob@v.transpose(0,1)).squeeze(1)
        self.close(out,expected)
        self.close(probs,expected_prob.squeeze(1))

    def test_fp8_conv_norm_and_swiglu(self):
        w=torch.randn(129,257,device='cuda').to(torch.float8_e4m3fn)
        scales=torch.rand(2,3,device='cuda')+0.1;out=torch.empty_like(w,dtype=torch.bfloat16)
        self.call('restore',out,w,scales,129,257);self.close(out,dequantize(w,scales))
        x=torch.randn(7,16,device='cuda').bfloat16();weight=torch.randn(16,4,device='cuda').bfloat16();out=torch.empty_like(x)
        expected=torch.nn.functional.silu(torch.nn.functional.conv1d(x.T[None],weight[:,None],padding=3,groups=16)[:,:,:7]).squeeze(0).T
        self.call('conv',out,x,weight,7,16);self.close(out,expected)
        normw=torch.randn(16,device='cuda').bfloat16()
        self.call('norm',out,x,normw,None,7,16,0)
        norm=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).bfloat16()*normw
        self.close(out,norm)
        bias=torch.randn(16,device='cuda').bfloat16()
        self.call('norm',out,x,normw,bias,7,16,1)
        self.close(out,torch.nn.functional.layer_norm(x,(16,),normw,bias,1e-6))
        up=torch.randn_like(x)*15;gate=x*15
        self.call('point',out,gate,up,x.numel(),1)
        self.close(out,torch.nn.functional.silu(gate.clamp(max=10))*up.clamp(-10,10))

    def test_mhc_kda_and_gated_norm(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection, recurrent_kimi_delta_attention, Glm5NextTextRMSNormGated
        module=Glm5NextTextHyperConnection(config()).cuda()
        for p in module.parameters():p.data.normal_(std=0.1)
        module.fn.data=module.fn.data.bfloat16()
        x=torch.randn(1,3,4,16,device='cuda').bfloat16();post,comb,collapsed=module(x)
        maps=torch.empty(3,24,device='cuda');out=torch.empty_like(collapsed)
        self.call('hc',maps,out,x,module.fn,module.base,module.scale,3,16)
        self.close(out,collapsed);self.close(maps[:,4:8],post[0]);self.close(maps[:,8:].reshape(3,4,4),comb[0])
        branch=torch.randn_like(collapsed);expand=torch.empty_like(x)
        expected=post.bfloat16()[...,None]*branch[...,None,:]+comb.bfloat16().transpose(-1,-2)@x
        self.call('expand',expand,x,branch,maps,3,16);self.close(expand,expected)
        q,k,v=[torch.randn(1,17,2,8,device='cuda').bfloat16() for _ in range(3)]
        projection=torch.randn_like(q);dt=torch.randn(2,8,device='cuda');alog=torch.randn(2,device='cuda')
        g=torch.empty_like(q,dtype=torch.float32)
        self.call('decay',g,projection,dt,alog,17,2,8)
        self.close(g,-5*torch.sigmoid(alog.exp()[None,None,:,None]*(projection.float()+dt)))
        beta=torch.rand(1,17,2,device='cuda').bfloat16();initial=torch.randn(1,2,8,8,device='cuda')*0.1
        state=initial.clone();out=torch.empty_like(q)
        expected,expected_state=recurrent_kimi_delta_attention(q,k,v,g,beta,initial,True,True)
        self.call('kda',out,state,q,k,v,g,beta,17,2,8);self.close(out,expected);self.close(state,expected_state)
        gate=torch.randn_like(out);norm=Glm5NextTextRMSNormGated(8,eps=1e-5).cuda().bfloat16()
        norm.weight.data.normal_();gated=torch.empty_like(out)
        self.call('gated',gated,out,gate,norm.weight,34,8);self.close(gated,norm(out,gate))

    def test_sparse_pruning_tail_and_attention(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer
        cfg=config();indexer=Glm5NextTextIndexer(cfg,3).cuda().bfloat16()
        for p in indexer.parameters():p.data.normal_(std=0.1)
        x=torch.randn(1,17,16,device='cuda').bfloat16();resid=torch.randn(1,17,8,device='cuda').bfloat16()
        ids=indexer(x,resid,torch.ones(1,17,device='cuda',dtype=torch.bool),None)
        q=indexer.wq_b(resid).reshape(17,2,8);key=indexer.k_norm(indexer.wk(x)).reshape(17,8)
        gate=torch.nn.functional.linear(x,indexer.index_kpool_compress_gate).reshape(17,8)
        pool_keys,_,_=indexer.get_pooled_states(torch.cat([key[None],gate[None],torch.ones(1,17,1,device='cuda',dtype=torch.bfloat16)],-1))
        pools=torch.empty(4,8,device='cuda',dtype=torch.bfloat16)
        self.call('pool',pools,key,gate,indexer.index_kpool_compress_ape,4,8);self.close(pools,pool_keys[0])
        weights=indexer.weights_proj(x).reshape(17,2);selected=torch.empty(17,11,device='cuda',dtype=torch.int32)
        scratch=torch.empty(17,4,device='cuda');self.call('select',selected,scratch,q,pools,weights,17,2,8,8)
        for t in range(17):self.assertEqual(sorted(selected[t][selected[t]>=0].tolist()),sorted(ids[0,t][ids[0,t]>=0].tolist()))
        q,k,v=[torch.randn(17,2,8,device='cuda').bfloat16() for _ in range(3)]
        output=torch.empty_like(v);prob=torch.empty(17,2,17,device='cuda',dtype=torch.bfloat16)
        self.call('attention',output,prob,q,k,v,selected,17,2,8,11)
        scores=(q.transpose(0,1)@k.transpose(0,1).transpose(-1,-2))/8**0.5
        mask=torch.zeros(17,17,device='cuda',dtype=torch.bool)
        for t in range(17):mask[t,selected[t][selected[t]>=0].long()]=True
        p=scores.masked_fill(~mask,torch.finfo(torch.bfloat16).min).float().softmax(-1).bfloat16()
        expect=(p@v.transpose(0,1)).transpose(0,1)
        self.close(output,expect)

    def test_router_and_expert_accumulation(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
        cfg=config();router=Glm5NextTextTopkRouter(cfg).cuda().bfloat16()
        router.e_score_correction_bias=router.e_score_correction_bias.float()
        router.weight.data.normal_();router.e_score_correction_bias.normal_()
        x=torch.randn(1,3,16,device='cuda').bfloat16();logits,weights,ids=router(x)
        got_logits=torch.empty_like(logits);got_weights=torch.empty_like(weights);got_ids=torch.empty_like(ids,dtype=torch.int32)
        self.call('route',got_logits,got_weights,got_ids,x,router.weight,router.e_score_correction_bias,3,16,4,2)
        self.close(got_logits,logits)
        for t in range(3):
            ref=dict(zip(ids[t].tolist(),weights[t].tolist()));actual=dict(zip(got_ids[t].tolist(),got_weights[t].tolist()))
            self.assertEqual(ref.keys(),actual.keys())
            self.close(torch.tensor([actual[k] for k in sorted(ref)]),torch.tensor([ref[k] for k in sorted(ref)]))
        result=torch.zeros(3,16,device='cuda',dtype=torch.bfloat16);expect=result.clone()
        for e in range(4):
            value=torch.randn_like(result)
            self.call('accumulate',result,value,got_ids,got_weights,3,16,2,e)
            for t in range(3):
                for j in range(2):
                    if int(got_ids[t,j])==e:expect[t]+=(value[t].float()*got_weights[t,j]).bfloat16()
        self.assertTrue(torch.equal(result,expect))


if __name__=='__main__':unittest.main()
