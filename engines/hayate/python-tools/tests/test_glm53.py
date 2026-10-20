import ctypes
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file
from hayate_tools.glm53_source import Source, PREFIX, dequantize
from hayate_tools.glm53_reference import text_config, load_layer, checkpoint_name, trace_layer
from hayate_tools.glm53_convert import plan, write_layer
from hayate_tools.golden import read_tensor

ROOT = Path(__file__).resolve().parents[2]


def config(layers=5):
    return text_config(dict(hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=5, vocab_size=64, pad_token_id=0, n_routed_experts=4, num_experts_per_tok=2,
        num_attention_heads=2, num_key_value_heads=2, q_lora_rank=8, kv_lora_rank=8,
        qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8, index_n_heads=2,
        index_head_dim=8, index_kpool=4, index_topk=8, hc_mult=4,
        linear_attn_config=dict(num_heads=2, head_dim=8, short_conv_kernel_size=4, gate_lower_bound=-5.0),
        layer_types=['deepseek_sparse_attention' if i%4==3 else 'linear_attention' for i in range(layers)],
        mlp_layer_types=['dense']*3+['sparse']*(layers-3),indexer_types=['full']*layers),layers=layers)


def fixture(root, layers=5):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    cfg=config(layers)
    data={}
    torch.manual_seed(53)
    for i in range(layers):
        layer=Glm5NextTextDecoderLayer(cfg,i).to(torch.bfloat16)
        for name, p in layer.named_parameters():
            if name.endswith('.scale'):
                p.data.fill_(0.1)
            else:
                p.data.normal_(std=0.08)
            full=PREFIX+f'layers.{i}.'+checkpoint_name(name)
            if name=='self_attn.conv1d.weight':
                for kind,part in zip('qkv',p.chunk(3)):
                    data[PREFIX+f'layers.{i}.self_attn.{kind}_conv1d.weight']=part.detach().contiguous()
            elif name in ('mlp.experts.gate_up_proj','mlp.experts.down_proj'):
                for e in range(cfg.n_routed_experts):
                    stem=PREFIX+f'layers.{i}.mlp.experts.{e}.'
                    if name.endswith('gate_up_proj'):
                        gate,up=p[e].chunk(2)
                        data[stem+'gate_proj.weight']=gate.detach().contiguous()
                        data[stem+'up_proj.weight']=up.detach().contiguous()
                    else:
                        data[stem+'down_proj.weight']=p[e].detach().contiguous()
            else:
                data[full]=p.detach().contiguous()
        for name,p in layer.named_buffers():
            data[PREFIX+f'layers.{i}.'+checkpoint_name(name)]=p.detach().float().clone()
    for name in list(data):
        if name.endswith(('_base','_scale','A_log','dt_bias')):
            data[name]=data[name].float()
        elif '.mlp.' in name and name.endswith('_proj.weight'):
            # Tiny fixture still exercises the FP8 restoration path.
            value=data[name].float()
            scale=value.abs().max().reshape(1,1)/448
            data[name]=(value/scale).to(torch.float8_e4m3fn)
            data[name.removesuffix('weight')+'weight_scale_inv']=scale
    data[PREFIX+'embed_tokens.weight'] = torch.randn(cfg.vocab_size, cfg.hidden_size).bfloat16()*0.1
    data[PREFIX+'norm.weight'] = torch.ones(cfg.hidden_size).bfloat16()
    data['lm_head.weight'] = torch.randn(cfg.vocab_size, cfg.hidden_size).bfloat16()*0.1
    raw=cfg.to_dict()
    root.joinpath('config.json').write_text(json.dumps({'model_type':'glm5_next','text_config':raw,
        'quantization_config':{'weight_block_size':[128,128]}}))
    save_file(data,str(root/'model.safetensors'))
    return cfg


class GLMCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory()
        cls.work=Path(cls.temp.name)
        cpp=ROOT/'crates/kernels/cu/glm53/math_cpu_test.cpp'
        subprocess.run(['c++','-std=c++20','-O2',str(cpp),'-o',str(cls.work/'math')],check=True)
        subprocess.run([str(cls.work/'math')],check=True)
        subprocess.run(['c++','-std=c++20','-O2','-shared','-fPIC',str(cpp),'-o',str(cls.work/'math.so')],check=True)
        cls.lib=ctypes.CDLL(str(cls.work/'math.so'))
        ptr=ctypes.POINTER(ctypes.c_float)
        cls.lib.glm53_cpu_hc.argtypes=[ptr]*4
        cls.lib.glm53_cpu_fp8.argtypes=[ptr,ptr,ctypes.POINTER(ctypes.c_ubyte),ptr,ctypes.c_uint,ctypes.c_uint]
        cls.lib.glm53_cpu_conv.argtypes=[ptr]*4+[ctypes.c_uint]
        cls.lib.glm53_cpu_step.argtypes=[ptr]*5+[ctypes.c_float,ptr,ctypes.c_uint]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def ptr(array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def test_fp8_block_boundaries(self):
        torch.manual_seed(53)
        w=torch.randn(129,257).to(torch.float8_e4m3fn)
        s=torch.tensor([[0.5,1,2],[3,4,5]],dtype=torch.float32)
        actual=dequantize(w,s)
        expected=torch.empty_like(actual)
        for r in range(129):
            for c in range(257):
                expected[r,c]=w[r,c].float()*s[r//128,c//128]
        self.assertTrue(torch.equal(actual,expected))
        with self.assertRaises(ValueError): dequantize(w,s[:1])
        with self.assertRaises(ValueError): dequantize(w[None],s)

    def test_kda_shared_cuda_math_against_transformers(self):
        from transformers.models.glm5_next.modeling_glm5_next import recurrent_kimi_delta_attention, chunk_kimi_delta_attention
        torch.manual_seed(53)
        errors=[]
        for tokens in (1,3,17,65):
            shape=(1,tokens,2,8)
            q,k,v=[torch.randn(shape).bfloat16() for _ in range(3)]
            g=-torch.rand(shape)*5
            beta=torch.rand(1,tokens,2).bfloat16()
            initial=torch.randn(1,2,8,8)*0.1
            expected,state=recurrent_kimi_delta_attention(q,k,v,g,beta,initial,True,True)
            chunk,chunk_state=chunk_kimi_delta_attention(q,k,v,g,beta,initial_state=initial,output_final_state=True,use_qk_l2norm_in_kernel=True)
            result=torch.empty_like(expected)
            current=initial[0].numpy().copy()
            for t in range(tokens):
                for h in range(2):
                    qq=q[0,t,h].float().numpy().copy();kk=k[0,t,h].float().numpy().copy()
                    qq/=np.sqrt(np.square(qq).sum()+1e-6);kk/=np.sqrt(np.square(kk).sum()+1e-6)
                    vv=v[0,t,h].float().numpy().copy();gg=g[0,t,h].numpy().copy();out=np.empty(8,dtype=np.float32)
                    self.lib.glm53_cpu_step(self.ptr(current[h]),self.ptr(qq),self.ptr(kk),self.ptr(vv),self.ptr(gg),float(beta[0,t,h]),self.ptr(out),8)
                    result[0,t,h]=torch.from_numpy(out)
            for a,b in [(result,expected),(torch.from_numpy(current)[None],state),(chunk,expected),(chunk_state,state)]:
                error=float((a.float()-b.float()).norm()/b.float().norm());errors.append(error);self.assertLess(error,0.01)
            # Recurrent state survives chunk boundaries, including decode.
            first,s1=recurrent_kimi_delta_attention(q[:,:tokens//2],k[:,:tokens//2],v[:,:tokens//2],g[:,:tokens//2],beta[:,:tokens//2],initial,True,True) if tokens>1 else (None,initial)
            offset=tokens//2 if tokens>1 else 0
            last,s2=recurrent_kimi_delta_attention(q[:,offset:],k[:,offset:],v[:,offset:],g[:,offset:],beta[:,offset:],s1,True,True)
            self.assertTrue(torch.equal(s2,state))
        print('GLM CPU KDA maximum relative L2:',max(errors))

    def test_mhc_shared_cuda_math_against_transformers(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection
        torch.manual_seed(53)
        module=Glm5NextTextHyperConnection(config())
        for p in module.parameters():p.data.normal_(std=0.2)
        x=torch.randn(1,3,4,16).bfloat16()
        post,comb,collapse=module(x)
        flat=x.flatten(2).float();flat=flat*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-5)
        logits=torch.nn.functional.linear(flat,module.fn.float()).detach().numpy()
        base=module.base.detach().numpy();scale=module.scale.detach().numpy()
        errors=[]
        for t in range(3):
            output=np.empty(24,dtype=np.float32)
            self.lib.glm53_cpu_hc(self.ptr(output),self.ptr(logits[0,t]),self.ptr(base),self.ptr(scale))
            errors.append(float((torch.from_numpy(output[8:].reshape(4,4))-comb[0,t].detach()).norm()/comb[0,t].detach().norm()))
            torch.testing.assert_close(torch.from_numpy(output[4:8]),post[0,t],rtol=2e-6,atol=2e-7)
            torch.testing.assert_close(torch.from_numpy(output[8:].reshape(4,4)),comb[0,t],rtol=2e-6,atol=2e-7)
            actual=(torch.from_numpy(output[:4])[:,None]*x[0,t]).sum(0).bfloat16()
            self.assertTrue(torch.equal(actual,collapse[0,t]))
            torch.testing.assert_close(comb[0,t].sum(0),torch.ones(4),rtol=2e-6,atol=2e-6)

        print('GLM CPU mHC maximum relative L2:',max(errors))

    def test_router_bias_only_affects_selection(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
        router=Glm5NextTextTopkRouter(config())
        router.weight.data.zero_();router.e_score_correction_bias.copy_(torch.tensor([0,3,1,2]))
        _,weights,ids=router(torch.ones(1,2,16).bfloat16())
        self.assertEqual(set(ids[0].tolist()),{1,3})
        torch.testing.assert_close(weights,torch.full_like(weights,1.25))

    def test_sparse_pools_tail_and_budget(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer
        indexer=Glm5NextTextIndexer(config(),3).bfloat16()
        for p in indexer.parameters():p.data.normal_(std=0.2)
        x=torch.randn(1,17,16).bfloat16();q=torch.randn(1,17,8).bfloat16()
        ids=indexer(x,q,torch.ones(1,17,dtype=torch.bool),None)
        for t,row in enumerate(ids[0]):
            valid=row[row>=0].tolist()
            self.assertEqual(len(valid),len(set(valid)))
            self.assertTrue(all(i<=t for i in valid))
            self.assertEqual(sorted(i for i in valid if i>=(t+1)//4*4),list(range((t+1)//4*4,t+1)))
            self.assertLessEqual(len(valid),8+3)
        self.assertEqual(len(ids[0,-1][ids[0,-1]>=0]),9)
        # Pooling starts at the first real key, including left padding.
        mask=torch.tensor([[False,False]+[True]*15])
        ids=indexer(x,q,mask,None)
        self.assertTrue(torch.all(ids[:,:2]==-1))
        self.assertTrue(torch.all((ids<0)|(ids>=2)))

    def test_native_fp8_and_conv_history(self):
        torch.manual_seed(53)
        weight=torch.randn(129,257).to(torch.float8_e4m3fn)
        scales=torch.rand(2,3)+0.1
        x=torch.randn(257).bfloat16()
        raw=weight.view(torch.uint8).numpy().copy();xx=x.float().numpy();ss=scales.numpy();out=np.empty(129,dtype=np.float32)
        self.lib.glm53_cpu_fp8(self.ptr(out),self.ptr(xx),raw.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),self.ptr(ss),129,257)
        expected=torch.nn.functional.linear(x,dequantize(weight,scales)).float()
        error=float((torch.from_numpy(out)-expected).norm()/expected.norm())
        self.assertLess(error,0.01)
        # Full causal convolution versus one-token updates, including chunk boundaries.
        x=torch.randn(17,16).bfloat16();weights=torch.randn(16,4).bfloat16()
        history=np.zeros((3,16),dtype=np.float32);result=[]
        for row in x:
            out=np.empty(16,dtype=np.float32);xx=row.float().numpy().copy();ww=weights.float().numpy().copy()
            self.lib.glm53_cpu_conv(self.ptr(out),self.ptr(history),self.ptr(xx),self.ptr(ww),16)
            result.append(torch.from_numpy(out))
        expected=torch.nn.functional.silu(torch.nn.functional.conv1d(x.T[None],weights[:,None],padding=3,groups=16)[:,:,:17]).squeeze(0).T.float()
        torch.testing.assert_close(torch.stack(result),expected,rtol=0,atol=0)
        np.testing.assert_array_equal(history,x[-3:].float().numpy())
        print('GLM CPU direct FP8 GEMV relative L2:',error,'conv 17/17 exact')

    def test_streaming_logits_and_cached_decode(self):
        from hayate_tools.glm53_reference import streaming_logits
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
        for layers in [8,16,45]:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);cfg=fixture(root, layers);source=Source(root)
                model=Glm5NextTextModel(cfg).bfloat16().eval()
                model.embed_tokens.weight.data.copy_(source.tensor(PREFIX+'embed_tokens.weight'))
                model.norm.weight.data.copy_(source.tensor(PREFIX+'norm.weight'))
                for index, dest in enumerate(model.layers):
                    loaded=load_layer(source,cfg,index,'cpu')
                    missing,extra=dest.load_state_dict(loaded.state_dict(),strict=False)
                    self.assertFalse(extra)
                    for expert in range(cfg.n_routed_experts) if index>=3 else []:
                        stem=PREFIX+f'layers.{index}.mlp.experts.{expert}.'
                        dest.mlp.experts.gate_up_proj.data[expert].copy_(torch.cat([source.tensor(stem+'gate_proj.weight'),source.tensor(stem+'up_proj.weight')]))
                        dest.mlp.experts.down_proj.data[expert].copy_(source.tensor(stem+'down_proj.weight'))
                    self.assertTrue(all('.experts.' in name for name in missing))
                ids=torch.tensor([[17,19,23,29,31]])
                errors=[]
                with torch.inference_mode():
                    cached=model(ids,use_cache=True)
                    cache=cached.past_key_values
                    for step in range(3):
                        stream=streaming_logits(source,cfg,ids,'cpu')
                        full=model(ids,use_cache=False)
                        expected=torch.nn.functional.linear(full.last_hidden_state[:,-1],source.tensor('lm_head.weight'))[0].float()
                        decode=torch.nn.functional.linear(cached.last_hidden_state[:,-1],source.tensor('lm_head.weight'))[0].float()
                        self.assertTrue(torch.equal(stream,expected))
                        error=float((stream-decode).norm()/stream.norm());errors.append(error)
                        self.assertLess(error,0.01)
                        self.assertEqual(int(stream.argmax()),int(decode.argmax()))
                        next_id=stream.argmax().reshape(1,1)
                        ids=torch.cat([ids,next_id],dim=1)
                        cached=model(next_id,past_key_values=cache,use_cache=True)
                print(f'GLM tiny {layers}-layer streaming/cache logits L2:',errors,'greedy 3/3')

    def test_reference_writes_final_logits_and_steps(self):
        from argparse import Namespace
        from hayate_tools.glm53_reference import generate
        from hayate_tools.glm53_convert import convert
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            out=root/'oracle'
            generate(Namespace(source=root,out=out,layers=5,tokens=7,device='cpu',prompt=None,max_tokens=3,raw=False))
            self.assertEqual(read_tensor(out/'logits')[0].shape,(1,7,64))
            ids=json.loads((out/'ids.json').read_text())
            self.assertEqual(len(ids['input']),7);self.assertEqual(len(ids['generated']),3)
            for step,token in enumerate(ids['generated']):
                logits,_=read_tensor(out/f'step{step}')
                self.assertEqual(int(logits.argmax()),token)
            from hayate_tools.glm53_compare import compare
            actual=root/'actual'
            actual.with_suffix('.ids.json').write_text(json.dumps(ids))
            for step in range(3):
                Path(f'{actual}.step{step}.bin').write_bytes((out/f'step{step}.bin').read_bytes())
            report=compare(out,actual)
            self.assertTrue(all(row['relative_l2']==0 and row['greedy_match'] for row in report))
            ids['generated'][0]+=1
            actual.with_suffix('.ids.json').write_text(json.dumps(ids))
            with self.assertRaises(ValueError):compare(out,actual)
            convert(root,root/'hyt',5)
            self.assertTrue((root/'hyt/globals.hyt').is_file())

    def test_five_layer_fixture_loading_trace_and_hyt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=fixture(root);source=Source(root)
            torch.manual_seed(53);hidden=torch.randn(1,7,4,16).bfloat16()
            for index in range(5):
                layer=load_layer(source,cfg,index,'cpu')
                folder=root/f'layer{index}'
                hidden=trace_layer(layer,hidden,folder)
                got,info=read_tensor(folder/'output')
                self.assertTrue(torch.equal(got,hidden));self.assertEqual(info['dtype'],'BF16')
                entries=plan(source,index)
                write_layer(folder/'weights.hyt',entries,bytes(32))
                raw=(folder/'weights.hyt').read_bytes();self.assertEqual(raw[:4],b'HYT0')
                if index>=3:
                    router_ids,_=read_tensor(folder/'mlp.gate_2');router_weights,_=read_tensor(folder/'mlp.gate_1')
                    value,_=read_tensor(folder/'post_attention_layernorm')
                    # Build the independent eager experts using restored checkpoint weights.
                    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
                    eager=Glm5NextTextExperts(cfg).bfloat16()
                    for e in range(4):
                        stem=PREFIX+f'layers.{index}.mlp.experts.{e}.'
                        eager.gate_up_proj.data[e].copy_(torch.cat([source.tensor(stem+'gate_proj.weight'),source.tensor(stem+'up_proj.weight')]))
                        eager.down_proj.data[e].copy_(source.tensor(stem+'down_proj.weight'))
                    expect=eager(value.reshape(-1,16),router_ids,router_weights)
                    got,_=read_tensor(folder/'mlp.experts')
                    self.assertTrue(torch.equal(expect,got))
            self.assertEqual(hidden.shape,(1,7,4,16))
            # Export never replaces an existing oracle.
            with self.assertRaises(FileExistsError):write_layer(folder/'weights.hyt',entries,bytes(32))


if __name__=='__main__':
    unittest.main()
