import argparse,json,pathlib,time
import numpy as np
import torch
from safetensors import safe_open
from PIL import Image,ImageDraw
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision("highest")
ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path);ap.add_argument('--kind',choices=['glm','qwen']);ap.add_argument('--patches',type=pathlib.Path);ap.add_argument('--grid',nargs=2,type=int);ap.add_argument('--out',type=pathlib.Path);a=ap.parse_args()
if a.model is None:
 p=pathlib.Path('local/validation/vision');im=Image.new('RGB',(224,224),'white');d=ImageDraw.Draw(im);d.rectangle((10,40,100,140),fill='red');d.ellipse((130,50,210,130),fill='blue');d.text((20,170),'DARASK 42',fill='black');im.save(p/'shapes.png')
 for name,col in [('red','red'),('blue','blue')]:Image.new('RGB',(224,224),col).save(p/(name+'.png'))
 for kind,size in [('glm',(448,336)),('qwen',(512,384))]:im.resize(size).save(p/(kind+'-large.png'))
 print('fixtures ready');raise SystemExit
if a.kind=='glm':
 from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig as C
 from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel as M
else:
 from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig as C
 from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel as M
config=C(**json.loads((a.model/'config.json').read_text())['vision_config']);config._attn_implementation='sdpa'
model=M(config)
idx=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map'];files=sorted({f for n,f in idx.items() if 'visual.' in n});state={}
for f in files:
 with safe_open(a.model/f,framework='pt',device='cpu') as sf:
  for n in sf.keys():
   if 'visual.' in n:state[n.split('visual.',1)[1]]=sf.get_tensor(n).float()
model.load_state_dict(state,assign=True);del state;model=model.cuda().eval()
pixels=torch.from_numpy(np.fromfile(a.patches,dtype=np.float32).reshape(-1,6*(14 if a.kind=='glm' else 16)**2)).cuda()
grid=torch.tensor([[1,*a.grid]],device='cuda')
with torch.no_grad():
 t=time.time();out=model(pixels,grid_thw=grid).pooler_output;torch.cuda.synchronize();print('reference encode',time.time()-t)
out.float().cpu().numpy().tofile(a.out)
actual=np.fromfile(str(a.out).replace('-ref.f32','.f32'),dtype=np.float32).reshape(out.shape);ref=out.cpu().numpy();diff=actual-ref
print(json.dumps({'shape':list(out.shape),'max_abs':float(np.abs(diff).max()),'rms_diff':float(np.sqrt(np.mean(diff**2))),'relative_rmse':float(np.linalg.norm(diff)/np.linalg.norm(ref)),'cosine':float(np.sum(actual*ref)/(np.linalg.norm(actual)*np.linalg.norm(ref)))},indent=2))
