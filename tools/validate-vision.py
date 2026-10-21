#!/usr/bin/env python3
"""Real-weight encoder parity and image API/batch regression. Run inside the shared GPU slot."""
import argparse,base64,json,pathlib,subprocess,sys,time,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'local/validation/vision'
MODELS={
 'ornith':('qwen',ROOT/'local/vision/ornith',[16,16]),
 'qwen38':('qwen',ROOT/'local/hayate/src/Qwen3.8-Flash-Next-NVFP4',[16,16]),
 'glm53':('glm',ROOT/'local/hayate/src/GLM-5.3-Flash-dealignai-FP8',[16,16]),
}
def run(cmd,**kw):
 print('+',' '.join(map(str,cmd)),flush=True);return subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT,**kw)
def http(port,path,body=None,timeout=1200):
 req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def image(name):return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((OUT/f'{name}.png').read_bytes()).decode()}}
def encoder(name,large=False):
 kind,model,grid=MODELS[name];stem=OUT/(name+('-large' if large else ''));grid=[24,32] if large else grid;source=OUT/(kind+'-large.png' if large else 'shapes.png')
 run([ROOT/'target/release/darask-vision','--model',model,'--kind',kind,'--image',source,'--out',str(stem)+'.f32','--patches',str(stem)+'-patches.f32'])
 ref=run([sys.executable,ROOT/'tools/vision-reference.py','--model',model,'--kind',kind,'--patches',str(stem)+'-patches.f32','--grid',*grid,'--out',str(stem)+'-ref.f32'],capture_output=True,text=True)
 print(ref.stdout,flush=True);pathlib.Path(str(stem)+'-parity.log').write_text(ref.stdout)
 metrics=json.loads(ref.stdout[ref.stdout.index('{'):]);assert metrics['relative_rmse']<.001,metrics
 return metrics

def api(name):
 port={'ornith':18080,'qwen38':18090,'glm53':17870}[name];kind,model,_=MODELS[name]
 if name=='ornith':cmd=[ROOT/'target/release/kogarashi-native','serve','--model',ROOT/'engines/kogarashi/models/Ornith-1.5-122B-Q4_K_M.gguf','--context','4096','--port',str(port)]
 elif name=='qwen38':cmd=[ROOT/'target/release/tsumuji-serve','--model',model,'--ctx','4096','--port',str(port),'--name',name]
 else:cmd=[ROOT/'target/release/hayate','serve','--glm-hyt',ROOT/'local/hayate/hyt/glm53x3-sc260','--glm-source',model,'--glm-context','4096','--glm-vram-gib','84','--gpu-lock','none','--port',str(port)]
 log=(OUT/f'{name}-server.log').open('w');proc=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.time()+600
  while time.time()<deadline:
   if proc.poll() is not None:raise RuntimeError(f'{name} exited: {proc.returncode}; see {log.name}')
   try:http(port,'/v1/models' if name=='glm53' else '/health',timeout=2);break
   except (OSError,ValueError):time.sleep(.5)
  else:raise RuntimeError('server start timed out')
  if name=='glm53':http(port,'/v1/models/load',{'id':'glm53'})
  results=[]
  def ask(parts):
   body={'model':name if name!='ornith' else 'kogarashi-ornith','messages':[{'role':'user','content':parts}],'enable_thinking':False,'temperature':0,'max_tokens':64,'stream':False,'return_token_ids':True}
   start=time.time();v=http(port,'/v1/chat/completions',body);text=v['choices'][0]['message']['content'];results.append({'text':text,'seconds':time.time()-start,'response':v});print(name,repr(text),flush=True);(OUT/f'{name}-api.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');return text.lower()
  question={'type':'text','text':'Name the single dominant color in this image. Answer with exactly one English word.'}
  text_question=[{'type':'text','text':'What is 2 + 2? Answer using only the digit.'}]
  baseline=ask(text_question)
  red=ask([image('red'),question]);assert 'red' in red,results[-1]
  blue=ask([image('blue'),question]);assert 'blue' in blue,results[-1]
  red2=ask([image('red'),question]);assert red==red2,(red,red2)
  text=ask(text_question);assert text==baseline,(baseline,text)
  multi=ask([{'type':'text','text':'Image 1:\n'},image('red'),{'type':'text','text':'\nImage 2:\n'},image('blue'),{'type':'text','text':'Name the color of the first image and the color of the second image, in that order.'}]);assert 'red' in multi and 'blue' in multi and multi.index('red')<multi.index('blue'),results[-1]
  caption=ask([image('shapes'),{'type':'text','text':'Describe the visible shapes and their colors in one sentence.'}]);assert 'red' in caption and 'blue' in caption,results[-1]
  images=OUT/'batch-input';images.mkdir(exist_ok=True)
  for col in ['red','blue']:(images/f'{col}.png').write_bytes((OUT/f'{col}.png').read_bytes())
  dest=OUT/f'{name}-batch.jsonl';dest.unlink(missing_ok=True)
  cmd=[ROOT/'target/release/darask','batch',{'ornith':'kogarashi','qwen38':'tsumuji','glm53':'hayate'}[name],'--input',images,'--out',dest,'--endpoint',f'http://127.0.0.1:{port}/v1','--jobs','2','--max-tokens','32','--prompt',question['text']]
  run(cmd);run(cmd)
  batch=[json.loads(s) for s in dest.read_text().splitlines()];assert len(batch)==2 and all(r['status']=='ok' for r in batch),batch
  for row in batch:assert pathlib.Path(row['image']).stem in row['caption'].lower(),row
  (OUT/f'{name}-api.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
  return {'answers':[v['text'] for v in results],'batch_jobs':2,'batch_resume_rows':len(batch)}
 finally:
  proc.terminate()
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--suite',choices=['all','encoder','api'],default='all');ap.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS));a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 run([sys.executable,ROOT/'tools/vision-reference.py'])
 summary=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {}
 for name in a.models:
  summary.setdefault(name,{})
  if a.suite in ['all','encoder']:summary[name]['encoder']={'small':encoder(name),'large':encoder(name,True)}
  if a.suite in ['all','api']:summary[name]['api']=api(name)
  (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
