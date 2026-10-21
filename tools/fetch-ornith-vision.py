#!/usr/bin/env python3
"""Fetch only Ornith's vision tensors, never the 244 GB language weights (stdlib only)."""
import argparse, concurrent.futures, json, pathlib, struct, urllib.request

def get(url, start=None, end=None):
    headers={}
    if start is not None:
        headers['Range']=f'bytes={start}-{end}'
        url+=f'?download=true&darask_range={start}-{end}'
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=180) as r:
        if start is not None and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')):
            raise RuntimeError('server did not honour the exact range; refused a full model download')
        b=r.read() if start is None else r.read(end-start+2)
    if start is not None and len(b)!=end-start+1: raise RuntimeError('incomplete range')
    return b

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('local/vision/ornith'));ap.add_argument('--revision',default='dfa07bab90e90746ffbfc0b0cba3126eb9265de2');a=ap.parse_args()
    repo='ornith-ai/Ornith-1.5-122B';meta=json.loads(get(f'https://huggingface.co/api/models/{repo}/revision/{a.revision}'));rev=meta['sha'];base=f'https://huggingface.co/{repo}/resolve/{rev}/'
    idx=json.loads(get(base+'model.safetensors.index.json'))['weight_map'];files=sorted({f for n,f in idx.items() if 'visual.' in n})
    tensors=[]
    for f in files:
        u=base+f;length=struct.unpack('<Q',get(u,0,7))[0];header=json.loads(get(u,8,7+length));base_off=8+length
        tensors.extend((n,t,u,base_off) for n,t in header.items() if 'visual.' in n)
    tensors.sort();offset=0;header={};segments=[]
    for name,t,u,base_off in tensors:
        lo,hi=t['data_offsets'];size=hi-lo;header[name]={'dtype':t['dtype'],'shape':t['shape'],'data_offsets':[offset,offset+size]};segments.append((u,base_off+lo,size,offset));offset+=size
    # Individual contiguous pieces allow bounded downloads and modest memory use.
    h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8);a.out.mkdir(parents=True,exist_ok=True)
    tmp=a.out/'vision.safetensors.partial'
    with tmp.open('wb') as out:
        out.write(struct.pack('<Q',len(h)));out.write(h);out.truncate(8+len(h)+offset)
    import os
    fd=os.open(tmp,os.O_WRONLY)
    def fetch(seg):
        u,lo,size,dest=seg
        for at in range(0,size,16<<20):
            n=min(16<<20,size-at);b=get(u,lo+at,lo+at+n-1);written=0
            while written<len(b):written+=os.pwrite(fd,b[written:],8+len(h)+dest+at+written)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i,_ in enumerate(pool.map(fetch,segments),1):
                if i%30==0:print(f'{i}/{len(segments)} tensors',flush=True)
    finally:os.close(fd)
    tmp.replace(a.out/'vision.safetensors')
    for f in ['config.json','preprocessor_config.json']:(a.out/f).write_bytes(get(base+f))
    (a.out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'source':repo,'revision':rev,'total_size':offset},'weight_map':{n:'vision.safetensors' for n in header}},indent=2)+'\n')
    print(f'{len(header)} vision tensors, {offset/2**20:.1f} MiB, revision {rev}',flush=True)
if __name__=='__main__': main()
