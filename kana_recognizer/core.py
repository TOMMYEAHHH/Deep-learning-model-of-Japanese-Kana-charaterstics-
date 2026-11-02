"""92-class single-kana recognizer. CLI: prepare, train, evaluate, predict, export.

No training or device initialization occurs on import. Paths are explicit; model
bundles include vocabulary and preprocessing version. See README for limitations.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import zipfile
from collections import Counter
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 1
SIZE = 48
HIRAGANA = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
KATAKANA = ''.join(chr(ord(c)+96) for c in HIRAGANA)
CLASSES = list(HIRAGANA + KATAKANA)
ROMAJI = 'a i u e o ka ki ku ke ko sa shi su se so ta chi tsu te to na ni nu ne no ha hi fu he ho ma mi mu me mo ya yu yo ra ri ru re ro wa wo n'.split()
PREPROCESS = {'version':VERSION,'size':SIZE,'foreground':'white','background':'black','content_size':40,'polarity':'border_median','normalization':'uint8 / 255','crop_threshold':0.12}

def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def preprocess(image: Image.Image) -> np.ndarray:
    """One isolated character; auto polarity, foreground crop, aspect preserving resize."""
    image=ImageOps.exif_transpose(image)
    if 'A' in image.getbands():
        bg=Image.new('RGBA',image.size,'white');bg.alpha_composite(image.convert('RGBA'));image=bg
    a=np.asarray(image.convert('L'),dtype=np.float32)
    if min(a.shape)<2: raise ValueError('Image is too small')
    border=np.concatenate([a[0],a[-1],a[:,0],a[:,-1]])
    if np.median(border)>127:a=255-a
    # Remove scanner background without per-dataset choices.
    lo=float(np.median(np.concatenate([a[0],a[-1],a[:,0],a[:,-1]])))
    hi=float(np.percentile(a,99.5))
    if hi-lo<10: raise ValueError('Blank or insufficient-contrast image')
    a=np.clip((a-lo)/(hi-lo),0,1)
    yy,xx=np.where(a>PREPROCESS['crop_threshold'])
    if len(xx)<4:raise ValueError('No visible character')
    a=a[yy.min():yy.max()+1,xx.min():xx.max()+1]
    h,w=a.shape;scale=40/max(h,w)
    patch=Image.fromarray(np.uint8(a*255)).resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR)
    canvas=Image.new('L',(SIZE,SIZE));canvas.paste(patch,((SIZE-patch.width)//2,(SIZE-patch.height)//2))
    return np.array(canvas)

class KanaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[];inch=1
        for ch in [32,64,128]:
            blocks.extend([nn.Conv2d(inch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(),
                nn.Conv2d(ch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(),nn.MaxPool2d(2)])
            inch=ch
        self.features=nn.Sequential(*blocks)
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(128*6*6,256),nn.ReLU(),nn.Dropout(.35),nn.Linear(256,len(CLASSES)))
    def forward(self,x):return self.classifier(self.features(x))

def device_for(name='auto'):
    if name=='auto':
        name='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    if device.type=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    return device

def prepare(args):
    """Use data provenance and group boundaries; never train on external benchmark."""
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Prepared dataset exists; choose a new output directory')
    records=[];source_info=[];rng=np.random.default_rng(args.seed)
    def add(raw,label,source,split,identity,group):
        image=preprocess(Image.open(io.BytesIO(raw)))
        records.append(dict(image=image,label=label,source=source,split=split,identity=identity,group=group,sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    def zip_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for zpath,source in [(args.hiragana_train,'hiraganani'),(args.hiragana_test,'hiraganani_test')]:
        z=zipfile.ZipFile(zpath)
        names=sorted(n for n in z.namelist() if n.lower().endswith('.png'))
        perclass={i:[] for i in range(46)}
        for n in names:perclass[int(Path(n).stem.split('_')[0])].append(n)
        for label,items in perclass.items():
            if not items:raise ValueError(f'Missing class {label} in {zpath}')
            order=rng.permutation(len(items));nval=max(1,round(len(items)*.15))
            val=set(order[:nval].tolist())
            for i,n in enumerate(items):
                split='test' if source.endswith('_test') else ('val' if i in val else 'train')
                add(z.read(n),label,source,split,n,source+':writer_unknown')
        source_info.append({'source':source,'path':str(zpath),'sha256':zip_hash(zpath),'split':'provided test; stratified validation; writer IDs unavailable'})
    # Read original complete font images, NOT the incomplete all_kana folders.
    fontroot=Path(args.font_root)
    allfonts=set();fontfiles=[]
    mapping=dict(zip(ROMAJI,range(46)))
    mapping.update({'si':11,'ti':16,'tu':17,'hu':27})
    for script,offset in [('hiragana',0),('katakana',46)]:
        for p in sorted((fontroot/script).glob('*.jpeg')):
            parts=p.stem.split('_')
            if parts[0] not in mapping or parts[-1]!='dakufalse':continue
            font='_'.join(parts[1:-1]);allfonts.add(font)
            fontfiles.append((p,mapping[parts[0]]+offset,font))
    fonts=sorted(allfonts);rng.shuffle(fonts)
    n=max(1,round(len(fonts)*.15));fontsplit={f:('test' if i<n else ('val' if i<2*n else 'train')) for i,f in enumerate(fonts)}
    for p,label,font in fontfiles:add(p.read_bytes(),label,'font_synthetic',fontsplit[font],str(p),font)
    source_info.append({'source':'font_synthetic','url':'https://github.com/Orzelius/7500-unique-kana-images','font_groups':fontsplit})
    if args.supplement_fonts:
        # Explicitly synthetic coverage supplement, not new handwriting samples.
        families=[('meiryo.ttc','train'),('msgothic.ttc','train'),('YuGothR.ttc','train'),
            ('YuGothM.ttc','train'),('YuGothB.ttc','train'),('msyh.ttc','train'),
            ('simkai.ttf','train'),('msjh.ttc','train'),('simsun.ttc','val'),
            ('YuMincho.ttc','val'),('msmincho.ttc','test'),('simhei.ttf','test')]
        used=[]
        for filename,split in families:
            path=Path(args.supplement_fonts)/filename
            if not path.exists():continue
            font=ImageFont.truetype(str(path),80)
            used.append({'file':filename,'split':split})
            for label,char in enumerate(CLASSES):
                im=Image.new('L',(128,128));draw=ImageDraw.Draw(im)
                draw.text((64,64),char,font=font,fill=255,anchor='mm')
                b=io.BytesIO();im.save(b,format='PNG')
                add(b.getvalue(),label,'system_font_synthetic',split,f'{filename}:{char}',filename)
        source_info.append({'source':'system_font_synthetic','fonts':used,'note':'Locally rendered, not handwriting. Installed fonts are not redistributed.'})
    # Optional canonical ETL/imported handwriting archive. Each writer stays in one split.
    for path in args.extra_npz:
        data=np.load(path,allow_pickle=False)
        images,labels,writers=data['images'],data['labels'],data['writers']
        if len(images)!=len(labels) or len(labels)!=len(writers):raise ValueError('Extra dataset lengths differ')
        groups=sorted(set(writers.tolist()));rng.shuffle(groups)
        n=max(1,round(len(groups)*.15));groupsplit={w:('test' if i<n else ('val' if i<2*n else 'train')) for i,w in enumerate(groups)}
        for i,(im,label,w) in enumerate(zip(images,labels,writers)):
            char=str(label)
            if char not in CLASSES:continue
            b=io.BytesIO();Image.fromarray(im).save(b,format='PNG')
            add(b.getvalue(),CLASSES.index(char),Path(path).stem,groupsplit[w],f'{path}:{i}',str(w))
        source_info.append({'source':Path(path).stem,'sha256':zip_hash(path),'split':'writer disjoint'})
    # Entire previously inspected benchmark stays external. Unsupported voiced labels skipped.
    if args.external:
        import re
        z=zipfile.ZipFile(args.external)
        for name in sorted(z.namelist()):
            m=re.fullmatch(r'kana([A-Z]+)\d+\.jpg',name)
            if m and m[1].lower() in mapping:add(z.read(name),mapping[m[1].lower()],'inoue_external','external',name,'unknown')
        source_info.append({'source':'inoue_external','sha256':zip_hash(args.external),'note':'Previous model was evaluated here; not a pristine benchmark for development decisions.'})
    # Keep each normalized pixel hash in only one split. Remove conflicting labels.
    hash_labels={}
    for r in records:hash_labels.setdefault(r['sha256'],set()).add(r['label'])
    conflicts={h for h,l in hash_labels.items() if len(l)>1}
    rank={'external':0,'test':1,'val':2,'train':3};seen=set();clean=[];dropped=Counter()
    for r in sorted(records,key=lambda x:rank[x['split']]):
        h=r['sha256']
        if h in conflicts:dropped['conflicting_label_pixels']+=1;continue
        if h in seen:dropped['duplicate_pixels']+=1;continue
        seen.add(h);clean.append(r)
    for split in ['train','val','test','external']:
        rs=[r for r in clean if r['split']==split]
        if not rs:continue
        labels=np.array([r['label'] for r in rs],dtype=np.int64)
        if split in ['train','val'] and len(set(labels))!=92:raise ValueError(f'{split} does not cover all 92 classes')
        np.savez_compressed(out/f'{split}.npz',images=np.stack([r['image'] for r in rs]),labels=labels,
            sources=np.array([r['source'] for r in rs]),ids=np.array([r['identity'] for r in rs]))
    save_json(out/'manifest.json',{'version':VERSION,'classes':CLASSES,'preprocessing':PREPROCESS,'seed':args.seed,
        'sources':source_info,'dropped':dict(dropped),'counts':{s:dict(Counter(r['source'] for r in clean if r['split']==s)) for s in rank},
        'per_class':{s:dict(Counter(CLASSES[r['label']] for r in clean if r['split']==s)) for s in rank},
        'records':[{k:v for k,v in r.items() if k!='image'} for r in clean]})
    print(json.dumps({'counts':{s:sum(r['split']==s for r in clean) for s in rank},'dropped':dict(dropped)}))

def augment(x):
    n=x.shape[0];d=x.device
    angle=(torch.rand(n,device=d)*2-1)*.14
    scale=.9+.2*torch.rand(n,device=d)
    shear=(torch.rand(n,device=d)*2-1)*.08
    theta=torch.zeros(n,2,3,device=d)
    theta[:,0,0]=torch.cos(angle)*scale;theta[:,0,1]=-torch.sin(angle)+shear
    theta[:,1,0]=torch.sin(angle);theta[:,1,1]=torch.cos(angle)*scale
    theta[:,:,2]=(torch.rand(n,2,device=d)*2-1)*.08
    x=F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False,padding_mode='zeros')
    # Stroke width variation, retaining character orientation.
    choice=torch.rand((),device=d).item()
    if choice<.12:x=F.max_pool2d(x,3,1,1)
    elif choice<.24:x=-F.max_pool2d(-x,3,1,1)
    return (x*(.8+.4*torch.rand(n,1,1,1,device=d))).clamp(0,1)

def arrays(path):
    a=np.load(path,allow_pickle=False)
    if a['images'].shape[1:]!=(SIZE,SIZE):raise ValueError('Wrong image dimensions')
    if a['labels'].min()<0 or a['labels'].max()>=92:raise ValueError('Labels outside vocabulary')
    return a

@torch.inference_mode()
def logits_for(model,x,batch=256):
    model.eval()
    return torch.cat([model(x[i:i+batch]).float().cpu() for i in range(0,len(x),batch)])

def metrics(logits,y):
    y=torch.as_tensor(y,dtype=torch.long);p=logits.softmax(1);pred=p.argmax(1)
    cm=torch.bincount(y*92+pred,minlength=92*92).reshape(92,92).numpy()
    support=cm.sum(1);tp=cm.diagonal();precision=tp/np.maximum(cm.sum(0),1);recall=tp/np.maximum(support,1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12);valid=support>0
    return {'samples':len(y),'accuracy':float((pred==y).float().mean()),'macro_recall':float(recall[valid].mean()),
        'macro_f1':float(f1[valid].mean()),'top3_accuracy':float((p.topk(3,1).indices==y[:,None]).any(1).float().mean()),
        'loss':float(F.cross_entropy(logits,y)),'per_class':{CLASSES[i]:{'support':int(support[i]),'recall':float(recall[i]),'precision':float(precision[i]),'f1':float(f1[i])} for i in range(92)},'confusion_matrix':cm.tolist()}

def checkpoint_save(path,payload):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)

def train(args):
    if args.epochs<1 or args.batch_size<1 or args.patience<1 or args.lr<=0:raise ValueError('Invalid training configuration')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:raise ValueError('Output already contains training; use --resume or a new output')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4);device=device_for(args.device)
    manifest=json.loads((Path(args.data)/'manifest.json').read_text(encoding='utf-8'))
    if manifest['classes']!=CLASSES or manifest['preprocessing']!=PREPROCESS:raise ValueError('Incompatible dataset manifest')
    fingerprint=hashlib.sha256((Path(args.data)/'manifest.json').read_bytes()).hexdigest()
    tr=arrays(Path(args.data)/'train.npz');va=arrays(Path(args.data)/'val.npz')
    x=torch.from_numpy(tr['images'][:,None].astype('float32')/255).to(device);y=torch.from_numpy(tr['labels']).to(device)
    vx=torch.from_numpy(va['images'][:,None].astype('float32')/255).to(device);vy=torch.from_numpy(va['labels'])
    model=KanaCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=args.lr*.02)
    history=[];best=-1.;start=0;stale=0
    if args.resume:
        if Path(args.resume).resolve()!= (out/'last.pt').resolve():raise ValueError('Resume must use last.pt in the same output directory')
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        if ck['fingerprint']!=fingerprint:raise ValueError('Resume dataset mismatch')
        if ck['config']['epochs']!=args.epochs:raise ValueError('Resume must preserve scheduled total epochs')
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];best=ck['best'];start=ck['epoch']+1;stale=ck['stale'];torch.set_rng_state(ck['rng'])
        if device.type=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    weights=1/torch.bincount(y,minlength=92).float();weights=weights[y]
    print(f'Device: {device}; train={len(y)}, validation={len(vy)}; classes=92',flush=True)
    for epoch in range(start,args.epochs):
        t=time.time();model.train();total=correct=0;loss_sum=0.
        order=torch.multinomial(weights,len(y),replacement=True)
        for ids in order.split(args.batch_size):
            bx=augment(x[ids]);by=y[ids];opt.zero_grad(set_to_none=True)
            logits=model(bx);loss=F.cross_entropy(logits,by,label_smoothing=.04)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            total+=len(ids);correct+=int((logits.argmax(1)==by).sum());loss_sum+=float(loss.detach())*len(ids)
        val_logits=logits_for(model,vx);vm=metrics(val_logits,vy);scheduler.step()
        row={'epoch':epoch+1,'train_loss':loss_sum/total,'train_accuracy':correct/total,'val_loss':vm['loss'],'val_accuracy':vm['accuracy'],'val_macro_recall':vm['macro_recall'],'seconds':time.time()-t}
        history.append(row);improved=vm['macro_recall']>best
        if improved:best=vm['macro_recall'];stale=0
        else:stale+=1
        payload={'version':VERSION,'state_dict':model.state_dict(),'classes':CLASSES,'preprocessing':PREPROCESS,'temperature':1.,'epoch':epoch,'best':best,'stale':stale,'history':history,
            'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            'fingerprint':fingerprint,'config':vars(args)}
        # argparse callbacks cannot be safely deserialized; store primitives only.
        payload['config']={k:v for k,v in vars(args).items() if k!='func'}
        checkpoint_save(out/'last.pt',payload)
        if improved:checkpoint_save(out/'best.pt',{k:payload[k] for k in ['version','state_dict','classes','preprocessing','temperature','epoch','fingerprint']})
        save_json(out/'history.json',history);print(json.dumps(row),flush=True)
        if stale>=args.patience:break
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'])
    vl=logits_for(model,vx)
    # Temperature chosen only from validation; does not change top-1 classes.
    candidates=np.linspace(.5,3.,101);temperature=float(min(candidates,key=lambda t:float(F.cross_entropy(vl/t,vy))))
    ck['temperature']=temperature;checkpoint_save(out/'best.pt',ck)
    save_json(out/'metadata.json',{'classes':CLASSES,'preprocessing':PREPROCESS,'temperature':temperature,'best_epoch':ck['epoch']+1,'torch':str(torch.__version__),'device':str(device),'data_fingerprint':fingerprint,'scope':'92 basic kana only; one isolated character','sources':manifest['sources'],'counts':manifest['counts']})
    save_json(out/'validation.json',metrics(vl/temperature,vy))
    print(f'Training complete. Best epoch={ck["epoch"]+1}; temperature={temperature}',flush=True)

class KanaRecognizer:
    """Reusable inference API. Scores are validation-temperature-scaled softmax."""
    def __init__(self,model_path,device='auto'):
        self.device=device_for(device)
        ck=torch.load(model_path,map_location='cpu',weights_only=True)
        if ck.get('version')!=VERSION or ck['classes']!=CLASSES or ck['preprocessing']!=PREPROCESS:raise ValueError('Unsupported model bundle')
        self.model=KanaCNN().to(self.device);self.model.load_state_dict(ck['state_dict']);self.model.eval()
        self.temperature=float(ck['temperature'])
    @torch.inference_mode()
    def predict(self,image,top_k=5,threshold=.8):
        if not 1<=top_k<=92 or not 0<=threshold<=1:raise ValueError('Invalid top_k or threshold')
        if not isinstance(image,Image.Image):
            with Image.open(image) as im:a=preprocess(im)
        else:a=preprocess(image)
        x=torch.from_numpy(a[None,None].astype('float32')/255).to(self.device)
        p=(self.model(x)/self.temperature).softmax(1)[0];scores,ids=p.topk(top_k)
        candidates=[{'character':CLASSES[i],'script':'hiragana' if i<46 else 'katakana','score':float(s)} for s,i in zip(scores.cpu().tolist(),ids.cpu().tolist())]
        return {'prediction':candidates[0]['character'],'needs_review':candidates[0]['score']<threshold,'candidates':candidates,
            'note':'Threshold is a review heuristic, not an out-of-distribution guarantee. Isolated へ and ヘ may be visually ambiguous.'}

def evaluate(args):
    r=KanaRecognizer(args.model,args.device);a=arrays(args.data)
    x=torch.from_numpy(a['images'][:,None].astype('float32')/255).to(r.device)
    logits=logits_for(r.model,x)/r.temperature;y=a['labels']
    result=metrics(logits,y);result['by_source']={str(s):metrics(logits[a['sources']==s],y[a['sources']==s]) for s in np.unique(a['sources'])}
    save_json(args.output,result)
    np.savez_compressed(Path(args.output).with_suffix('.npz'),probabilities=logits.softmax(1).numpy(),labels=y,ids=a['ids'],sources=a['sources'])
    print(json.dumps({k:v for k,v in result.items() if k not in ['confusion_matrix','per_class','by_source']}))

def export(args):
    r=KanaRecognizer(args.model,'cpu');out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    example=torch.zeros(1,1,SIZE,SIZE)
    traced=torch.jit.trace(r.model,example);torch.jit.save(traced,str(out/'model.torchscript'))
    save_json(out/'metadata.json',{'version':VERSION,'classes':CLASSES,'preprocessing':PREPROCESS,'temperature':r.temperature,'output':'logits; divide by temperature and softmax','input':'float32 N,1,48,48 preprocessed grayscale'})
    x=torch.rand(2,1,SIZE,SIZE);torch.testing.assert_close(traced(x),r.model(x))
    if hasattr(torch,'export'):
        exported=torch.export.export(r.model,(x,),dynamic_shapes={'x':{0:torch.export.Dim('batch',min=1,max=256)}})
        torch.export.save(exported,out/'model.pt2')
        loaded=torch.export.load(out/'model.pt2').module()
        torch.testing.assert_close(loaded(x),r.model(x))
        torch.testing.assert_close(loaded(x[:1]),r.model(x[:1]))
    print('Exported TorchScript and torch.export; batch-size-1/2 numerical parity passed.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--hiragana-train',required=True);p.add_argument('--hiragana-test',required=True);p.add_argument('--font-root',required=True);p.add_argument('--external');p.add_argument('--extra-npz',action='append',default=[]);p.add_argument('--supplement-fonts');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261029);p.set_defaults(func=prepare)
    p=sub.add_parser('train');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=.001);p.add_argument('--patience',type=int,default=12);p.add_argument('--seed',type=int,default=20261029);p.add_argument('--device',default='auto');p.add_argument('--resume');p.set_defaults(func=train)
    p=sub.add_parser('evaluate');p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='auto');p.set_defaults(func=evaluate)
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--image',required=True);p.add_argument('--top-k',type=int,default=5);p.add_argument('--threshold',type=float,default=.8);p.add_argument('--device',default='auto');p.set_defaults(func=lambda a:print(json.dumps(KanaRecognizer(a.model,a.device).predict(a.image,a.top_k,a.threshold),ensure_ascii=False,indent=2)))
    p=sub.add_parser('export');p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.set_defaults(func=export)
    args=parser.parse_args()
    try:args.func(args)
    except (ValueError,OSError) as exc:parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
