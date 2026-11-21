"""Generate artifacts for the complete 92-class v3 training run."""
from __future__ import annotations
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs'/'v3_complete';REPORT=ROOT/'reports'/'v3_complete';DATA=ROOT/'data'/'prepared_v3_complete'
REPORT.mkdir(parents=True,exist_ok=True)
history=json.loads((RUN/'history.json').read_text(encoding='utf-8'))
manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
metrics={key:json.loads((REPORT/f'{key}_robust_metrics.json').read_text(encoding='utf-8'))['conditions']
         for key in ('v2_baseline','v3_partial','v3_complete')}
with (REPORT/'epoch_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)

# Verify per-class coverage and supports for every split.
with (REPORT/'class_coverage.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['kana','train','validation','test'])
    for i,char in enumerate(manifest['classes']):
        w.writerow([char]+[manifest['per_class'][split][char] for split in ('train','val','test')])

x=np.arange(1,len(history)+1)
def smooth(v):return np.convolve(np.asarray(v,float),np.ones(3)/3,mode='valid')
fig,axes=plt.subplots(1,2,figsize=(12,4.6),dpi=170)
for key,label,color in [('train_loss','Train objective','#4C78A8'),('val_loss','Validation loss','#F58518')]:
    vals=[r[key] for r in history];axes[0].plot(x,vals,color=color,alpha=.22)
    axes[0].plot(x[1:-1],smooth(vals),color=color,lw=2,label=f'{label} (3-epoch mean)')
axes[0].set(title='Loss',xlabel='Epoch',ylabel='Train objective / validation cross-entropy');axes[0].grid(alpha=.2);axes[0].legend()
series=[('train_accuracy','Train accuracy','#4C78A8'),('val_accuracy','Validation accuracy','#F58518'),
 ('val_gaussian_020_accuracy','Gaussian σ=0.02','#54A24B'),('val_salt_pepper_010_accuracy','Salt & pepper 1%','#E45756'),
 ('val_low_contrast_045_accuracy','Low contrast','#B279A2')]
for key,label,color in series:
    vals=[r[key] for r in history];axes[1].plot(x,vals,color=color,alpha=.22)
    axes[1].plot(x[1:-1],smooth(vals),color=color,lw=2,label=label)
axes[1].set(title='Accuracy',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.02));axes[1].grid(alpha=.2);axes[1].legend(fontsize=8,loc='lower right')
fig.suptitle('Kana CNN v3 — complete 92-class training',fontsize=13);fig.tight_layout()
fig.savefig(REPORT/'learning_curves.png',bbox_inches='tight');plt.close(fig)

conditions=[('clean','Clean'),('gaussian_020','Gaussian σ=0.02'),('gaussian_050','Gaussian σ=0.05'),
 ('gaussian_100','Gaussian σ=0.10'),('salt_pepper_010','Salt & pepper 1%'),('low_contrast_045','Low contrast 0.45')]
rows=[]
for key,label in conditions:
    row={'condition':label,'samples':json.loads((REPORT/'v3_complete_robust_metrics.json').read_text())['samples']}
    for model in ('v2_baseline','v3_partial','v3_complete'):row[f'{model}_accuracy_pct']=round(metrics[model][key]['accuracy']*100,2)
    row['complete_vs_partial_pp']=round(row['v3_complete_accuracy_pct']-row['v3_partial_accuracy_pct'],2)
    row['complete_vs_v2_pp']=round(row['v3_complete_accuracy_pct']-row['v2_baseline_accuracy_pct'],2)
    rows.append(row)
with (REPORT/'robust_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Robustness comparison — complete 92-class test split','',
 f"Samples: {rows[0]['samples']} across all 92 labels. The original flat font families and system-font families are held out by split.",'',
 '| Condition | V2 baseline | Previous v3 (80 labels) | Complete v3 | Complete vs previous | Complete vs v2 |',
 '|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['condition']} | {r['v2_baseline_accuracy_pct']:.2f}% | {r['v3_partial_accuracy_pct']:.2f}% | {r['v3_complete_accuracy_pct']:.2f}% | {r['complete_vs_partial_pp']:+.2f} pp | {r['complete_vs_v2_pp']:+.2f} pp |")
lines+=['',"This test set includes locally rendered font images for the 12 classes missing from the supplied corpus. It is not a handwritten benchmark. Gaussian σ is measured on normalized [0,1] pixels; low contrast uses `x × 0.45 + 0.55 × 0.25`."]
(REPORT/'robust_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Visualize the new classes and the corruptions the model was trained to handle.
a=np.load(DATA/'test.npz',allow_pickle=False);target=0;idx=int(np.where(a['labels']==target)[0][0]);image=a['images'][idx].astype(np.float32)/255
rng=np.random.default_rng(20261122)
g02=np.clip(image+rng.normal(0,.02,image.shape),0,1);g05=np.clip(image+rng.normal(0,.05,image.shape),0,1)
mask=rng.random(image.shape);sp=np.where(mask<.01,0,np.where(mask>.99,1,image));lc=np.clip(image*.45+.55*.25,0,1)
variants=[('Clean',image),('Gaussian σ=0.02',g02),('Gaussian σ=0.05',g05),('Salt & pepper 1%',sp),('Low contrast 0.45',lc)]
fig,axes=plt.subplots(1,len(variants),figsize=(11,2.7),dpi=170)
for ax,(title,img) in zip(axes,variants):ax.imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=8);ax.axis('off')
fig.suptitle('Newly covered class A — synthetic font render',fontsize=10);fig.tight_layout()
fig.savefig(REPORT/'complete_class_augmentation.png',bbox_inches='tight');plt.close(fig)

model_dir=ROOT/'kana_recognizer'/'models'
partial=model_dir/'v3_partial_80class.pt';partial_meta=model_dir/'v3_partial_80class_metadata.json'
if not partial.exists():shutil.copy2(model_dir/'v3_best.pt',partial)
if not partial_meta.exists():shutil.copy2(model_dir/'v3_metadata.json',partial_meta)
shutil.copy2(RUN/'best.pt',model_dir/'v3_best.pt')
shutil.copy2(RUN/'metadata.json',model_dir/'v3_metadata.json')
best=max(history,key=lambda row:row['val_robust_macro_recall'])
summary={'epochs':len(history),'best_epoch':best['epoch'],'best_validation_robust_macro_recall':best['val_robust_macro_recall'],
 'class_coverage':len(manifest['class_coverage']),'samples':manifest['counts'],'sources':manifest['sources'],
 'hardware':json.loads((RUN/'metadata.json').read_text(encoding='utf-8'))['hardware']}
(REPORT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'best_epoch':best['epoch'],'class_coverage':len(manifest['class_coverage']),'samples':manifest['counts'],'hardware':summary['hardware']}))
