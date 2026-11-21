"""Generate the v3 comparison table, training curves, and augmentation examples."""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports'/'v3'
RUN=ROOT/'runs'/'v3'
REPORT.mkdir(parents=True,exist_ok=True)
history=json.loads((RUN/'history.json').read_text(encoding='utf-8'))
v2=json.loads((REPORT/'v2_baseline_robust_metrics.json').read_text(encoding='utf-8'))['conditions']
v3=json.loads((REPORT/'v3_robust_metrics.json').read_text(encoding='utf-8'))['conditions']
manifest=json.loads((ROOT/'data'/'prepared_v3'/'manifest.json').read_text(encoding='utf-8'))

# Epoch-level table for reproducibility.
with (REPORT/'epoch_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(history[0]))
    writer.writeheader();writer.writerows(history)

def smooth(values,window=3):
    return np.convolve(values,np.ones(window)/window,mode='valid')

epochs=np.array([r['epoch'] for r in history])
fig,axes=plt.subplots(1,2,figsize=(12,4.6),dpi=170)
ax=axes[0]
for key,label,color in [('train_loss','Train loss','#4C78A8'),('val_loss','Validation loss','#F58518')]:
    vals=np.array([r[key] for r in history],float)
    ax.plot(epochs,vals,color=color,alpha=.25,linewidth=1.1)
    ax.plot(epochs[1:-1],smooth(vals),color=color,linewidth=2.2,label=f'{label} (3-epoch mean)')
ax.set(title='Loss',xlabel='Epoch',ylabel='Cross-entropy / combined train objective');ax.grid(alpha=.2);ax.legend()
ax=axes[1]
series=[('train_accuracy','Train accuracy','#4C78A8'),('val_accuracy','Validation accuracy','#F58518'),
        ('val_gaussian_020_accuracy','Validation: Gaussian σ=0.02','#54A24B'),
        ('val_salt_pepper_010_accuracy','Validation: salt & pepper 1%','#E45756'),
        ('val_low_contrast_045_accuracy','Validation: low contrast','#B279A2')]
for key,label,color in series:
    vals=np.array([r[key] for r in history],float)
    ax.plot(epochs,vals,color=color,alpha=.25,linewidth=1)
    ax.plot(epochs[1:-1],smooth(vals),color=color,linewidth=2,label=label)
ax.set(title='Accuracy',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.02));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Kana CNN v3 — font-group validation',fontsize=13)
fig.tight_layout();fig.savefig(REPORT/'learning_curves.png',bbox_inches='tight');plt.close(fig)

conditions=[('clean','Clean'),('gaussian_020','Gaussian σ=0.02'),('gaussian_050','Gaussian σ=0.05'),
            ('gaussian_100','Gaussian σ=0.10'),('salt_pepper_010','Salt & pepper 1%'),('low_contrast_045','Low contrast 0.45')]
rows=[]
for key,label in conditions:
    a=v2[key]['accuracy']*100;b=v3[key]['accuracy']*100
    rows.append({'condition':label,'samples':json.loads((REPORT/'v3_robust_metrics.json').read_text())['samples'],
                 'v2_accuracy_pct':f'{a:.2f}','v3_accuracy_pct':f'{b:.2f}','change_percentage_points':f'{b-a:+.2f}'})
with (REPORT/'robust_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# V2 vs V3 robustness on the held-out test split','',
       f"Samples: {rows[0]['samples']} from {len(manifest['class_coverage'])}/92 classes. Font groups are held out from training; the source is synthetic font renders.",'',
       '| Condition | V2 accuracy | V3 accuracy | Change |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['condition']} | {r['v2_accuracy_pct']}% | {r['v3_accuracy_pct']}% | {r['change_percentage_points']} pp |")
lines+=['','V3 was selected using the validation split. The v2 and v3 checkpoints were evaluated on the same held-out test images. Gaussian noise σ is measured on normalized [0,1] pixels. Low contrast uses `x × 0.45 + 0.55 × 0.25`. Results do not measure handwriting generalization.']
(REPORT/'robust_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Show the exact types of image corruption used for augmentation and evaluation.
data=np.load(ROOT/'data'/'prepared_v3'/'test.npz',allow_pickle=False)
image=data['images'][0].astype(np.float32)/255
rng=np.random.default_rng(20261121)
gaussian02=np.clip(image+rng.normal(0,.02,image.shape),0,1)
gaussian05=np.clip(image+rng.normal(0,.05,image.shape),0,1)
mask=rng.random(image.shape);salt_pepper=np.where(mask<.01,0,np.where(mask>.99,1,image))
low_contrast=np.clip(image*.45+.55*.25,0,1)
combined=np.clip(np.where(rng.random(image.shape)<.01,0,image)+rng.normal(0,.02,image.shape),0,1)
variants=[('Clean',image),('Gaussian σ=0.02',gaussian02),('Gaussian σ=0.05',gaussian05),
          ('Salt & pepper 1%',salt_pepper),('Low contrast 0.45',low_contrast),('Combined',combined)]
fig,axes=plt.subplots(1,len(variants),figsize=(12,2.8),dpi=170)
for ax,(title,img) in zip(axes,variants):
    ax.imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=8);ax.axis('off')
fig.suptitle(f"Example input (class {int(data['labels'][0])}; synthetic font render)",fontsize=10)
fig.tight_layout();fig.savefig(REPORT/'augmentation_examples.png',bbox_inches='tight');plt.close(fig)

# Keep a conveniently loadable v3 model beside the shipped v2 checkpoint.
shutil.copy2(RUN/'best.pt',ROOT/'kana_recognizer'/'models'/'v3_best.pt')
shutil.copy2(RUN/'metadata.json',ROOT/'kana_recognizer'/'models'/'v3_metadata.json')
print('best_epoch',max(history,key=lambda r:r['val_robust_macro_recall'])['epoch'])
print('test_samples',rows[0]['samples'],'class_coverage',len(manifest['class_coverage']))
print('reports',REPORT)
