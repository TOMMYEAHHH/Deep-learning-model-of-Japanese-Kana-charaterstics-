"""Live training dashboard and honest historical replay. See README.md."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F
from kana_predict import (VERSION, SIZE, HIRAGANA, CLASSES, ROMAJI, PREPROCESS,
                          KanaCNN, KanaRecognizer, preprocess, device_for)
def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def prepare(args):
    """Use data provenance and group boundaries; never train on external benchmark."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('Prepared dataset exists; choose a new output directory')
    records = []
    source_info = []
    rng = np.random.default_rng(args.seed)

    def add(raw, label, source, split, identity, group):
        image = preprocess(Image.open(io.BytesIO(raw)))
        records.append(dict(image=image, label=label, source=source, split=split, identity=identity, group=group, sha256=hashlib.sha256(image.tobytes()).hexdigest()))

    def zip_hash(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for zpath, source in [(args.hiragana_train, 'hiraganani'), (args.hiragana_test, 'hiraganani_test')]:
        z = zipfile.ZipFile(zpath)
        names = sorted((n for n in z.namelist() if n.lower().endswith('.png')))
        perclass = {i: [] for i in range(46)}
        for n in names:
            perclass[int(Path(n).stem.split('_')[0])].append(n)
        for label, items in perclass.items():
            if not items:
                raise ValueError(f'Missing class {label} in {zpath}')
            order = rng.permutation(len(items))
            nval = max(1, round(len(items) * 0.15))
            val = set(order[:nval].tolist())
            for i, n in enumerate(items):
                split = 'test' if source.endswith('_test') else 'val' if i in val else 'train'
                add(z.read(n), label, source, split, n, source + ':writer_unknown')
        source_info.append({'source': source, 'path': str(zpath), 'sha256': zip_hash(zpath), 'split': 'provided test; stratified validation; writer IDs unavailable'})
    fontroot = Path(args.font_root)
    allfonts = set()
    fontfiles = []
    mapping = dict(zip(ROMAJI, range(46)))
    mapping.update({'si': 11, 'ti': 16, 'tu': 17, 'hu': 27})
    for script, offset in [('hiragana', 0), ('katakana', 46)]:
        for p in sorted((fontroot / script).glob('*.jpeg')):
            parts = p.stem.split('_')
            if parts[0] not in mapping or parts[-1] != 'dakufalse':
                continue
            font = '_'.join(parts[1:-1])
            allfonts.add(font)
            fontfiles.append((p, mapping[parts[0]] + offset, font))
    fonts = sorted(allfonts)
    rng.shuffle(fonts)
    n = max(1, round(len(fonts) * 0.15))
    fontsplit = {f: 'test' if i < n else 'val' if i < 2 * n else 'train' for i, f in enumerate(fonts)}
    for p, label, font in fontfiles:
        add(p.read_bytes(), label, 'font_synthetic', fontsplit[font], str(p), font)
    source_info.append({'source': 'font_synthetic', 'url': 'https://github.com/Orzelius/7500-unique-kana-images', 'font_groups': fontsplit})
    if args.supplement_fonts:
        families = [('meiryo.ttc', 'train'), ('msgothic.ttc', 'train'), ('YuGothR.ttc', 'train'), ('YuGothM.ttc', 'train'), ('YuGothB.ttc', 'train'), ('msyh.ttc', 'train'), ('simkai.ttf', 'train'), ('msjh.ttc', 'train'), ('simsun.ttc', 'val'), ('YuMincho.ttc', 'val'), ('msmincho.ttc', 'test'), ('simhei.ttf', 'test')]
        used = []
        for filename, split in families:
            path = Path(args.supplement_fonts) / filename
            if not path.exists():
                continue
            font = ImageFont.truetype(str(path), 80)
            used.append({'file': filename, 'split': split})
            for label, char in enumerate(CLASSES):
                im = Image.new('L', (128, 128))
                draw = ImageDraw.Draw(im)
                draw.text((64, 64), char, font=font, fill=255, anchor='mm')
                b = io.BytesIO()
                im.save(b, format='PNG')
                add(b.getvalue(), label, 'system_font_synthetic', split, f'{filename}:{char}', filename)
        source_info.append({'source': 'system_font_synthetic', 'fonts': used, 'note': 'Locally rendered, not handwriting. Installed fonts are not redistributed.'})
    for path in args.extra_npz:
        data = np.load(path, allow_pickle=False)
        images, labels, writers = (data['images'], data['labels'], data['writers'])
        if len(images) != len(labels) or len(labels) != len(writers):
            raise ValueError('Extra dataset lengths differ')
        groups = sorted(set(writers.tolist()))
        rng.shuffle(groups)
        n = max(1, round(len(groups) * 0.15))
        groupsplit = {w: 'test' if i < n else 'val' if i < 2 * n else 'train' for i, w in enumerate(groups)}
        for i, (im, label, w) in enumerate(zip(images, labels, writers)):
            char = str(label)
            if char not in CLASSES:
                continue
            b = io.BytesIO()
            Image.fromarray(im).save(b, format='PNG')
            add(b.getvalue(), CLASSES.index(char), Path(path).stem, groupsplit[w], f'{path}:{i}', str(w))
        source_info.append({'source': Path(path).stem, 'sha256': zip_hash(path), 'split': 'writer disjoint'})
    if args.external:
        import re
        z = zipfile.ZipFile(args.external)
        for name in sorted(z.namelist()):
            m = re.fullmatch('kana([A-Z]+)\\d+\\.jpg', name)
            if m and m[1].lower() in mapping:
                add(z.read(name), mapping[m[1].lower()], 'inoue_external', 'external', name, 'unknown')
        source_info.append({'source': 'inoue_external', 'sha256': zip_hash(args.external), 'note': 'Previous model was evaluated here; not a pristine benchmark for development decisions.'})
    hash_labels = {}
    for r in records:
        hash_labels.setdefault(r['sha256'], set()).add(r['label'])
    conflicts = {h for h, l in hash_labels.items() if len(l) > 1}
    rank = {'external': 0, 'test': 1, 'val': 2, 'train': 3}
    seen = set()
    clean = []
    dropped = Counter()
    for r in sorted(records, key=lambda x: rank[x['split']]):
        h = r['sha256']
        if h in conflicts:
            dropped['conflicting_label_pixels'] += 1
            continue
        if h in seen:
            dropped['duplicate_pixels'] += 1
            continue
        seen.add(h)
        clean.append(r)
    for split in ['train', 'val', 'test', 'external']:
        rs = [r for r in clean if r['split'] == split]
        if not rs:
            continue
        labels = np.array([r['label'] for r in rs], dtype=np.int64)
        if split in ['train', 'val'] and len(set(labels)) != 92:
            raise ValueError(f'{split} does not cover all 92 classes')
        np.savez_compressed(out / f'{split}.npz', images=np.stack([r['image'] for r in rs]), labels=labels, sources=np.array([r['source'] for r in rs]), ids=np.array([r['identity'] for r in rs]))
    save_json(out / 'manifest.json', {'version': VERSION, 'classes': CLASSES, 'preprocessing': PREPROCESS, 'seed': args.seed, 'sources': source_info, 'dropped': dict(dropped), 'counts': {s: dict(Counter((r['source'] for r in clean if r['split'] == s))) for s in rank}, 'per_class': {s: dict(Counter((CLASSES[r['label']] for r in clean if r['split'] == s))) for s in rank}, 'records': [{k: v for k, v in r.items() if k != 'image'} for r in clean]})
    print(json.dumps({'counts': {s: sum((r['split'] == s for r in clean)) for s in rank}, 'dropped': dict(dropped)}))

def augment(x):
    n = x.shape[0]
    d = x.device
    angle = (torch.rand(n, device=d) * 2 - 1) * 0.14
    scale = 0.9 + 0.2 * torch.rand(n, device=d)
    shear = (torch.rand(n, device=d) * 2 - 1) * 0.08
    theta = torch.zeros(n, 2, 3, device=d)
    theta[:, 0, 0] = torch.cos(angle) * scale
    theta[:, 0, 1] = -torch.sin(angle) + shear
    theta[:, 1, 0] = torch.sin(angle)
    theta[:, 1, 1] = torch.cos(angle) * scale
    theta[:, :, 2] = (torch.rand(n, 2, device=d) * 2 - 1) * 0.08
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False, padding_mode='zeros')
    choice = torch.rand((), device=d).item()
    if choice < 0.12:
        x = F.max_pool2d(x, 3, 1, 1)
    elif choice < 0.24:
        x = -F.max_pool2d(-x, 3, 1, 1)
    return (x * (0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=d))).clamp(0, 1)

def arrays(path):
    a = np.load(path, allow_pickle=False)
    if a['images'].shape[1:] != (SIZE, SIZE):
        raise ValueError('Wrong image dimensions')
    if a['labels'].min() < 0 or a['labels'].max() >= 92:
        raise ValueError('Labels outside vocabulary')
    return a

@torch.inference_mode()
def logits_for(model, x, batch=256):
    model.eval()
    return torch.cat([model(x[i:i + batch]).float().cpu() for i in range(0, len(x), batch)])

def metrics(logits, y):
    y = torch.as_tensor(y, dtype=torch.long)
    p = logits.softmax(1)
    pred = p.argmax(1)
    cm = torch.bincount(y * 92 + pred, minlength=92 * 92).reshape(92, 92).numpy()
    support = cm.sum(1)
    tp = cm.diagonal()
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    valid = support > 0
    return {'samples': len(y), 'accuracy': float((pred == y).float().mean()), 'macro_recall': float(recall[valid].mean()), 'macro_f1': float(f1[valid].mean()), 'top3_accuracy': float((p.topk(3, 1).indices == y[:, None]).any(1).float().mean()), 'loss': float(F.cross_entropy(logits, y)), 'per_class': {CLASSES[i]: {'support': int(support[i]), 'recall': float(recall[i]), 'precision': float(precision[i]), 'f1': float(f1[i])} for i in range(92)}, 'confusion_matrix': cm.tolist()}

def checkpoint_save(path, payload):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)

def train(args, observer=None):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or (args.lr <= 0):
        raise ValueError('Invalid training configuration')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and (not args.resume):
        raise ValueError('Output already contains training; use --resume or a new output')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = device_for(args.device)
    manifest = json.loads((Path(args.data) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['classes'] != CLASSES or manifest['preprocessing'] != PREPROCESS:
        raise ValueError('Incompatible dataset manifest')
    fingerprint = hashlib.sha256((Path(args.data) / 'manifest.json').read_bytes()).hexdigest()
    tr = arrays(Path(args.data) / 'train.npz')
    va = arrays(Path(args.data) / 'val.npz')
    x = torch.from_numpy(tr['images'][:, None].astype('float32') / 255).to(device)
    y = torch.from_numpy(tr['labels']).to(device)
    vx = torch.from_numpy(va['images'][:, None].astype('float32') / 255).to(device)
    vy = torch.from_numpy(va['labels'])
    model = KanaCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.02)
    history = []
    best = -1.0
    start = 0
    stale = 0
    if args.resume:
        if Path(args.resume).resolve() != (out / 'last.pt').resolve():
            raise ValueError('Resume must use last.pt in the same output directory')
        ck = torch.load(args.resume, map_location='cpu', weights_only=True)
        if ck['fingerprint'] != fingerprint:
            raise ValueError('Resume dataset mismatch')
        if ck['config']['epochs'] != args.epochs:
            raise ValueError('Resume must preserve scheduled total epochs')
        model.load_state_dict(ck['state_dict'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        history = ck['history']
        best = ck['best']
        start = ck['epoch'] + 1
        stale = ck['stale']
        torch.set_rng_state(ck['rng'])
        if device.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
    weights = 1 / torch.bincount(y, minlength=92).float()
    weights = weights[y]
    print(f'Device: {device}; train={len(y)}, validation={len(vy)}; classes=92', flush=True)
    for epoch in range(start, args.epochs):
        t = time.time()
        if observer is not None:
            observer.pump()
        model.train()
        total = correct = 0
        loss_sum = 0.0
        order = torch.multinomial(weights, len(y), replacement=True)
        for ids in order.split(args.batch_size):
            bx = augment(x[ids])
            by = y[ids]
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by, label_smoothing=0.04)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += len(ids)
            correct += int((logits.argmax(1) == by).sum())
            loss_sum += float(loss.detach()) * len(ids)
        val_logits = logits_for(model, vx)
        vm = metrics(val_logits, vy)
        scheduler.step()
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / total, 'train_accuracy': correct / total, 'val_loss': vm['loss'], 'val_accuracy': vm['accuracy'], 'val_macro_recall': vm['macro_recall'], 'seconds': time.time() - t}
        history.append(row)
        if observer is not None:
            observer.update(history, model, vx, vy)
        improved = vm['macro_recall'] > best
        if improved:
            best = vm['macro_recall']
            stale = 0
        else:
            stale += 1
        payload = {'version': VERSION, 'state_dict': model.state_dict(), 'classes': CLASSES, 'preprocessing': PREPROCESS, 'temperature': 1.0, 'epoch': epoch, 'best': best, 'stale': stale, 'history': history, 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, 'fingerprint': fingerprint, 'config': vars(args)}
        payload['config'] = {k: v for k, v in vars(args).items() if k != 'func'}
        checkpoint_save(out / 'last.pt', payload)
        if improved:
            checkpoint_save(out / 'best.pt', {k: payload[k] for k in ['version', 'state_dict', 'classes', 'preprocessing', 'temperature', 'epoch', 'fingerprint']})
        save_json(out / 'history.json', history)
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(ck['state_dict'])
    vl = logits_for(model, vx)
    candidates = np.linspace(0.5, 3.0, 101)
    temperature = float(min(candidates, key=lambda t: float(F.cross_entropy(vl / t, vy))))
    ck['temperature'] = temperature
    checkpoint_save(out / 'best.pt', ck)
    save_json(out / 'metadata.json', {'classes': CLASSES, 'preprocessing': PREPROCESS, 'temperature': temperature, 'best_epoch': ck['epoch'] + 1, 'torch': str(torch.__version__), 'device': str(device), 'data_fingerprint': fingerprint, 'scope': '92 basic kana only; one isolated character', 'sources': manifest['sources'], 'counts': manifest['counts']})
    save_json(out / 'validation.json', metrics(vl / temperature, vy))
    print(f"Training complete. Best epoch={ck['epoch'] + 1}; temperature={temperature}", flush=True)

def evaluate(args):
    r = KanaRecognizer(args.model, args.device)
    a = arrays(args.data)
    x = torch.from_numpy(a['images'][:, None].astype('float32') / 255).to(r.device)
    logits = logits_for(r.model, x) / r.temperature
    y = a['labels']
    result = metrics(logits, y)
    result['by_source'] = {str(s): metrics(logits[a['sources'] == s], y[a['sources'] == s]) for s in np.unique(a['sources'])}
    save_json(args.output, result)
    np.savez_compressed(Path(args.output).with_suffix('.npz'), probabilities=logits.softmax(1).numpy(), labels=y, ids=a['ids'], sources=a['sources'])
    print(json.dumps({k: v for k, v in result.items() if k not in ['confusion_matrix', 'per_class', 'by_source']}))

def export(args):
    r = KanaRecognizer(args.model, 'cpu')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 1, SIZE, SIZE)
    traced = torch.jit.trace(r.model, example)
    torch.jit.save(traced, str(out / 'model.torchscript'))
    save_json(out / 'metadata.json', {'version': VERSION, 'classes': CLASSES, 'preprocessing': PREPROCESS, 'temperature': r.temperature, 'output': 'logits; divide by temperature and softmax', 'input': 'float32 N,1,48,48 preprocessed grayscale'})
    x = torch.rand(2, 1, SIZE, SIZE)
    torch.testing.assert_close(traced(x), r.model(x))
    if hasattr(torch, 'export'):
        exported = torch.export.export(r.model, (x,), dynamic_shapes={'x': {0: torch.export.Dim('batch', min=1, max=256)}})
        torch.export.save(exported, out / 'model.pt2')
        loaded = torch.export.load(out / 'model.pt2').module()
        torch.testing.assert_close(loaded(x), r.model(x))
        torch.testing.assert_close(loaded(x[:1]), r.model(x[:1]))
    print('Exported TorchScript and torch.export; batch-size-1/2 numerical parity passed.')


BASE = Path(__file__).resolve().parent


def setup_plot(headless=False):
    import matplotlib
    if headless:
        matplotlib.use('Agg')
    from matplotlib import font_manager
    for name in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/meiryo.ttc']:
        if Path(name).exists():
            font_manager.fontManager.addfont(name)
            matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=name).get_name()
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    return plt


class TrainingDashboard:
    """Four panels: accuracy, loss, fixed validation predictions and status."""
    def __init__(self, output, headless=False):
        self.plt = setup_plot(headless)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.indices = None
        self.fig = self.plt.figure(figsize=(13, 8), facecolor='#f4f6fa')
        layout = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
        self.acc = self.fig.add_subplot(layout[0, 0])
        self.loss = self.fig.add_subplot(layout[0, 1])
        self.samples = [self.fig.add_subplot(cell) for cell in layout[1, 0].subgridspec(2, 4)]
        self.status = self.fig.add_subplot(layout[1, 1])
        if not headless:
            self.plt.ion()
            self.plt.show(block=False)

    def pump(self):
        # Closing the window disables live drawing, but training and PNG saving continue.
        if not self.headless and self.plt.fignum_exists(self.fig.number):
            self.plt.pause(.001)

    def update(self, history, model, images, labels):
        row = history[-1]
        epochs = [h['epoch'] for h in history]
        for ax, key, title in [(self.acc, 'accuracy', 'Accuracy / 正確率'), (self.loss, 'loss', 'Loss / 損失')]:
            ax.clear()
            ax.plot(epochs, [h['train_'+key] for h in history], color='#2463eb', label='Train (augmented)')
            ax.plot(epochs, [h['val_'+key] for h in history], color='#f28e2b', label='Validation')
            ax.set(xlabel='Epoch', title=title)
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
        self.acc.set_ylim(0, 1.04)
        if self.indices is None:
            # One fixed example per evenly spaced class; no picking by prediction quality.
            available = sorted(set(labels.tolist()))
            classes = [available[i] for i in np.linspace(0, len(available)-1, 8, dtype=int)]
            self.indices = [int(torch.where(labels==c)[0][0]) for c in classes]
        with torch.inference_mode():
            model.eval()
            probs = model(images[self.indices]).softmax(1).cpu()
        for ax, index, p in zip(self.samples, self.indices, probs):
            ax.clear()
            truth = int(labels[index]); prediction = int(p.argmax())
            ax.imshow(images[index, 0].detach().cpu(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'T:{CLASSES[truth]} P:{CLASSES[prediction]}\n{float(p.max()):.1%}', fontsize=10,
                         color='#18794e' if truth==prediction else '#c53636')
            ax.axis('off')
        best = max(history, key=lambda h: h['val_macro_recall'])
        self.status.clear(); self.status.axis('off')
        lines = [f"LIVE TRAINING · Epoch {row['epoch']}",
                 f"Train accuracy      {row['train_accuracy']:.2%}",
                 f"Validation accuracy {row['val_accuracy']:.2%}",
                 f"Validation macro recall {row['val_macro_recall']:.2%}",
                 f"Best epoch (macro recall): {best['epoch']}",
                 f"Epoch compute time: {row['seconds']:.2f}s", '',
                 'Fixed validation samples; green=correct, red=wrong.',
                 'Training loss uses augmentation + label smoothing.',
                 'Validation loss uses neither; not directly equivalent.',
                 'Test/external datasets are not used in this dashboard.',
                 '92 basic kana; handwritten katakana not fully validated.']
        self.status.text(.02, .97, '\n'.join(lines), va='top', fontsize=10, linespacing=1.6)
        self.fig.suptitle('Kana CNN · Training dashboard', fontsize=18, fontweight='bold')
        self.fig.tight_layout(rect=(0, 0, 1, .96))
        self.fig.savefig(self.output/'training_dashboard.png', dpi=140)
        self.pump()

    def finish(self):
        if not self.headless and self.plt.fignum_exists(self.fig.number):
            self.plt.ioff()
            self.plt.show()
        self.plt.close(self.fig)


def replay(args):
    """Replay genuine logged metrics. This does not reconstruct historical weights."""
    history = json.loads(Path(args.history).read_text(encoding='utf-8'))
    required = {'epoch', 'train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'val_macro_recall'}
    if not history or any(not required.issubset(row) for row in history):
        raise ValueError('History is empty or has missing metric fields')
    if args.interval <= 0:
        raise ValueError('Replay interval must be positive')
    plt = setup_plot(args.headless)
    from matplotlib.widgets import Slider, Button
    fig, (acc, loss) = plt.subplots(1, 2, figsize=(12, 6), facecolor='#f4f6fa')
    fig.subplots_adjust(bottom=.31, top=.79, wspace=.25)
    status = fig.text(.08, .87, '', fontsize=11)
    fig.suptitle('RECORDED TRAINING · 历史训练回放', fontsize=18, fontweight='bold')
    fig.text(.08, .19, 'Real history.json metrics; no training is running. Historical predictions are not reconstructed.', fontsize=9)
    fig.text(.08, .16, 'Train: augmentation + label smoothing. Validation: neither. The two loss values are not directly equivalent.', fontsize=8)
    lines=[]
    for ax, title in [(acc,'Accuracy'), (loss,'Loss')]:
        lines.append((ax.plot([], [], color='#2463eb', label='Train (augmented)')[0],
                      ax.plot([], [], color='#f28e2b', label='Validation')[0]))
        ax.set(xlabel='Epoch', title=title, xlim=(1,max(2,history[-1]['epoch'])))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    acc.set_ylim(0,1.04)
    loss.set_ylim(0,max(max(r['train_loss'],r['val_loss']) for r in history)*1.08+.01)
    slider = Slider(fig.add_axes([.15,.09,.59,.04]), 'Epoch', 1, max(2,len(history)), valinit=1, valstep=1)
    button = Button(fig.add_axes([.8,.075,.12,.065]), 'Pause')
    state={'playing':True}
    def draw(value):
        count=min(int(value),len(history)); rows=history[:count];row=rows[-1]
        for (trainline,valline),key in zip(lines,['accuracy','loss']):
            trainline.set_data([r['epoch'] for r in rows],[r['train_'+key] for r in rows])
            valline.set_data([r['epoch'] for r in rows],[r['val_'+key] for r in rows])
        status.set_text(f"Epoch {row['epoch']} / {history[-1]['epoch']}   |   Train {row['train_accuracy']:.2%}   |   Validation {row['val_accuracy']:.2%}   |   Macro recall {row['val_macro_recall']:.2%}")
        fig.canvas.draw_idle()
    slider.on_changed(draw)
    def toggle(event):
        state['playing']=not state['playing'];button.label.set_text('Pause' if state['playing'] else 'Play')
    button.on_clicked(toggle)
    def tick():
        if state['playing']:
            slider.set_val(1 if int(slider.val)>=len(history) else int(slider.val)+1)
    timer=fig.canvas.new_timer(interval=args.interval);timer.add_callback(tick)
    draw(1)
    if args.headless:
        slider.set_val(len(history))
    else:
        timer.start()
        plt.show()
        timer.stop()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=150)
    plt.close(fig)
    print(f'Replay figure saved: {target}')


def main(argv=None):
    parser=argparse.ArgumentParser(description='假名 CNN：实时训练演示、历史回放、数据准备与评估。')
    sub=parser.add_subparsers(dest='command')
    p=sub.add_parser('replay',help='播放已有真实日志；不重新训练')
    p.add_argument('--history',default=str(BASE/'reports/history.json'))
    p.add_argument('--interval',type=int,default=450,help='每轮回放间隔，毫秒')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--output',default=str(BASE/'runs/replay.png'))
    p.set_defaults(func=replay)
    p=sub.add_parser('train',help='真正训练并实时更新曲线、固定验证样例')
    p.add_argument('--data',default=str(BASE/'data/prepared'))
    p.add_argument('--output',default=str(BASE/'runs'/time.strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--device',default='auto',choices=['auto','cuda','mps','cpu'])
    p.add_argument('--resume')
    p.add_argument('--headless',action='store_true',help='不弹窗，每轮仍保存 dashboard PNG')
    p.set_defaults(func=None)
    p=sub.add_parser('prepare',help='从原始数据构建规范划分；详情见 README')
    p.add_argument('--hiragana-train',required=True);p.add_argument('--hiragana-test',required=True)
    p.add_argument('--font-root',required=True);p.add_argument('--external');p.add_argument('--supplement-fonts')
    p.add_argument('--extra-npz',action='append',default=[]);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261029);p.set_defaults(func=prepare)
    p=sub.add_parser('evaluate',help='固定预处理评估，不训练')
    p.add_argument('--model',default=str(BASE/'models/best.pt'));p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='auto');p.set_defaults(func=evaluate)
    p=sub.add_parser('export',help='导出 TorchScript 与 torch.export 模型')
    p.add_argument('--model',default=str(BASE/'models/best.pt'));p.add_argument('--output',required=True);p.set_defaults(func=export)
    args=parser.parse_args(argv)
    if args.command is None:
        args=parser.parse_args(['replay'])
    try:
        if args.command=='train':
            if not (Path(args.data)/'manifest.json').is_file():
                raise ValueError(f'找不到训练数据：{args.data}。请先 prepare；只看演示可运行 replay。')
            dashboard=TrainingDashboard(args.output,args.headless)
            try:
                train(args,observer=dashboard)
            except BaseException:
                dashboard.plt.close(dashboard.fig)
                raise
            dashboard.finish()
        else:
            args.func(args)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':
    main()
