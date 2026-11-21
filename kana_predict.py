"""Portable single-kana inference API and CLI. Importing does not train a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 1
SIZE = 48
HIRAGANA = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
KATAKANA = ''.join((chr(ord(c) + 96) for c in HIRAGANA))
CLASSES = list(HIRAGANA + KATAKANA)
ROMAJI = 'a i u e o ka ki ku ke ko sa shi su se so ta chi tsu te to na ni nu ne no ha hi fu he ho ma mi mu me mo ya yu yo ra ri ru re ro wa wo n'.split()
PREPROCESS = {'version': VERSION, 'size': SIZE, 'foreground': 'white', 'background': 'black', 'content_size': 40, 'polarity': 'border_median', 'normalization': 'uint8 / 255', 'crop_threshold': 0.12}

def preprocess(image: Image.Image) -> np.ndarray:
    """One isolated character; auto polarity, foreground crop, aspect preserving resize."""
    image = ImageOps.exif_transpose(image)
    if 'A' in image.getbands():
        bg = Image.new('RGBA', image.size, 'white')
        bg.alpha_composite(image.convert('RGBA'))
        image = bg
    a = np.asarray(image.convert('L'), dtype=np.float32)
    if min(a.shape) < 2:
        raise ValueError('Image is too small')
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if np.median(border) > 127:
        a = 255 - a
    lo = float(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))
    hi = float(np.percentile(a, 99.5))
    if hi - lo < 10:
        raise ValueError('Blank or insufficient-contrast image')
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    yy, xx = np.where(a > PREPROCESS['crop_threshold'])
    if len(xx) < 4:
        raise ValueError('No visible character')
    a = a[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    h, w = a.shape
    scale = 40 / max(h, w)
    patch = Image.fromarray(np.uint8(a * 255)).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new('L', (SIZE, SIZE))
    canvas.paste(patch, ((SIZE - patch.width) // 2, (SIZE - patch.height) // 2))
    return np.array(canvas)

class KanaCNN(nn.Module):

    def __init__(self):
        super().__init__()
        blocks = []
        inch = 1
        for ch in [32, 64, 128]:
            blocks.extend([nn.Conv2d(inch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(), nn.MaxPool2d(2)])
            inch = ch
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.35), nn.Linear(256, len(CLASSES)))

    def forward(self, x):
        return self.classifier(self.features(x))

def device_for(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA unavailable')
    if device.type == 'mps' and (not torch.backends.mps.is_available()):
        raise ValueError('MPS unavailable')
    return device

class KanaRecognizer:
    """Reusable inference API. Scores are validation-temperature-scaled softmax."""

    def __init__(self, model_path, device='auto'):
        self.device = device_for(device)
        ck = torch.load(model_path, map_location='cpu', weights_only=True)
        if ck.get('version') != VERSION or ck['classes'] != CLASSES or ck['preprocessing'] != PREPROCESS:
            raise ValueError('Unsupported model bundle')
        self.model = KanaCNN().to(self.device)
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        self.temperature = float(ck['temperature'])

    @torch.inference_mode()
    def predict(self, image, top_k=5, threshold=0.8):
        if not 1 <= top_k <= 92 or not 0 <= threshold <= 1:
            raise ValueError('Invalid top_k or threshold')
        if not isinstance(image, Image.Image):
            with Image.open(image) as im:
                a = preprocess(im)
        else:
            a = preprocess(image)
        x = torch.from_numpy(a[None, None].astype('float32') / 255).to(self.device)
        p = (self.model(x) / self.temperature).softmax(1)[0]
        scores, ids = p.topk(top_k)
        candidates = [{'character': CLASSES[i], 'script': 'hiragana' if i < 46 else 'katakana', 'score': float(s)} for s, i in zip(scores.cpu().tolist(), ids.cpu().tolist())]
        return {'prediction': candidates[0]['character'], 'needs_review': candidates[0]['score'] < threshold, 'candidates': candidates, 'note': 'Threshold is a review heuristic, not an out-of-distribution guarantee. Isolated へ and ヘ may be visually ambiguous.'}


BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / 'models' / 'best.pt'


def open_gui(model_path, device='auto'):
    """A local file picker and top-five predictions; model loads once."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    recognizer = KanaRecognizer(model_path, device)
    root = tk.Tk()
    root.title('假名单字识别 · 92 类基础版')
    root.geometry('640x640')
    root.minsize(560, 540)
    panel = ttk.Frame(root, padding=20)
    panel.pack(fill='both', expand=True)
    ttk.Label(panel, text='假名单字识别', font=('', 22, 'bold')).pack(anchor='w')
    ttk.Label(panel, text='请选择一张仅包含单个假名的图片。').pack(anchor='w', pady=(6, 12))
    preview = ttk.Label(panel, anchor='center')
    preview.pack(fill='x', pady=10)
    result_text = tk.StringVar(value='等待选择图片')
    ttk.Label(panel, textvariable=result_text, font=('', 15)).pack(anchor='w', pady=12)
    table = ttk.Treeview(panel, columns=('character', 'script', 'score'), show='headings', height=5)
    for column, title in [('character', '候选假名'), ('script', '类型'), ('score', '模型分数')]:
        table.heading(column, text=title)
        table.column(column, width=150, anchor='center')
    table.pack(fill='x')

    def choose():
        filename = filedialog.askopenfilename(title='选择单字图片', filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp'), ('All files', '*.*')])
        if not filename:
            return
        try:
            with Image.open(filename) as im:
                normalized = preprocess(im)
                result = recognizer.predict(im)
            shown = Image.fromarray(normalized).resize((200, 200), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(shown)
            preview.configure(image=photo)
            preview.image = photo
            suffix = ' · 建议人工复核' if result['needs_review'] else ''
            result_text.set('预测：' + result['prediction'] + suffix)
            for item in table.get_children():
                table.delete(item)
            for row in result['candidates']:
                table.insert('', 'end', values=(row['character'], '平假名' if row['script']=='hiragana' else '片假名', f"{row['score']:.2%}"))
        except (OSError, ValueError, RuntimeError) as exc:
            result_text.set('无法识别该图片')
            messagebox.showerror('识别失败', str(exc))

    ttk.Button(panel, text='选择图片并识别', command=choose).pack(fill='x', pady=16)
    ttk.Label(panel, text='仅支持 92 个基本假名；不含浊音、小假名或整句 OCR。\n片假名主要来自字体训练，真实手写效果尚未充分验证。\n模型分数不是正确率保证；へ / ヘ 可能存在字形歧义。', wraplength=570).pack(anchor='w')
    root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description='92 类假名单字识别：命令行、Python API 或本地图形界面。')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='训练权重路径，默认 models/best.pt')
    parser.add_argument('--image', type=Path, help='单字符图片路径')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--output', type=Path, help='可选：保存 JSON 结果')
    parser.add_argument('--gui', action='store_true', help='打开图片选择与预测窗口')
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f'找不到权重：{args.model}。请放入 best.pt 或使用 --model 指定。')
    try:
        if args.gui or args.image is None:
            open_gui(args.model, args.device)
            return
        result = KanaRecognizer(args.model, args.device).predict(args.image, args.top_k, args.threshold)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding='utf-8')
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
