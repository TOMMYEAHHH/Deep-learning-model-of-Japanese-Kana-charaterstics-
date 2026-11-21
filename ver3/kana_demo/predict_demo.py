"""PyCharm 中右键 -> Run 即可：加载训练好的模型，对真实手写样本做预测。

要点：
  * 工作目录必须是 E:\\Python程序（kana_recognizer 才能被 import）
  * 解释器选 .venv\\Scripts\\python.exe
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # E:\Python程序
if str(ROOT) not in sys.path:                          # 终端直接跑也能 import
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from kana_recognizer import KanaRecognizer

MODEL = ROOT / 'kana_recognizer' / 'models' / 'best.pt'
SAMPLES = Path(__file__).resolve().parent / 'samples'
N_SAMPLES = 8


def ensure_samples(n=N_SAMPLES, seed=7):
    """从外部手写集中抽几张存成 PNG，方便在 PyCharm 里点开看原图。"""
    existing = sorted(SAMPLES.glob('*.png'))
    if len(existing) >= n:
        return existing

    SAMPLES.mkdir(parents=True, exist_ok=True)
    classes = json.load(open(ROOT / 'kana_recognizer' / 'models' / 'metadata.json',
                             encoding='utf-8'))['classes']
    data = np.load(ROOT / 'kana_recognizer' / 'data' / 'prepared' / 'external.npz',
                   allow_pickle=False)
    idx = np.random.default_rng(seed).choice(len(data['labels']), n, replace=False)
    paths = []
    for i in idx:
        char = classes[data['labels'][i]]
        p = SAMPLES / f'{len(paths):02d}_{char}.png'
        Image.fromarray(data['images'][i]).save(p)
        paths.append(p)
    return paths


def main():
    print(f'模型: {MODEL}')
    recognizer = KanaRecognizer(str(MODEL), device='auto')
    print(f'设备: {recognizer.device}   温度: {recognizer.temperature}')
    print()

    # 单张：reports/example.png
    example = ROOT / 'kana_recognizer' / 'reports' / 'example.png'
    if example.exists():
        r = recognizer.predict(str(example), top_k=3)
        print(f'[单张] {example.name} -> {r["prediction"]}  '
              f'置信度 {r["candidates"][0]["score"]:.4f}  '
              f'需复核={r["needs_review"]}')
        print()

    print(f'{"文件":<14}{"真实":<6}{"预测":<6}{"置信度":>9}  {"前三候选"}')
    print('-' * 62)
    ok = 0
    for p in ensure_samples():
        truth = p.stem.split('_')[1]
        r = recognizer.predict(str(p), top_k=3)
        top3 = '  '.join(f'{c["character"]}:{c["score"]:.2f}' for c in r['candidates'])
        hit = r['prediction'] == truth
        ok += hit
        print(f'{p.name:<14}{truth:<6}{r["prediction"]:<6}'
              f'{r["candidates"][0]["score"]:>9.4f}  {top3}   {"OK" if hit else "MISS"}')
    print('-' * 62)
    print(f'命中 {ok}/{N_SAMPLES}')


if __name__ == '__main__':
    main()
