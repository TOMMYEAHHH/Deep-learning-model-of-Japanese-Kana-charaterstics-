"""PyCharm 中右键 -> Run 即可：完整训练一次，并演示中断续训。

直接调用 Python API（不走命令行），好处是可以在任意一行下断点看数据、看梯度、看指标。
已存在 last.pt 时自动以续训方式接上，所以中断后再点一次 Run 不会报错。
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # E:\Python程序
if str(ROOT) not in sys.path:                          # 终端直接跑也能 import
    sys.path.insert(0, str(ROOT))

from kana_recognizer.core import train

DATA = ROOT / 'kana_recognizer' / 'data' / 'prepared'
OUTPUT = ROOT / 'kana_runs' / 'demo'
EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3
PATIENCE = 12
DEVICE = 'auto'          # auto = CUDA -> MPS -> CPU


def main():
    last = OUTPUT / 'last.pt'
    resume = str(last) if last.exists() else None
    if resume:
        print(f'发现既有训练，自动续训: {last}')

    args = Namespace(data=str(DATA), output=str(OUTPUT), epochs=EPOCHS,
                     batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
                     seed=20261029, device=DEVICE, resume=resume)
    train(args)
    print()

    history = json.load(open(OUTPUT / 'history.json', encoding='utf-8'))
    meta = json.load(open(OUTPUT / 'metadata.json', encoding='utf-8'))
    print(f'共 {len(history)} 轮；最佳轮次 = {meta["best_epoch"]}；温度 = {meta["temperature"]}')
    best = max(history, key=lambda r: r['val_macro_recall'])
    print(f'最佳验证 macro-recall = {best["val_macro_recall"]:.4f}（第 {best["epoch"]} 轮）')
    print(f'产物: {OUTPUT}\\best.pt, last.pt, history.json, validation.json, metadata.json')


if __name__ == '__main__':
    main()
