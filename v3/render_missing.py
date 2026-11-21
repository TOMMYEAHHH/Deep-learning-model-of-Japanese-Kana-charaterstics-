"""Render the 12 kana classes absent from the supplied flat font corpus."""
from __future__ import annotations
import argparse, json, unicodedata
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

MISSING={'hiragana':'あいうえおん','katakana':'アイウエオン'}

def normalized_name(path):return unicodedata.normalize('NFC',path.name)

def choose_font(predicate):
    files=[]
    for base in (Path('/System/Library/Fonts'),Path('/Library/Fonts')):
        if base.exists():files.extend(p for p in base.rglob('*') if p.suffix.lower() in {'.ttf','.ttc','.otf'})
    hits=sorted(p for p in files if predicate(normalized_name(p)))
    if not hits:raise FileNotFoundError('Could not find a required Japanese system font')
    return hits[0]

def font_specs():
    # Training, validation, and test use distinct font families.
    return [
        ('STHeiti','train',choose_font(lambda n:n=='STHeiti Medium.ttc')),
        ('HiraginoSansGB','train',choose_font(lambda n:n=='Hiragino Sans GB.ttc')),
        ('AppleGothic','train',choose_font(lambda n:n=='AppleGothic.ttf')),
        ('HiraginoKaku','train',choose_font(lambda n:'角ゴシック W3.ttc' in n)),
        ('HiraginoMincho','val',choose_font(lambda n:'明朝 ProN.ttc' in n)),
        ('HiraginoMaru','test',choose_font(lambda n:'丸ゴ ProN W4.ttc' in n)),
        ('AppleSDGothicNeo','test',choose_font(lambda n:n=='AppleSDGothicNeo.ttc')),
    ]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/supplement_v3_complete')
    parser.add_argument('--variants',type=int,default=12)
    args=parser.parse_args()
    if args.variants<1:parser.error('--variants must be positive')
    root=Path(args.output);specs=font_specs();written=0
    for family,split,path in specs:
        # Reject missing-glyph boxes: a supported face must render distinct masks for every class.
        check_font=ImageFont.truetype(str(path),88)
        chars=''.join(MISSING.values())
        signatures={(check_font.getmask(char).size,bytes(check_font.getmask(char))) for char in chars}
        if len(signatures)!=len(chars):raise ValueError(f'{path} does not render all 12 missing kana as distinct glyphs')
        for script,chars in MISSING.items():
            for char in chars:
                for i in range(args.variants):
                    size=(68,72,76,80,84,88,92,96,82,86,90,94)[i%12]
                    font=ImageFont.truetype(str(path),size)
                    canvas=Image.new('L',(128,128),255);draw=ImageDraw.Draw(canvas)
                    dx=((i%5)-2)*1.1;dy=(((i*3)%5)-2)*1.1
                    stroke=1 if i%4==0 else 0
                    draw.text((64+dx,64+dy),char,font=font,fill=0,anchor='mm',stroke_width=stroke,stroke_fill=0)
                    angle=(-2,-1,0,1,2)[i%5]
                    canvas=canvas.rotate(angle,resample=Image.Resampling.BICUBIC,fillcolor=255)
                    out=root/split/family/script/char/f'{i:03d}.png';out.parent.mkdir(parents=True,exist_ok=True)
                    canvas.save(out);written+=1
    metadata={'classes_added':{k:list(v) for k,v in MISSING.items()},'fonts':[{'family':f,'split':s,'path':str(p)} for f,s,p in specs],
              'variants_per_class_per_font':args.variants,'samples_written':written,
              'note':'Synthetic local system-font renders; not handwriting and font files are not redistributed.'}
    (root/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'samples_written':written,'fonts':len(specs),'added_classes':12,'splits':{s:sum(1 for _ in (root/s).rglob('*.png')) for s in ('train','val','test')}}))
if __name__=='__main__':main()
