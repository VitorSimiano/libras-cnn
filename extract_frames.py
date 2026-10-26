"""
Extrai frames de VIDEOS para montar o dataset (alternativa ao collect.py).

Organize os videos assim (nome da pasta = nome da classe):
  videos/
    oi/        1_escuro.mov  2_claro.mov  3_escuro.mov  4_claro.mov
    joia/      ...
    nada/      ...
A ordem alfabetica dos arquivos define a ordem no tempo, entao numere
alternando fundo escuro e claro (1_escuro, 2_claro, 3_escuro, 4_claro).

Uso:  python extract_frames.py                     (le videos/, grava em data/)
      python extract_frames.py videos data_pessoas (pastas de entrada e saida)
"""
import os
import sys
import cv2
import numpy as np

VIDEOS_DIR = sys.argv[1] if len(sys.argv) > 1 else "videos"
DATA_DIR = sys.argv[2] if len(sys.argv) > 2 else "data"
IMG_SIZE = 224
TARGET_PER_CLASS = 300
CROP_FRAC = 0.70      # parte central usada (fracao do menor lado do video); 1.0 = quadrado inteiro
FLIP = True           # espelha, igual ao collect.py/play.py (webcam em modo espelho)
EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp")
FRAME_MS = 100        # espacamento entre fotos do mesmo video
VIDEO_GAP_MS = 10000  # espacamento entre videos (> 1500 ms, para o train.py separar os blocos)

if not os.path.isdir(VIDEOS_DIR):
    raise SystemExit(f"Pasta '{VIDEOS_DIR}' nao encontrada.")
classes = sorted(d for d in os.listdir(VIDEOS_DIR)
                 if os.path.isdir(os.path.join(VIDEOS_DIR, d)))
if not classes:
    raise SystemExit(f"Nenhuma subpasta de classe dentro de '{VIDEOS_DIR}'.")

t = 1_700_000_000_000
for cls in classes:
    pasta_in = os.path.join(VIDEOS_DIR, cls)
    pasta_out = os.path.join(DATA_DIR, cls)
    os.makedirs(pasta_out, exist_ok=True)
    existentes = len([f for f in os.listdir(pasta_out) if f.endswith(".jpg")])
    faltam = TARGET_PER_CLASS - existentes
    videos = sorted(f for f in os.listdir(pasta_in) if f.lower().endswith(EXTS))
    if not videos:
        print(f"{cls}: nenhum video encontrado, pulando.")
        continue
    if faltam <= 0:
        print(f"{cls}: ja tem {existentes} fotos, pulando.")
        continue
    por_video = -(-faltam // len(videos))   # divisao arredondada para cima
    salvas = 0
    for nome in videos:
        cap = cv2.VideoCapture(os.path.join(pasta_in, nome))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            print(f"  nao consegui ler {cls}/{nome}")
            continue
        n = min(por_video, total)
        alvo = set(np.linspace(0, total - 1, n).astype(int).tolist())
        i = 0
        while salvas < faltam:
            ok, frame = cap.read()
            if not ok:
                break
            if i in alvo:
                h, w = frame.shape[:2]
                side = int(min(h, w) * CROP_FRAC)
                x1, y1 = w // 2 - side // 2, h // 2 - side // 2
                roi = frame[y1:y1 + side, x1:x1 + side]
                if FLIP:
                    roi = cv2.flip(roi, 1)
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(pasta_out, f"{t}.jpg"), img)
                t += FRAME_MS
                salvas += 1
            i += 1
        cap.release()
        t += VIDEO_GAP_MS
    print(f"{cls}: {salvas} fotos novas de {len(videos)} video(s)")

print("\nPronto! Confira algumas fotos em cada pasta de", DATA_DIR,
      "(a mao precisa estar inteira e centralizada). Depois rode: python train.py")