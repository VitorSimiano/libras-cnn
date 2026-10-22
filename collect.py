"""
Passo 1 (adaptado) - Coleta de dados pela webcam: 10 sinais de Libras + nada.

Uso:  python collect.py                 (salva em data/)
      python collect.py data_pessoas    (salva em outra pasta)

IMPORTANTE: clique uma vez na JANELA DO VIDEO para ela receber as teclas.

Teclas (aperte UMA vez, nao segure):
  1 a 9 e 0 = os 10 sinais (veja a lista SINAIS abaixo)
  n         = nada (sem mao)
  q         = sair

Ao apertar, ha contagem de 3s e a gravacao comeca sozinha. A cada
AUTO_PAUSE_EVERY fotos o script PAUSA sozinho: troque o fundo (escuro/claro)
e aperte a tecla de novo para gravar o proximo bloco.
"""
import os
import sys
import time
import cv2

# ---- EDITE AQUI: nomes dos sinais (sem acento e sem espaco, viram nomes de pasta) ----
SINAIS = ["oi", "joia", "eu_te_amo", "telefone", "ok",
          "nao", "sim", "casa", "paz", "obrigado"]
CLASSES = SINAIS + ["nada"]
KEY_CHARS = "1234567890"          # tecla i+1 (e 0 para o 10o sinal)

KEYS = {ord(KEY_CHARS[i]): SINAIS[i] for i in range(len(SINAIS))}
KEYS[ord("n")] = "nada"

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"
IMG_SIZE = 224
SAVE_EVERY_N_FRAMES = 3
MAX_PER_CLASS = 300               # 4 blocos de 75 (escuro, claro, escuro, claro)
AUTO_PAUSE_EVERY = 75             # pausa automatica a cada bloco
COUNTDOWN = 3.0
DEBOUNCE = 0.5

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)


def count(cls):
    return len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.endswith(".jpg")])


def label_key(i):
    return KEY_CHARS[i] if i < len(SINAIS) else "n"


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam. Feche outros apps que usam a camera "
                     "(Zoom, Meet...) ou tente cv2.VideoCapture(1).")

recording = None
countdown_start = None
last_key_time = 0
frame_count = 0
burst_count = 0
notice = ""
WIN = "Coleta"
print(f"Salvando em '{DATA_DIR}'. Clique na janela e aperte 1-9/0 (sinais) ou n (nada). 'q' sai.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler a webcam.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    status, color = "PARADO - aperte 1-9/0/n na janela", (0, 255, 0)
    if recording and countdown_start is not None:
        left = COUNTDOWN - (time.time() - countdown_start)
        if left > 0:
            status, color = f"{recording}: prepare a mao... {int(left) + 1}", (0, 255, 255)
        else:
            countdown_start = None
    if recording and countdown_start is None:
        status, color = f"GRAVANDO: {recording} ({burst_count}/{AUTO_PAUSE_EVERY})", (0, 0, 255)
        frame_count += 1
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            if count(recording) < MAX_PER_CLASS:
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(DATA_DIR, recording, f"{int(time.time() * 1000)}.jpg"), img)
                burst_count += 1
                if burst_count >= AUTO_PAUSE_EVERY:
                    notice = f"Bloco de {recording} pronto! TROQUE O FUNDO e aperte a tecla de novo"
                    print(notice)
                    recording = None
            else:
                notice = f"{recording}: completo ({MAX_PER_CLASS} fotos)."
                print(notice)
                recording = None

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), color, 3)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for i, c in enumerate(CLASSES):
        cv2.putText(frame, f"[{label_key(i)}] {c}: {count(c)}/{MAX_PER_CLASS}", (10, 60 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if notice:
        cv2.putText(frame, notice, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 255:
        continue
    if key == ord("q"):
        break
    if key in KEYS and time.time() - last_key_time > DEBOUNCE:
        last_key_time = time.time()
        cls = KEYS[key]
        if recording == cls:
            recording = None
            print(f"Pausado: {cls}")
        else:
            recording, countdown_start = cls, time.time()
            burst_count = 0
            notice = ""
            print(f"Preparando: {cls}")
    elif key not in KEYS:
        print(f"(tecla nao reconhecida: codigo {key})")

cap.release()
cv2.destroyAllWindows()
print({c: count(c) for c in CLASSES})
