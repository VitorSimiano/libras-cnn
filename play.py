"""
Passo 3 (adaptado) - Reconhecimento de sinais de Libras em tempo real.

Uso:  python play.py            -> usa gesture_model.keras
      python play.py pessoas    -> usa pessoas_gesture_model.keras (modelo do experimento)

Teclas:  s = salva uma captura da tela (png)   q = sair
"""
import json
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
CONF_MIN = 0.60       # abaixo disso o programa diz "incerto"
PREFIX = (sys.argv[1] + "_") if len(sys.argv) > 1 else ""

model = tf.keras.models.load_model(f"{PREFIX}gesture_model.keras")
class_names = json.load(open(f"{PREFIX}class_names.json"))


def predict(roi):
    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(img[None].astype("float32"), verbose=0)[0]
    return class_names[int(np.argmax(probs))], float(np.max(probs)), probs


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam.")

recent = deque(maxlen=7)   # voto da maioria nos ultimos 7 frames (suaviza)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    label, conf, probs = predict(roi)
    recent.append(label)
    stable = Counter(recent).most_common(1)[0][0]

    if stable == "nada":
        texto, cor = "Nenhum sinal", (200, 200, 200)
    elif conf < CONF_MIN:
        texto, cor = "Incerto...", (0, 165, 255)
    else:
        texto, cor = stable.replace("_", " ").upper(), (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), cor, 3)
    cv2.putText(frame, f"Modelo ve: {label} ({conf:.0%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    for i, c in enumerate(class_names):
        bw = int(probs[i] * 150)
        cv2.rectangle(frame, (10, 50 + i * 22), (10 + bw, 64 + i * 22), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, 63 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, texto, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, cor, 3)
    cv2.imshow("Libras com CNN", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        nome = f"captura_{int(time.time())}.png"
        cv2.imwrite(nome, frame)
        print("Salvo:", nome)

cap.release()
cv2.destroyAllWindows()
