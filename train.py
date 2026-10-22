"""
Passo 2 (adaptado) - Treino com Transfer Learning (MobileNetV2).

Uso:  python train.py                        -> le data/, gera gesture_model.keras etc.
      python train.py data_pessoas pessoas   -> le data_pessoas/ e prefixa as saidas com "pessoas_"

Saidas: <prefixo>gesture_model.keras, <prefixo>class_names.json, <prefixo>curvas.png,
        <prefixo>resultados.txt (matriz de confusao + acerto por classe),
        <prefixo>dataset_imagens.pdf (arquivo da ENTREGA).
"""
import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"
PREFIX = (sys.argv[2] + "_") if len(sys.argv) > 2 else ""
IMG_SIZE = 224
BATCH = 16
EPOCHS = 10
VAL_FRACTION = 0.2
GAP_MS = 1500          # pausa > 1,5 s entre fotos = novo bloco de coleta
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
print("Classes:", class_names)


def split_por_blocos(files):
    """Separa as fotos em blocos de coleta (pelo intervalo de tempo entre elas) e usa
    os ultimos 20% de CADA bloco como validacao. Assim a validacao tem fundo escuro e
    claro, e fotos quase identicas (frames vizinhos) nao vazam entre treino e validacao."""
    try:
        blocos, atual, ultimo = [], [], None
        for f in files:
            t = int(os.path.splitext(os.path.basename(f))[0])
            if ultimo is not None and t - ultimo > GAP_MS:
                blocos.append(atual); atual = []
            atual.append(f); ultimo = t
        if atual:
            blocos.append(atual)
    except ValueError:  # nomes de arquivo sem timestamp: usa o corte simples do enunciado
        blocos = [files]
    tr, va = [], []
    for b in blocos:
        cut = int(len(b) * (1 - VAL_FRACTION))
        tr += b[:cut]; va += b[cut:]
    return tr, va


train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    files = sorted(f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.lower().endswith(".jpg"))
    files = [os.path.join(DATA_DIR, cls, f) for f in files]
    tr, va = split_por_blocos(files)
    train_paths += tr; train_y += [idx] * len(tr)
    val_paths += va; val_y += [idx] * len(va)
    print(f"  {cls}: {len(tr)} treino / {len(va)} validacao")


def load(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_paths, train_y, True)
val_ds = make_ds(val_paths, val_y, False)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.25),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save(f"{PREFIX}gesture_model.keras")
with open(f"{PREFIX}class_names.json", "w") as f:
    json.dump(class_names, f)

plt.figure(figsize=(10, 4))
for i, (m, t) in enumerate([("accuracy", "Acuracia"), ("loss", "Loss")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(hist.history[m], label="treino")
    plt.plot(hist.history["val_" + m], label="validacao")
    plt.title(t); plt.xlabel("epoca"); plt.legend()
plt.tight_layout(); plt.savefig(f"{PREFIX}curvas.png")

# Matriz de confusao + acerto por classe (salva em .txt para usar no relatorio)
y_true = np.array(val_y)
y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

linhas = []
def out(s=""):
    print(s); linhas.append(s)

out(f"Acuracia de validacao final: {hist.history['val_accuracy'][-1]:.2%}")
out(f"Acuracia de treino final:   {hist.history['accuracy'][-1]:.2%}")
out("\nMatriz de confusao (linhas = real, colunas = previsto):")
out(" " * 11 + " ".join(f"{c[:8]:>8}" for c in class_names))
for i, c in enumerate(class_names):
    out(f"{c[:10]:>10} " + " ".join(f"{v:>8}" for v in cm[i]))
out("\nAcerto por classe (recall):")
for i, c in enumerate(class_names):
    total = cm[i].sum()
    out(f"  {c}: {cm[i, i] / total:.0%} ({cm[i, i]}/{total})" if total else f"  {c}: sem fotos de validacao")
with open(f"{PREFIX}resultados.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(linhas))

print(f"\nPronto! Modelo salvo em {PREFIX}gesture_model.keras. Agora rode: python play.py {PREFIX.rstrip('_')}")

# ---- DICA DO ENUNCIADO: gera um PDF com as imagens do dataset, separadas por classe ----
from PIL import Image, ImageDraw, ImageFont


def gerar_pdf_dataset(data_dir="data", saida="dataset_imagens.pdf", cols=10, qualidade=70):
    pag_w, pag_h, margem, topo = 1240, 1754, 40, 110
    cell = (pag_w - 2 * margem) // cols
    rows = (pag_h - topo - margem) // cell
    por_pagina = cols * rows
    try:
        fonte = ImageFont.load_default(size=30)
    except TypeError:
        fonte = ImageFont.load_default()
    paginas = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for classe in classes:
        pasta = os.path.join(data_dir, classe)
        fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        n_pag = max(1, -(-len(fotos) // por_pagina))
        for p in range(n_pag):
            pagina = Image.new("RGB", (pag_w, pag_h), "white")
            draw = ImageDraw.Draw(pagina)
            draw.text((margem, 40),
                      f"Classe: {classe} | {len(fotos)} imagens | pagina {p + 1}/{n_pag}",
                      fill="black", font=fonte)
            for i, nome in enumerate(fotos[p * por_pagina:(p + 1) * por_pagina]):
                img = Image.open(os.path.join(pasta, nome)).convert("RGB")
                img.thumbnail((cell - 6, cell - 6))
                x = margem + (i % cols) * cell + 3
                y = topo + (i // cols) * cell + 3
                pagina.paste(img, (x, y))
            paginas.append(pagina)
    paginas[0].save(saida, save_all=True, append_images=paginas[1:],
                    quality=qualidade, resolution=150)
    print(f"PDF salvo: {saida} ({len(paginas)} paginas)")


gerar_pdf_dataset(DATA_DIR, saida=f"{PREFIX}dataset_imagens.pdf")
