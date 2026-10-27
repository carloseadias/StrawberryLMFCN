import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import balanced_accuracy_score, recall_score

from TCNN import TCNN
from LoadImagesGeneral import LoadImages

SETTING_TEST = "settings/setting1.1-test.txt"
PESOS_DIR    = "pesos_rpi"
WIDTH, HEIGHT = 275, 183
device = torch.device("cpu")

# 1. Carregar dados de referencia (x_min e x_max do treino)
info_a1 = torch.load(os.path.join(PESOS_DIR, "A1_lmfcn.pth"), map_location=device, weights_only=False)
x_min_ref = info_a1["x_min"]
x_max_ref = info_a1["x_max"]

print(f"Carregando as 200 imagens de '{SETTING_TEST}' na memoria RAM...")
li_ts = LoadImages(imglist=SETTING_TEST, batch_size=1, aug=1, width=WIDTH, height=HEIGHT,
                   x_min=x_min_ref, x_max=x_max_ref, classcount=[100, 100])
li_ts.generate_batches()

imagens_ram = []
rotulos_ram = []
for i in range(li_ts.nr_batches):
    X_in, y_in = li_ts.get_batch(idx=i)
    imagens_ram.append(X_in.to(device)) # Tensor [1, 3, 183, 275]
    rotulos_ram.append(int(y_in[0]))

rotulos_ram = np.array(rotulos_ram)
print(f"[OK] {len(imagens_ram)} tensores carregados na memoria.\n")

# ==============================================================================
# FUNCOES DE AVALIACAO DE LATENCIA (IMAGEM A IMAGEM)
# ==============================================================================
def medir_lmfcn(caminho_peso):
    ckpt = torch.load(caminho_peso, map_location=device, weights_only=False)
    model = TCNN(output_size=ckpt["feature"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    clf = ckpt["clf"]
    X_ref = ckpt["X"].to(device)
    X_ref_pow2 = torch.pow(X_ref, 2).sum(1)
    gamma_auto = ckpt["gamma_auto"]

    # Warm-up (5 imagens para aquecer cache da CPU ARM)
    with torch.no_grad():
        for img in imagens_ram[:5]:
            feat = model(img)[0]
            dist = torch.dot(feat, feat) + X_ref_pow2 - (feat * X_ref).sum(1) * 2
            k = torch.exp(-gamma_auto * dist).unsqueeze(0).numpy()
            _ = clf.predict(k)

    tempos_ms = []
    preds = []
    with torch.no_grad():
        for img in imagens_ram:
            t0 = time.perf_counter()
            feat = model(img)[0]
            dist = torch.dot(feat, feat) + X_ref_pow2 - (feat * X_ref).sum(1) * 2
            k = torch.exp(-gamma_auto * dist).unsqueeze(0).numpy()
            pred = clf.predict(k)[0]
            t1 = time.perf_counter()
            tempos_ms.append((t1 - t0) * 1000.0)
            preds.append(pred)

    del model, clf, X_ref
    gc.collect()
    return np.array(tempos_ms), np.array(preds)


def construir_baseline_cpu(model_name, state_dict):
    if model_name == "mobilenet":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        m.load_state_dict(state_dict)
        return m.to(device)
    elif model_name == "yolo11":
        from ultralytics import YOLO
        arq_base = "yolo11n-cls.yaml" if os.path.isfile("yolo11n-cls.yaml") else "yolo11n-cls.pt"
        yolo = YOLO(arq_base)
        net = yolo.model
        for mod in net.modules():
            if hasattr(mod, "linear") and isinstance(mod.linear, nn.Linear):
                mod.linear = nn.Linear(mod.linear.in_features, 2)
        net.load_state_dict(state_dict)
        return net.to(device)


def forward_logits(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[1] if len(out) > 1 else out[0]
    return out


def medir_baseline(caminho_peso):
    ckpt = torch.load(caminho_peso, map_location=device, weights_only=False)
    model = construir_baseline_cpu(ckpt["model_name"], ckpt["model_state"])
    model.eval()

    # Warm-up
    with torch.no_grad():
        for img in imagens_ram[:5]:
            _ = forward_logits(model, img)

    tempos_ms = []
    preds = []
    with torch.no_grad():
        for img in imagens_ram:
            t0 = time.perf_counter()
            logits = forward_logits(model, img)
            pred = int(torch.argmax(logits, dim=1).item())
            t1 = time.perf_counter()
            tempos_ms.append((t1 - t0) * 1000.0)
            preds.append(pred)

    del model
    gc.collect()
    return np.array(tempos_ms), np.array(preds)


# ==============================================================================
# EXECUCAO DOS 5 SUBMETODOS E RELATORIO FINAL
# ==============================================================================
submetodos = [
    ("A1 (LMFCN - Scratch)",       "A1_lmfcn.pth",          "lmfcn"),
    ("B1 (YOLO11n - Scratch)",     "B1_yolo11_zero.pth",    "baseline"),
    ("B2 (YOLO11n - Pre-trained)", "B2_yolo11_pre.pth",     "baseline"),
    ("C1 (MobileNetV2 - Scratch)", "C1_mobilenet_zero.pth", "baseline"),
    ("C2 (MobileNetV2 - Pre)",     "C2_mobilenet_pre.pth",  "baseline"),
]

print("=" * 88)
print("       RESULTADOS DE LATENCIA NA BORDA (RASPBERRY PI 4 - 200 IMAGENS DE TESTE)       ")
print("=" * 88)
print(f"{'Submetodo':<28} | {'Latencia Media (ms/img)':<25} | {'Taxa (FPS)':<12} | {'BalAcc':<8} | {'Rec0':<8}")
print("-" * 88)

linhas_txt = []

for nome, arq, tipo in submetodos:
    caminho = os.path.join(PESOS_DIR, arq)
    if tipo == "lmfcn":
        tempos, preds = medir_lmfcn(caminho)
    else:
        tempos, preds = medir_baseline(caminho)

    media_ms = np.mean(tempos)
    dp_ms    = np.std(tempos, ddof=0)
    fps      = 1000.0 / media_ms if media_ms > 0 else 0.0
    bal_acc  = balanced_accuracy_score(rotulos_ram, preds)
    rec0     = recall_score(rotulos_ram, preds, pos_label=0, zero_division=0)

    linha_fmt = f"{nome:<28} | {media_ms:8.2f} (+-{dp_ms:6.2f}) ms   | {fps:8.2f} FPS | {bal_acc:6.4f}   | {rec0:6.4f}"
    print(linha_fmt)
    linhas_txt.append(linha_fmt)

print("=" * 88)

with open("relatorio_latencia_rpi.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(linhas_txt) + "\n")
print("\n[OK] Resultado salvo em 'relatorio_latencia_rpi.txt'")