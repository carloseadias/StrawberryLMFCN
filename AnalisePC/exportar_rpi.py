import os
import copy
import shutil
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

from LoadImagesGeneral import LoadImages
from LMFCN import LMFCN

# Configuracoes identicas ao ExecAnalise.bat
SETTING_TRAIN = "settings/setting1.1-train.txt"
SETTING_VAL   = "settings/setting1.1-val.txt"
SETTING_TEST  = "settings/setting1.1-test.txt"
WIDTH, HEIGHT = 275, 183
BATCH_SIZE    = 32
EPOCHS        = 5
SEED          = 0

PACOTE_DIR = "pacote_rpi"
PESOS_DIR  = os.path.join(PACOTE_DIR, "pesos_rpi")
os.makedirs(PESOS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_arrays(loader):
    x_arr, y_arr = None, None
    for i in range(loader.nr_batches):
        X_input, y_input = loader.get_batch(idx=i)
        if x_arr is None:
            x_arr = X_input.cpu().numpy()
            y_arr = y_input
        else:
            x_arr = np.concatenate((x_arr, X_input.cpu().numpy()), axis=0)
            y_arr = np.concatenate((y_arr, y_input), axis=0)
    return x_arr, y_arr

print(f"[1/4] Carregando dados de {SETTING_TRAIN} e {SETTING_VAL}...")
set_seeds(SEED)
li_tr = LoadImages(imglist=SETTING_TRAIN, batch_size=BATCH_SIZE, aug=1, width=WIDTH, height=HEIGHT, classcount=[200, 200])
li_val = LoadImages(imglist=SETTING_VAL, batch_size=BATCH_SIZE, aug=1, width=WIDTH, height=HEIGHT, x_min=li_tr.x_min, x_max=li_tr.x_max, classcount=[200, 200])
li_tr.generate_batches()
li_val.generate_batches()

x_train, y_train = extract_arrays(li_tr)
x_val, y_val = extract_arrays(li_val)

# ==============================================================================
# 1. TREINAR E EXPORTAR A1 (LMFCN)
# ==============================================================================
print("\n[2/4] Treinando e exportando A1 (LMFCN)...")
set_seeds(SEED)
lmfcn = LMFCN(feature=64, width=WIDTH, height=HEIGHT, epoch=EPOCHS,
              gamma=1.0, relax=100.0, lr=1.0, shake=0, shclose=0, svclose=5,
              wrclose=5, aug=1, norm=1, seed=0, batch=BATCH_SIZE, i=None)

best_state, best_clf, best_X, best_gamma_auto = None, None, None, None
best_val_bal_acc = -1.0

for ep in range(EPOCHS):
    lmfcn.li_tr = lmfcn.li_val = lmfcn.li_ts = None
    lmfcn.epoch = 2
    history = lmfcn.fit(X=x_train, y=y_train, X_val=x_val, y_val=y_val)
    val_bal_acc = history[-1]["val balanced acc"]
    if val_bal_acc > best_val_bal_acc:
        best_val_bal_acc = val_bal_acc
        best_state = {k: v.cpu() for k, v in copy.deepcopy(lmfcn.model.state_dict()).items()}
        best_clf = copy.deepcopy(lmfcn.clf)
        best_X = lmfcn.X.detach().cpu().clone()
        best_gamma_auto = float(lmfcn.gamma_auto.detach().cpu().item() if torch.is_tensor(lmfcn.gamma_auto) else lmfcn.gamma_auto)

torch.save({
    "model_state": best_state,
    "clf": best_clf,
    "X": best_X,
    "gamma_auto": best_gamma_auto,
    "feature": 64,
    "x_min": float(li_tr.x_min),
    "x_max": float(li_tr.x_max)
}, os.path.join(PESOS_DIR, "A1_lmfcn.pth"))
print("   [OK] Salvo em pesos_rpi/A1_lmfcn.pth")

# ==============================================================================
# 2. TREINAR E EXPORTAR B1, B2 (YOLO11) E C1, C2 (MOBILENETV2)
# ==============================================================================
print("\n[3/4] Treinando e exportando B1, B2, C1 e C2...")
train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset   = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def build_baseline(model_name, pretrained):
    if model_name == 'mobilenet':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    elif model_name == 'yolo11':
        from ultralytics import YOLO
        yolo = YOLO('yolo11n-cls.pt' if pretrained else 'yolo11n-cls.yaml')
        net = yolo.model
        for mod in net.modules():
            if hasattr(mod, 'linear') and isinstance(mod.linear, nn.Linear):
                mod.linear = nn.Linear(mod.linear.in_features, 2)
        for param in net.parameters():
            param.requires_grad = True
        return net

def forward_logits(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[1] if len(out) > 1 else out[0]
    return out

from sklearn.metrics import balanced_accuracy_score

configs_baselines = [
    ("B1_yolo11_zero.pth", "yolo11", False),
    ("B2_yolo11_pre.pth",  "yolo11", True),
    ("C1_mobilenet_zero.pth", "mobilenet", False),
    ("C2_mobilenet_pre.pth",  "mobilenet", True),
]

for nome_arq, mod_name, is_pre in configs_baselines:
    set_seeds(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    model = build_baseline(mod_name, is_pre).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_b_state = None
    best_b_val = -1.0

    for ep in range(EPOCHS):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(forward_logits(model, X_b), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        y_t, y_p = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds = torch.argmax(forward_logits(model, X_b.to(device)), dim=1).cpu().numpy()
                y_p.extend(preds)
                y_t.extend(y_b.numpy())
        bal = balanced_accuracy_score(y_t, y_p)
        if bal > best_b_val:
            best_b_val = bal
            best_b_state = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}

    torch.save({
        "model_state": best_b_state,
        "model_name": mod_name,
        "x_min": float(li_tr.x_min),
        "x_max": float(li_tr.x_max)
    }, os.path.join(PESOS_DIR, nome_arq))
    print(f"   [OK] Salvo em pesos_rpi/{nome_arq}")

# ==============================================================================
# 3. EMPACOTAR APENAS AS 200 IMAGENS DE TESTE E SCRIPTS PARA O RASPBERRY
# ==============================================================================
print("\n[4/4] Montando pasta 'pacote_rpi/' pronta para enviar via SCP...")
os.makedirs(os.path.join(PACOTE_DIR, "settings"), exist_ok=True)
os.makedirs(os.path.join(PACOTE_DIR, "dataset", "fresh"), exist_ok=True)
os.makedirs(os.path.join(PACOTE_DIR, "dataset", "rotten"), exist_ok=True)

linhas_linux = []
with open(SETTING_TEST, "r", encoding="utf-8") as f:
    for linha in f:
        linha = linha.strip()
        if not linha:
            continue
        partes = linha.split(";")
        cam_win = partes[2].replace("/", os.sep).replace("\\", os.sep)
        cam_linux = partes[2].replace("\\", "/")
        linhas_linux.append(f"{partes[0]};{partes[1]};{cam_linux}")
        destino_img = os.path.join(PACOTE_DIR, cam_win)
        os.makedirs(os.path.dirname(destino_img), exist_ok=True)
        shutil.copy2(cam_win, destino_img)

with open(os.path.join(PACOTE_DIR, "settings", "setting1.1-test.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(linhas_linux) + "\n")

for arq_py in ["TCNN.py", "LoadImagesGeneral.py", "yolo11n-cls.yaml"]:
    if os.path.isfile(arq_py):
        shutil.copy2(arq_py, os.path.join(PACOTE_DIR, arq_py))

print("   [OK] Pacote completo gerado na pasta 'pacote_rpi/'!")