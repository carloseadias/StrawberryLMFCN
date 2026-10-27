import argparse
import copy
import os
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
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, confusion_matrix, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='both', choices=['mobilenet', 'yolo11', 'both'],
                    help='Modelo: mobilenet, yolo11 ou both')
parser.add_argument('--pretrained', type=str, default='both', choices=['0', '1', 'both'],
                    help='0: Nao pre-treinado | 1: Pre-treinado | both: Roda ambos')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--width', type=int, default=275)
parser.add_argument('--height', type=int, default=183)
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--outdir', type=str, default="strawberry_out")
parser.add_argument('--outfile', type=str, default="strawberry_1")
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('-i', type=str, required=True, help='Arquivo txt de treino (ex: setting1-train.txt)')
opt = parser.parse_args()

if opt.i == "gaussians":
    from LoadGaussians import LoadImages
else:
    from LoadImagesGeneral import LoadImages

def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(opt.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# 1. CARREGAMENTO DOS DADOS (IDÊNTICO AO main.py DO LMFCN)
# =====================================================================
tr_val = opt.i.replace("train", "val")
tr = opt.i

print(f"Carregando dados de: {tr}, {tr_val} ({opt.width}x{opt.height})...")
li_tr = LoadImages(imglist=tr, batch_size=opt.batch, aug=opt.aug, width=opt.width, height=opt.height, classcount=[200, 200])
li_val = LoadImages(imglist=tr_val, batch_size=opt.batch, aug=1, width=opt.width, height=opt.height, x_min=li_tr.x_min, x_max=li_tr.x_max, classcount=[200, 200])

li_tr.generate_batches()
li_val.generate_batches()

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

x_train, y_train = extract_arrays(li_tr)
x_val, y_val = extract_arrays(li_val)

train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset   = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=opt.batch, shuffle=False)
val_loader   = DataLoader(val_dataset, batch_size=opt.batch, shuffle=False)

# =====================================================================
# 2. CONSTRUÇÃO DOS MODELOS (Pré-treinados e Não Pré-treinados)
# =====================================================================
def build_model(model_name, num_classes=2, pretrained=False):
    if model_name == 'mobilenet':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif model_name == 'yolo11':
        from ultralytics import YOLO
        yolo = YOLO('yolo11n-cls.pt' if pretrained else 'yolo11n-cls.yaml')
        net = yolo.model
        for m in net.modules():
            if hasattr(m, 'linear') and isinstance(m.linear, nn.Linear):
                m.linear = nn.Linear(m.linear.in_features, num_classes)
        for param in net.parameters():
            param.requires_grad = True
        return net

def forward_logits(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[1] if len(out) > 1 else out[0]
    return out

def evaluate(model, loader, prefix="ts"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            logits = forward_logits(model, X_b)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_b.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc_reg = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    rec_0 = class_report.get('0', {}).get('recall', 0.0)
    rec_1 = class_report.get('1', {}).get('recall', 0.0)

    print("{:12s}: {:.4f} ({:.4f}) Recall(0): {:.4f} Recall(1): {:.4f}".format(
        prefix, bal_acc, acc_reg, rec_0, rec_1
    ))
    return {
        'bal_acc': bal_acc,
        'acc': acc_reg,
        'recall_0': rec_0,
        'recall_1': rec_1,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'y_true': y_true,
        'y_pred': y_pred
    }

def load_holdout():
    ts = opt.i.replace("train", "test")
    li_ts = LoadImages(
        imglist=ts,
        batch_size=opt.batch,
        aug=1,
        width=opt.width,
        height=opt.height,
        x_min=li_tr.x_min,
        x_max=li_tr.x_max,
        classcount=[200, 200],
    )
    li_ts.generate_batches()
    x_test, y_test = extract_arrays(li_ts)
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    return DataLoader(test_dataset, batch_size=opt.batch, shuffle=False)

# =====================================================================
# 3. TREINAMENTO, AVALIAÇÃO E MATRIZ DE CONFUSÃO
# =====================================================================
def run_experiment(model_name, is_pretrained):
    set_seeds(opt.seed)
    
    g = torch.Generator()
    g.manual_seed(opt.seed)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch, shuffle=True, generator=g)
    
    mode_tag = "PRE-TREINADO" if is_pretrained else "NAO PRE-TREINADO (DO ZERO)"
    short_tag = f"{model_name.upper()}_{'PRE' if is_pretrained else 'ZERO'}"

    print("\n" + "="*68)
    print(f"Start {model_name.upper()} [{mode_tag}] | Cenário: {opt.i} | Épocas: {opt.epoch}")
    print("="*68)

    model = build_model(model_name, num_classes=2, pretrained=is_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-5)
    best_state = None
    best_epoch = -1
    best_val_bal_acc = -1.0

    for epoch in range(opt.epoch):
        model.train()
        torch.set_grad_enabled(True)
        running_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = forward_logits(model, X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        print("------------------ Epoch {} --------------------".format(epoch))
        evaluate(model, train_loader, prefix=f"{short_tag} tr")
        val_metrics = evaluate(model, val_loader, prefix=f"{short_tag} val")
        if val_metrics['bal_acc'] > best_val_bal_acc:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_bal_acc = val_metrics['bal_acc']
            print(
                f"Melhor época de validação: {best_epoch + 1} "
                f"(balanced accuracy={best_val_bal_acc:.4f})"
            )
        print(f"Loss: {running_loss / len(train_loader):.4f}")

    if best_state is None:
        raise RuntimeError("Nenhum estado de modelo foi selecionado pela validação.")
    model.load_state_dict(best_state)
    test_loader = load_holdout()
    print(f"\nModelo restaurado da melhor época: {best_epoch + 1}")
    final_ts = evaluate(model, test_loader, prefix=f"{short_tag} holdout")
    print(len(final_ts['y_pred']))

    cm = confusion_matrix(final_ts['y_true'], final_ts['y_pred'], labels=[0, 1])

    print(f"\n--- RESUMO FINAL NO HOLDOUT ({short_tag} | {opt.i}) ---")
    print(f"Época selecionada pela validação: {best_epoch + 1}")
    print(f"Acc (Acurácia Global):          {final_ts['acc']:.4f}  (Tabela: {final_ts['acc']:.2f})")
    print(f"Rec (Balanced Acc / Recall M.): {final_ts['bal_acc']:.4f}  (Tabela: {final_ts['bal_acc']:.2f})")
    print(f"Recall Classe 0 (Rotten):       {final_ts['recall_0']:.4f}  (Tabela: {final_ts['recall_0']:.2f})")
    print(f"Recall Classe 1 (Fresh):        {final_ts['recall_1']:.4f}  (Tabela: {final_ts['recall_1']:.2f})")
    print(f"F1 Macro:                       {final_ts['f1_macro']:.4f}  (Tabela: {final_ts['f1_macro']:.2f})")
    print("\nMatriz de Confusão (Holdout):")
    print("                 Previsto Rotten(0) | Previsto Fresh(1)")
    print(f"Real Rotten (0): {cm[0][0]:18d} | {cm[0][1]:17d}   (FN = {cm[0][1]})")
    print(f"Real Fresh  (1): {cm[1][0]:18d} | {cm[1][1]:17d}   (FP = {cm[1][0]})")
    print("------------------------------------------------------------\n")

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

models_to_run = ['yolo11', 'mobilenet'] if opt.model == 'both' else [opt.model]
pretrained_modes = [False, True] if opt.pretrained == 'both' else [opt.pretrained == '1']

for m in models_to_run:
    for p in pretrained_modes:
        run_experiment(m, p)