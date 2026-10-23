import copy
import numpy as np
import torch
import argparse
from LMFCN import LMFCN

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--relax', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--feature', type=int, default=2)
parser.add_argument('--shake', type=int, default=0)
parser.add_argument('--svclose', type=int, default=8)
parser.add_argument('--wrclose', type=int, default=3)
parser.add_argument('--shclose', type=int, default=3)
parser.add_argument('--width', type=int, default=200)
parser.add_argument('--height', type=int, default=200)
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--norm', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--outdir', type=str, default="graphs/")
parser.add_argument('--outfile', type=str, default="gaussians")
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('-i', type=str, default="")
opt = parser.parse_args()

if(opt.i == "gaussians"):
    from LoadGaussians import LoadImages
else:
    from LoadImagesGeneral import LoadImages

# Controlled tests
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

# Loading data
ts = opt.i.replace("train","test")
tr_val = opt.i.replace("train","val")
tr = opt.i

li_tr = LoadImages(imglist=tr, batch_size=opt.batch, aug=opt.aug, width=opt.width, height=opt.height, classcount=[200,200])
li_val = LoadImages(imglist=tr_val, batch_size=opt.batch, aug=1, width=opt.width, height=opt.height, x_min=li_tr.x_min, x_max=li_tr.x_max, classcount=[200,200])

li_tr.generate_batches()
li_val.generate_batches()

x_train = None
y_train = None

x_val = None
y_val = None

# Converting data from the dataloader to arrays
# This is necessary for compatibility with
# fit, predict and predict_proba methods
for i in range(li_tr.nr_batches):
    X_input, y_input = li_tr.get_batch(idx=i)

    if(x_train is None):
        x_train = X_input.cpu().numpy()
        y_train = y_input
    else:
        x_train = np.concatenate((x_train, X_input.cpu().numpy()), axis=0)
        y_train = np.concatenate((y_train, y_input), axis=0)

for i in range(li_val.nr_batches):
    X_input, y_input = li_val.get_batch(idx=i)

    if(x_val is None):
        x_val = X_input.cpu().numpy()
        y_val = y_input
    else:
        x_val = np.concatenate((x_val, X_input.cpu().numpy()), axis=0)
        y_val = np.concatenate((y_val, y_input), axis=0)

##########################################

print("Start LMFCN")
lmfcn = LMFCN(feature=opt.feature, width=opt.width, height=opt.height, epoch=opt.epoch, 
            gamma=opt.gamma, relax=opt.relax, lr=opt.lr, shake=opt.shake, shclose=opt.shclose, svclose=opt.svclose, 
            wrclose=opt.wrclose, aug=opt.aug, norm=opt.norm, seed=0, outdir="graphs/", outfile="fcnsv_output", batch=opt.batch, i=None)

print("Start fit")
best_state = None
best_clf = None
best_X = None
best_gamma_auto = None
best_epoch = -1
best_val_bal_acc = -1.0

for epoch in range(opt.epoch):
    # O LMFCN original atualiza os pesos depois da avaliacao da epoca.
    # Com epoch=2, o segundo passo apenas avalia o estado apos uma atualizacao.
    lmfcn.li_tr = None
    lmfcn.li_val = None
    lmfcn.li_ts = None
    lmfcn.epoch = 2
    history = lmfcn.fit(X=x_train, y=y_train, X_val=x_val, y_val=y_val)
    val_metrics = history[-1]
    val_bal_acc = val_metrics["val balanced acc"]

    if val_bal_acc > best_val_bal_acc:
        best_state = copy.deepcopy(lmfcn.model.state_dict())
        best_clf = copy.deepcopy(lmfcn.clf)
        best_X = lmfcn.X.detach().clone()
        best_gamma_auto = lmfcn.gamma_auto
        best_epoch = epoch
        best_val_bal_acc = val_bal_acc
        print(
            f"Melhor epoca de validacao: {best_epoch + 1} "
            f"(balanced accuracy={best_val_bal_acc:.4f})"
        )

if best_state is None or best_clf is None or best_X is None or best_gamma_auto is None:
    raise RuntimeError("Nenhum estado LMFCN foi selecionado pela validacao.")

lmfcn.model.load_state_dict(best_state)
lmfcn.clf = best_clf
lmfcn.X = best_X
lmfcn.gamma_auto = best_gamma_auto

print("Carregando holdout final após a seleção da época")
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
x_test = None
y_test = None
for i in range(li_ts.nr_batches):
    X_input, y_input = li_ts.get_batch(idx=i)
    if x_test is None:
        x_test = X_input.cpu().numpy()
        y_test = y_input
    else:
        x_test = np.concatenate((x_test, X_input.cpu().numpy()), axis=0)
        y_test = np.concatenate((y_test, y_input), axis=0)

print("Predict all")
q = lmfcn.predict(X=x_test, y=y_test)
print(len(q))

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
cm = confusion_matrix(y_test, q, labels=[0, 1])
report = classification_report(y_test, q, output_dict=True, zero_division=0)
acc = accuracy_score(y_test, q)
balanced_acc = balanced_accuracy_score(y_test, q)
f1_macro = f1_score(y_test, q, average="macro", zero_division=0)

print(f"\n--- RESUMO FINAL NO HOLDOUT (LMFCN | {opt.i}) ---")
print(f"Época selecionada pela validação: {best_epoch + 1}")
print(f"Acc (Acurácia Global):          {acc:.4f}  (Tabela: {acc:.2f})")
print(f"Rec (Balanced Accuracy):        {balanced_acc:.4f}  (Tabela: {balanced_acc:.2f})")
print(f"Recall Classe 0 (Rotten):       {report['0']['recall']:.4f}  (Tabela: {report['0']['recall']:.2f})")
print(f"Recall Classe 1 (Fresh):        {report['1']['recall']:.4f}  (Tabela: {report['1']['recall']:.2f})")
print(f"F1 Macro:                       {f1_macro:.4f}  (Tabela: {f1_macro:.2f})")
print("\nMatriz de Confusão (Holdout):")
print("                 Previsto Rotten(0) | Previsto Fresh(1)")
print(f"Real Rotten (0): {cm[0][0]:18d} | {cm[0][1]:17d}")
print(f"Real Fresh  (1): {cm[1][0]:18d} | {cm[1][1]:17d}")
print("----------------------------------\n")