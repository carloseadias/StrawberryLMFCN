import sys
import os
import re
import numpy as np
from collections import defaultdict

if len(sys.argv) < 2:
    print("[ERRO] Informe o arquivo de relatorio bruto.")
    sys.exit(1)

arq_log_bruto = sys.argv[1]
if len(sys.argv) >= 3:
    arq_consolidado = sys.argv[2]
else:
    base_dir, nome_arq = os.path.split(arq_log_bruto)
    if nome_arq.startswith("relatorio_"):
        novo_nome = nome_arq.replace("relatorio_", "relatorio_consolidado_", 1)
    else:
        novo_nome = "consolidado_" + nome_arq
    arq_consolidado = os.path.join(base_dir, novo_nome)

# Leitura binaria com deteccao automatica de UTF-16 (padrao do Tee-Object) ou UTF-8
with open(arq_log_bruto, "rb") as f:
    raw = f.read()

if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
    conteudo = raw.decode("utf-16", errors="replace")
else:
    conteudo = raw.decode("utf-8-sig", errors="replace").replace("\x00", "")

# Expressao regular imune a variacoes de acentuacao
padrao_bloco = re.compile(
    r"--- RESUMO FINAL NO HOLDOUT \((?P<modelo>[^|]+)\|\s*settings[\\/]setting(?P<cenario>\d+)\.(?P<subcenario>\d+)-train\.txt\) ---\s*"
    r"[^\n:]+:\s*(?P<epoca>\d+)\s*"
    r"Acc [^\n:]+:\s*(?P<acc>[\d.]+).*?\n"
    r"Rec [^\n:]+:\s*(?P<rec>[\d.]+).*?\n"
    r"Recall Classe 0 [^\n:]+:\s*(?P<rec0>[\d.]+).*?\n"
    r"Recall Classe 1 [^\n:]+:\s*(?P<rec1>[\d.]+).*?\n"
    r"F1 Macro:\s*(?P<f1>[\d.]+).*?\n"
    r".*?Real Rotten \(0\):\s*(?P<tn>\d+)\s*\|\s*(?P<fn>\d+).*?\n"
    r"Real Fresh\s*\(1\):\s*(?P<fp>\d+)\s*\|\s*(?P<tp>\d+).*?"
    r"TEMPO TOTAL [^\n:]+ SETTING (?P=cenario)\.(?P=subcenario)\s*:\s*(?P<tempo>[\d.,]+)\s*segundos",
    re.DOTALL
)

dados = defaultdict(list)

for m in padrao_bloco.finditer(conteudo):
    mod = m.group("modelo").strip()
    cen = int(m.group("cenario"))
    sub = int(m.group("subcenario"))
    tempo_val = float(m.group("tempo").replace(",", "."))

    dados[(mod, cen)].append({
        "sub": sub,
        "epoca": int(m.group("epoca")),
        "acc": float(m.group("acc")),
        "rec": float(m.group("rec")),
        "rec0": float(m.group("rec0")),
        "rec1": float(m.group("rec1")),
        "f1": float(m.group("f1")),
        "tn": int(m.group("tn")),
        "fn": int(m.group("fn")),
        "fp": int(m.group("fp")),
        "tp": int(m.group("tp")),
        "tempo": tempo_val,
    })

linhas_saida = []

def registrar(texto=""):
    print(texto)
    linhas_saida.append(texto)

registrar("====================================================================================")
registrar("             RELATORIO CONSOLIDADO: MEDIAS E DESVIOS PADRAO (1 A N)                 ")
registrar(f"             Arquivo de origem: {arq_log_bruto}")
registrar("====================================================================================")

if not dados:
    registrar("[AVISO] Nenhum bloco de resultado completo foi localizado para consolidacao.")
else:
    ordem_modelos = ["LMFCN", "YOLO11_ZERO", "YOLO11_PRE", "MOBILENET_ZERO", "MOBILENET_PRE"]
    modelos_presentes = sorted(
        {k[0] for k in dados.keys()},
        key=lambda x: ordem_modelos.index(x) if x in ordem_modelos else 99
    )
    cenarios_presentes = sorted({k[1] for k in dados.keys()})

    for mod in modelos_presentes:
        registrar(f"\n====================================================================================")
        registrar(f" MODELO: {mod}")
        registrar(f"====================================================================================")
        registrar(f"{'Cenario':<10} | {'N':<3} | {'Acc (Media+-DP)':<17} | {'Rec/BalAcc':<17} | {'Rec0 (Rotten)':<17} | {'Rec1 (Fresh)':<17} | {'F1 Macro':<17} | {'Tempo (s)':<15}")
        registrar("-" * 126)

        for cen in cenarios_presentes:
            lista = dados.get((mod, cen), [])
            if not lista:
                continue
            n = len(lista)

            def med_dp(chave):
                vals = [item[chave] for item in lista]
                return np.mean(vals), np.std(vals, ddof=0)

            acc_m, acc_s = med_dp("acc")
            rec_m, rec_s = med_dp("rec")
            r0_m, r0_s   = med_dp("rec0")
            r1_m, r1_s   = med_dp("rec1")
            f1_m, f1_s   = med_dp("f1")
            tp_m, tp_s   = med_dp("tempo")

            registrar(
                f"Setting {cen:<2d} | {n:<3d} | "
                f"{acc_m:.4f} (+-{acc_s:.4f}) | "
                f"{rec_m:.4f} (+-{rec_s:.4f}) | "
                f"{r0_m:.4f} (+-{r0_s:.4f}) | "
                f"{r1_m:.4f} (+-{r1_s:.4f}) | "
                f"{f1_m:.4f} (+-{f1_s:.4f}) | "
                f"{tp_m:6.2f}s (+-{tp_s:5.2f}s)"
            )

        registrar("\n   --- MATRIZES DE CONFUSAO (MEDIA vs PIOR CASO) ---")
        for cen in cenarios_presentes:
            lista = dados.get((mod, cen), [])
            if not lista:
                continue

            tn_m = np.mean([x["tn"] for x in lista])
            fn_m = np.mean([x["fn"] for x in lista])
            fp_m = np.mean([x["fp"] for x in lista])
            tp_m = np.mean([x["tp"] for x in lista])

            pior = min(lista, key=lambda x: (x["rec"], -x["fn"], x["f1"]))

            registrar(f"\n   [Setting {cen} | {mod}]")
            registrar(f"   MATRIZ MEDIA ({len(lista)} rodadas):                     PIOR MATRIZ (Ocorreu no setting{cen}.{pior['sub']} | BalAcc={pior['rec']:.4f}):")
            registrar(f"                    Prev Rotten(0) | Prev Fresh(1)                     Prev Rotten(0) | Prev Fresh(1)")
            registrar(f"   Real Rotten (0): {tn_m:14.2f} | {fn_m:13.2f}    Real Rotten (0): {pior['tn']:14d} | {pior['fn']:13d}  (FN={pior['fn']})")
            registrar(f"   Real Fresh  (1): {fp_m:14.2f} | {tp_m:13.2f}    Real Fresh  (1): {pior['fp']:14d} | {pior['tp']:13d}  (FP={pior['fp']})")

    registrar("\n====================================================================================")

with open(arq_consolidado, "w", encoding="utf-8") as out_f:
    out_f.write("\n".join(linhas_saida) + "\n")