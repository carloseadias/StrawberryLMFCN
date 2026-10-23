import os
import re
import glob
import shutil
import zipfile
import urllib.request
import ssl
import sys
from PIL import Image

try:
    import imagehash
except ImportError:
    print(" Biblioteca 'imagehash' não encontrada! Instale com: pip install ImageHash Pillow")
    sys.exit(1)

ZIP_PATH = "freshness44.zip"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/siavash93/freshness44"


def normalizar_nome_kaggle(nome_esperado):
    """Remove o sufixo ' (1)' antes da extensão."""
    return re.sub(r" \(1\)(\.[a-zA-Z]+)$", r"\1", nome_esperado)


def sem_extensao(nome):
    return os.path.splitext(nome)[0]


def chave_identidade_nome(caminho_raw):
    base = os.path.basename(caminho_raw.strip().replace("\\", "/"))
    base_sem_1 = normalizar_nome_kaggle(base).lower()
    return sem_extensao(base_sem_1)


def normalizar_caminho(caminho_raw):
    return os.path.normpath(caminho_raw.strip().replace("\\", os.sep).replace("/", os.sep))


def ler_registros_txt(arquivo_txt):
    """Retorna lista de tuplas: (num_linha, caminho_normalizado, chave_nome, classe_str)"""
    if not os.path.isfile(arquivo_txt):
        return None
    registros = []
    with open(arquivo_txt, "r", encoding="utf-8") as f:
        for num_linha, linha in enumerate(f, 1):
            linha = linha.strip()
            if not linha or ";" not in linha:
                continue
            partes = linha.split(";")
            id_cls = partes[0].strip()
            nome_cls = partes[1].strip().lower() if len(partes) >= 2 else ""
            caminho_raw = partes[2].strip() if len(partes) >= 3 else partes[-1].strip()
            
            cls = "fresh" if (id_cls == "1" or "fresh" in nome_cls) else "rotten"
            caminho_norm = normalizar_caminho(caminho_raw)
            chave_nome = chave_identidade_nome(caminho_raw)
            registros.append((num_linha, caminho_norm, chave_nome, cls))
    return registros


# ==============================================================================
# PASSO 1: CHECAR VAZAMENTO POR NOME DE ARQUIVO (IGNORANDO A PASTA!)
# ==============================================================================
def auditar_vazamento_nomes():
    print("[1/6] Auditando vazamento por NOME DE ARQUIVO nos .txt (ignorando pastas)...")
    arquivos_txt = sorted(glob.glob("setting*.txt"))
    if not arquivos_txt:
        print("    ERRO: Nenhum arquivo 'setting*.txt' encontrado!")
        sys.exit(1)

    erros = False
    dados_cenarios = {}

    for i in range(1, 6):
        cenario = f"setting{i}"
        tr = ler_registros_txt(f"{cenario}-train.txt")
        val = ler_registros_txt(f"{cenario}-val.txt")
        ts = ler_registros_txt(f"{cenario}-test.txt")

        if tr is None and val is None and ts is None:
            continue
        if tr is None or val is None:
            print(f"    ERRO: Arquivos train ou val ausentes para {cenario}.")
            sys.exit(1)

        dados_cenarios[cenario] = {"train": tr, "val": val, "test": ts or []}

        nomes_tr = {r[2]: r[1] for r in tr}
        nomes_val = {r[2]: r[1] for r in val}
        nomes_ts = {r[2]: r[1] for r in (ts or [])}

        for etapa_nome, lista_reg in [("train", tr), ("val", val), ("test", ts or [])]:
            vistos = {}
            for lin, cam, ch, _ in lista_reg:
                if ch in vistos:
                    erros = True
                    print(f"    DUPLICATA INTERNA em {cenario}-{etapa_nome}.txt (linha {lin}):")
                    print(f"      -> '{cam}' tem o mesmo nome base de '{vistos[ch]}'")
                vistos[ch] = cam

        pares = [
            ("TRAIN", "VAL", nomes_tr, nomes_val),
            ("TRAIN", "TEST", nomes_tr, nomes_ts),
            ("VAL", "TEST", nomes_val, nomes_ts),
        ]
        for n1, n2, d1, d2 in pares:
            intersecao = set(d1.keys()) & set(d2.keys())
            if intersecao:
                erros = True
                print(f"    VAZAMENTO DE NOME no {cenario.upper()} entre {n1} e {n2} ({len(intersecao)} arquivos):")
                for ch in sorted(intersecao):
                    print(f"      -> {n1}: {d1[ch]}  <===>  {n2}: {d2[ch]}")

    if erros:
        print("\n[ABORTADO] Vazamento por nome detectado nos .txt! Corrija antes de prosseguir.")
        sys.exit(1)

    caminhos_exigidos = {}
    for txt in arquivos_txt:
        for _, cam, _, _ in ler_registros_txt(txt):
            caminhos_exigidos[cam.lower()] = cam

    print("    Zero vazamento por nome de arquivo!")
    return arquivos_txt, dados_cenarios, caminhos_exigidos


# ==============================================================================
# PASSO 2: GARANTIR QUE O ARQUIVO ZIP EXISTE
# ==============================================================================
def garantir_zip(zip_path=ZIP_PATH):
    print(f"\n[2/6] Verificando arquivo '{zip_path}'...")
    if os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0:
        print(f"    '{zip_path}' já existe localmente.")
        return

    print(f"   ⬇️ '{zip_path}' não encontrado. Baixando de {KAGGLE_URL} ...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(KAGGLE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("    Download concluído!")


# ==============================================================================
# PASSO 3: CHECAR SE TUDO DOS .TXT ESTÁ NO ZIP
# ==============================================================================
def mapear_zip(caminhos_exigidos, zip_path=ZIP_PATH):
    print(f"\n[3/6] Checando se todas as imagens dos .txt existem em '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        lista_zip = zf.namelist()

    indice_exato = {}
    indice_sem_ext = {}
    for item in lista_zip:
        if item.endswith("/"):
            continue
        base = os.path.basename(item)
        if "strawberry" in base.lower():
            b_low = base.lower()
            indice_exato[b_low] = item
            indice_sem_ext[sem_extensao(b_low)] = item

    mapeamento = {}
    faltando = []
    for destino_rel in caminhos_exigidos.values():
        nome_alvo = os.path.basename(destino_rel)
        alvo_low = nome_alvo.lower()
        alvo_sem_1 = normalizar_nome_kaggle(nome_alvo).lower()

        if alvo_low in indice_exato:
            mapeamento[destino_rel] = indice_exato[alvo_low]
        elif alvo_sem_1 in indice_exato:
            mapeamento[destino_rel] = indice_exato[alvo_sem_1]
        elif sem_extensao(alvo_sem_1) in indice_sem_ext:
            mapeamento[destino_rel] = indice_sem_ext[sem_extensao(alvo_sem_1)]
        else:
            faltando.append(destino_rel)

    if faltando:
        print(f"    ERRO: {len(faltando)} arquivo(s) não existem no ZIP:")
        for f in faltando:
            print(f"      -> {f}")
        sys.exit(1)

    print(f"    Todas as {len(mapeamento)} imagens foram localizadas no ZIP.")
    return mapeamento


# ==============================================================================
# PASSO 4: CHECAR VAZAMENTO VISUAL (pHash) ENTRE TRAIN, VAL E TEST
# ==============================================================================
def auditar_vazamento_phash(dados_cenarios, mapeamento, zip_path=ZIP_PATH):
    print("\n[4/6] Calculando Perceptual Hash (pHash) e checando vazamento visual entre splits...")
    phash_por_caminho = {}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        phash_por_membro_zip = {}
        for destino_rel, item_zip in mapeamento.items():
            if item_zip not in phash_por_membro_zip:
                with zf.open(item_zip) as stream:
                    with Image.open(stream) as img:
                        phash_por_membro_zip[item_zip] = str(imagehash.phash(img))
            phash_por_caminho[destino_rel.lower()] = phash_por_membro_zip[item_zip]

    erros_phash = False
    for cenario, splits in dados_cenarios.items():
        hashes_tr = {}
        hashes_val = {}
        hashes_ts = {}

        for etapa_nome, reg_lista, dic_destino in [
            ("TRAIN", splits["train"], hashes_tr),
            ("VAL", splits["val"], hashes_val),
            ("TEST", splits["test"], hashes_ts),
        ]:
            for lin, cam, _, _ in reg_lista:
                h = phash_por_caminho[cam.lower()]
                if h in dic_destino and dic_destino[h] != cam:
                    erros_phash = True
                    print(f"    DUPLICATA VISUAL (mesmo pHash={h}) dentro de {cenario}-{etapa_nome}:")
                    print(f"      -> '{cam}' é visualmente idêntica a '{dic_destino[h]}'")
                dic_destino[h] = cam

        pares = [
            ("TRAIN", "VAL", hashes_tr, hashes_val),
            ("TRAIN", "TEST", hashes_tr, hashes_ts),
            ("VAL", "TEST", hashes_val, hashes_ts),
        ]
        for n1, n2, d1, d2 in pares:
            vaz_visual = set(d1.keys()) & set(d2.keys())
            if vaz_visual:
                erros_phash = True
                print(f"    VAZAMENTO VISUAL (pHash idêntico) no {cenario.upper()} entre {n1} e {n2} ({len(vaz_visual)} casos):")
                for h in sorted(vaz_visual):
                    print(f"      -> [pHash: {h}] {n1}: {d1[h]}  <===>  {n2}: {d2[h]}")

    if erros_phash:
        print("\n[ABORTADO] Vazamento ou duplicata visual (pHash) detectado nos .txt!")
        sys.exit(1)

    print("    Zero vazamento visual (pHash)! Todas as imagens entre Train, Val e Test são visualmente únicas.")


# ==============================================================================
# PASSO 5 & 6: MONTAR PASTA DATASET E EXIBIR CONTAGEM DETALHADA (FRESH / ROTTEN)
# ==============================================================================
def montar_e_checar(mapeamento, arquivos_txt, zip_path=ZIP_PATH, dataset_dir="dataset"):
    print(f"\n[5/6] Montando a pasta '{dataset_dir}/' com base nos .txt...")
    if os.path.exists(dataset_dir):
        try:
            shutil.rmtree(dataset_dir)
        except PermissionError:
            for root, _, files in os.walk(dataset_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except PermissionError:
                        pass

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for destino_rel, item_zip in mapeamento.items():
            os.makedirs(os.path.dirname(destino_rel), exist_ok=True)
            with zf.open(item_zip) as src, open(destino_rel, "wb") as dst:
                shutil.copyfileobj(src, dst)

    print(f"\n[6/6] Conferindo fisicamente cada linha dos {len(arquivos_txt)} arquivos .txt no disco...\n")
    faltando_total = 0
    resumo_cenarios = {}

    for txt in arquivos_txt:
        regs = ler_registros_txt(txt)
        enc_fresh = 0
        enc_rotten = 0
        falt = 0

        for _, cam, _, cls in regs:
            if os.path.isfile(cam):
                if cls == "fresh":
                    enc_fresh += 1
                else:
                    enc_rotten += 1
            else:
                falt += 1

        faltando_total += falt
        total_enc = enc_fresh + enc_rotten
        status = " OK" if falt == 0 else f" FALTAM {falt}"
        print(
            f"   {txt:18s} -> Encontradas: {total_enc:4d} "
            f"(Fresh: {enc_fresh:3d} | Rotten: {enc_rotten:3d}) | "
            f"Faltando: {falt:2d} [{status}]"
        )

        # Agrupa para mostrar o somatório Train + Val por cenário
        m = re.match(r"(setting\d+)-(train|val|test)\.txt", txt.lower())
        if m:
            cen, etapa = m.group(1), m.group(2)
            resumo_cenarios.setdefault(cen, {})[etapa] = (enc_fresh, enc_rotten)

    print("\n================ SOMA (TRAIN + VAL) POR CENÁRIO vs TABELA DO ARTIGO ================")
    metas = {
        "setting1": (600, 600),
        "setting2": (600, 300),
        "setting3": (300, 600),
        "setting4": (60, 30),
        "setting5": (30, 60),
    }
    for cen in sorted(resumo_cenarios.keys()):
        tr_f, tr_r = resumo_cenarios[cen].get("train", (0, 0))
        va_f, va_r = resumo_cenarios[cen].get("val", (0, 0))
        ts_f, ts_r = resumo_cenarios[cen].get("test", (0, 0))
        soma_f, soma_r = tr_f + va_f, tr_r + va_r
        meta_f, meta_r = metas.get(cen, (0, 0))
        diff_f, diff_r = soma_f - meta_f, soma_r - meta_r

        status_cen = " BATEU" if (diff_f == 0 and diff_r == 0) else f"⚠️ DIFERENÇA (Fresh: {diff_f:+d} | Rotten: {diff_r:+d})"
        print(
            f"  {cen.upper()}: Train+Val = {soma_f + soma_r:4d} "
            f"(Fresh: {soma_f:3d}/{meta_f:3d} | Rotten: {soma_r:3d}/{meta_r:3d}) "
            f"| Test = {ts_f + ts_r:3d} (F:{ts_f}|R:{ts_r}) -> {status_cen}"
        )
    print("====================================================================================")

    if faltando_total == 0:
        print("\n Todos os arquivos listados existem fisicamente no disco!")


if __name__ == "__main__":
    lista_txts, dados_cenarios, caminhos_unicos = auditar_vazamento_nomes()
    garantir_zip()
    mapa_zip = mapear_zip(caminhos_unicos)
    auditar_vazamento_phash(dados_cenarios, mapa_zip)
    montar_e_checar(mapa_zip, lista_txts)