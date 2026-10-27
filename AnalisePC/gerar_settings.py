import os
import re
import sys
import shutil
import zipfile
import random
import argparse
import urllib.request
import ssl
from PIL import Image

try:
    import imagehash
except ImportError:
    print("[ERRO] Biblioteca 'imagehash' nao encontrada. Instale com: pip install ImageHash Pillow")
    sys.exit(1)

# ==============================================================================
# ARGUMENTOS DE LINHA DE COMANDO E PARAMETROS CONFIGURAVEIS
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--grupos", type=int, default=20, help="Quantidade de rodadas/grupos (padrao: 20)")
parser.add_argument("--usar_seeds_fixas", type=int, default=1, choices=[0, 1],
                    help="1: Usa as sementes fixas do array SEEDS (padrao) | 0: Gera sementes 100% aleatorias")
args = parser.parse_args()

NUM_GRUPOS = args.grupos
USAR_SEEDS_FIXAS = bool(args.usar_seeds_fixas)

ZIP_PATH = "freshness44.zip"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/siavash93/freshness44"

DATASET_DIR = "dataset"
FRESH_DIR = os.path.join(DATASET_DIR, "fresh")
ROTTEN_DIR = os.path.join(DATASET_DIR, "rotten")

SETTINGS_DIR = "settings"
RELATORIO_PHASH = os.path.join(SETTINGS_DIR, "relatorio_phash.txt")

# 50 sementes fixas pre-definidas para total reprodutibilidade
SEEDS = [
    42, 100, 2026, 7, 13, 99, 123, 456, 789, 1010,
    2020, 3030, 4040, 5050, 6060, 7070, 8080, 9090, 1111, 2222,
    3333, 4444, 5555, 6666, 7777, 8888, 9999, 1357, 2468, 3579,
    4680, 5791, 6802, 7913, 8024, 9135, 1024, 2048, 4096, 8192,
    16384, 32768, 65535, 31415, 27182, 16180, 14142, 17320, 22360, 24494
]

TEST_SPLIT = {
    "fresh": 100,
    "rotten": 100
}

CENARIOS = {
    "1": {"fresh_train": 510, "rotten_train": 510, "fresh_val": 90, "rotten_val": 90},
    "2": {"fresh_train": 510, "rotten_train": 255, "fresh_val": 90, "rotten_val": 45},
    "3": {"fresh_train": 255, "rotten_train": 510, "fresh_val": 45, "rotten_val": 90},
    "4": {"fresh_train":  51, "rotten_train":  26, "fresh_val":  9, "rotten_val":  4},
    "5": {"fresh_train":  26, "rotten_train":  51, "fresh_val":  4, "rotten_val":  9},
}


# ==============================================================================
# FUNCOES AUXILIARES DE NORMALIZACAO E LIMPEZA DE DIRETORIOS
# ==============================================================================
def normalizar_nome_kaggle(nome_esperado):
    return re.sub(r" \(1\)(\.[a-zA-Z]+)$", r"\1", nome_esperado)


def sem_extensao(nome):
    return os.path.splitext(nome)[0]


def chave_identidade_nome(caminho_raw):
    base = os.path.basename(caminho_raw.strip().replace("\\", "/"))
    base_sem_1 = normalizar_nome_kaggle(base).lower()
    return sem_extensao(base_sem_1)


def normalizar_caminho(caminho_raw):
    return os.path.normpath(caminho_raw.strip().replace("\\", os.sep).replace("/", os.sep))


def limpar_diretorio_seguro(diretorio):
    if not os.path.exists(diretorio):
        return
    try:
        shutil.rmtree(diretorio)
    except PermissionError:
        for root, dirs, files in os.walk(diretorio, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except PermissionError:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except OSError:
                    pass


def inicializar_pastas_limpas():
    print(f"[0/4] Apagando integralmente as pastas '{DATASET_DIR}/' e '{SETTINGS_DIR}/'...")
    limpar_diretorio_seguro(DATASET_DIR)
    limpar_diretorio_seguro(SETTINGS_DIR)
    os.makedirs(FRESH_DIR, exist_ok=True)
    os.makedirs(ROTTEN_DIR, exist_ok=True)
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    print("   [OK] Pastas limpas e recriadas do zero.")


# ==============================================================================
# PASSO 1: GARANTIR E DESCOMPACTAR O FRESHNESS44.ZIP
# ==============================================================================
def descompactar_zip():
    print(f"\n[1/4] Verificando e descompactando '{ZIP_PATH}'...")

    precisa_baixar = not os.path.isfile(ZIP_PATH) or os.path.getsize(ZIP_PATH) == 0
    if not precisa_baixar:
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.namelist()
            print(f"   [OK] '{ZIP_PATH}' ja existe localmente e esta integro.")
        except zipfile.BadZipFile:
            print(f"   [AVISO] '{ZIP_PATH}' corrompido. Sera baixado novamente.")
            os.remove(ZIP_PATH)
            precisa_baixar = True

    if precisa_baixar:
        print(f"   Baixando '{ZIP_PATH}' de {KAGGLE_URL} ...")
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(KAGGLE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp, open(ZIP_PATH, "wb") as out:
            shutil.copyfileobj(resp, out)
        print("   [OK] Download concluido.")

    cont_fresh = 0
    cont_rotten = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            member_norm = member.replace("\\", "/")
            nome_arq = os.path.basename(member_norm)
            if not nome_arq:
                continue

            if "Freshness44/Strawberry_Fresh/" in member_norm:
                destino = normalizar_caminho(os.path.join(FRESH_DIR, nome_arq))
                with zf.open(member) as src, open(destino, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                cont_fresh += 1

            elif "Freshness44/Strawberry_Rotten/" in member_norm:
                destino = normalizar_caminho(os.path.join(ROTTEN_DIR, nome_arq))
                with zf.open(member) as src, open(destino, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                cont_rotten += 1

    print(f"   [OK] Extraidos para '{FRESH_DIR}': {cont_fresh} arquivos.")
    print(f"   [OK] Extraidos para '{ROTTEN_DIR}': {cont_rotten} arquivos.")


# ==============================================================================
# PASSO 2: HIGIENIZACAO POR NOME BASE E PERCEPTUAL HASH (pHash) + RELATORIO
# ==============================================================================
def limpar_duplicatas_phash():
    print(f"\n[2/4] Executando deduplicacao (Nome Base + pHash) em '{DATASET_DIR}/'...")

    arquivos = []
    for pasta in [FRESH_DIR, ROTTEN_DIR]:
        for nome in os.listdir(pasta):
            caminho = normalizar_caminho(os.path.join(pasta, nome))
            if os.path.isfile(caminho) and nome.lower().endswith((".jpg", ".jpeg", ".png")):
                arquivos.append(caminho)

    arquivos.sort(key=lambda p: (
        1 if " (1)" in p else 0,
        0 if p.lower().endswith(".jpg") else 1,
        p.lower()
    ))

    nomes_base_vistos = {}
    hashes_vistos = {}
    duplicatas_removidas = []
    erros_leitura = []

    for caminho in arquivos:
        ch_nome = chave_identidade_nome(caminho)

        try:
            with Image.open(caminho) as img:
                h = str(imagehash.phash(img))
        except Exception as e:
            erros_leitura.append((caminho, str(e)))
            continue

        if ch_nome in nomes_base_vistos:
            original = nomes_base_vistos[ch_nome]
            os.remove(caminho)
            duplicatas_removidas.append((caminho, original, h, "MESMO_NOME_BASE"))
        elif h in hashes_vistos:
            original = hashes_vistos[h]
            os.remove(caminho)
            duplicatas_removidas.append((caminho, original, h, "MESMO_PHASH"))
        else:
            nomes_base_vistos[ch_nome] = caminho
            hashes_vistos[h] = caminho

    fresh_unicos = sorted([
        normalizar_caminho(os.path.join(FRESH_DIR, f))
        for f in os.listdir(FRESH_DIR)
        if os.path.isfile(os.path.join(FRESH_DIR, f))
    ])
    rotten_unicos = sorted([
        normalizar_caminho(os.path.join(ROTTEN_DIR, f))
        for f in os.listdir(ROTTEN_DIR)
        if os.path.isfile(os.path.join(ROTTEN_DIR, f))
    ])

    mapa_phash_ativos = {cam.lower(): h for h, cam in hashes_vistos.items()}

    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(RELATORIO_PHASH, "w", encoding="utf-8") as rel:
        rel.write("================ RELATORIO DE LIMPEZA (NOME BASE + PHASH) ================\n")
        rel.write(f"Total de imagens analisadas : {len(arquivos)}\n")
        rel.write(f"Duplicatas fisicas excluidas: {len(duplicatas_removidas)}\n")
        rel.write(f"Imagens unicas restantes    : {len(fresh_unicos) + len(rotten_unicos)}\n")
        rel.write(f"  -> Fresh unicas ({FRESH_DIR})  : {len(fresh_unicos)}\n")
        rel.write(f"  -> Rotten unicas ({ROTTEN_DIR}): {len(rotten_unicos)}\n")
        rel.write("==========================================================================\n\n")

        if duplicatas_removidas:
            rel.write("LISTA DE ARQUIVOS EXCLUIDOS FISICAMENTE:\n")
            for dup, orig, h, motivo in duplicatas_removidas:
                rel.write(f" [{motivo} | pHash: {h}] Excluido: {dup}  (Mantido: {orig})\n")

        if erros_leitura:
            rel.write("\nERROS DE LEITURA:\n")
            for arq, err in erros_leitura:
                rel.write(f" [ERRO] {arq}: {err}\n")

    print(f"   Analisadas: {len(arquivos)} | Excluidas: {len(duplicatas_removidas)}")
    print(f"   [OK] Restantes: {len(fresh_unicos)} Fresh | {len(rotten_unicos)} Rotten")
    print(f"   [OK] Relatorio salvo em: '{RELATORIO_PHASH}'")

    return fresh_unicos, rotten_unicos, mapa_phash_ativos


# ==============================================================================
# PASSO 3: GERAR OS N GRUPOS DE CENARIOS NA PASTA settings/
# ==============================================================================
def preparar_sementes(num_grupos, seeds_iniciais, usar_fixas=True):
    if not usar_fixas:
        print(f"   [MODO ALEATORIO] Sorteando {num_grupos} semente(s) totalmente aleatoria(s)...")
        sementes = []
        usadas = set()
        sys_rng = random.SystemRandom()
        while len(sementes) < num_grupos:
            nova_seed = sys_rng.randint(1, 9999999)
            if nova_seed not in usadas:
                sementes.append(nova_seed)
                usadas.add(nova_seed)
        return sementes

    print("   [MODO FIXO] Utilizando sementes pre-definidas para reprodutibilidade...")
    sementes = list(seeds_iniciais[:num_grupos])
    if len(sementes) < num_grupos:
        faltam = num_grupos - len(sementes)
        print(f"   Gerando {faltam} semente(s) complementar(es) de forma deterministica...")
        usadas = set(sementes)
        rng_extra = random.Random(0)
        while len(sementes) < num_grupos:
            nova_seed = rng_extra.randint(100000, 999999)
            if nova_seed not in usadas:
                sementes.append(nova_seed)
                usadas.add(nova_seed)
    return sementes


def formatar_linhas(lista_caminhos, classe_id, classe_nome):
    return [f"{classe_id};{classe_nome};{cam}" for cam in lista_caminhos]


def gerar_arquivos_settings(fresh_pool, rotten_pool):
    print(f"\n[3/4] Gerando {NUM_GRUPOS} grupo(s) para {len(CENARIOS)} cenarios na pasta '{SETTINGS_DIR}/'...")

    max_fresh_req = TEST_SPLIT["fresh"] + max(c["fresh_train"] + c["fresh_val"] for c in CENARIOS.values())
    max_rotten_req = TEST_SPLIT["rotten"] + max(c["rotten_train"] + c["rotten_val"] for c in CENARIOS.values())

    if len(fresh_pool) < max_fresh_req or len(rotten_pool) < max_rotten_req:
        print("   [ERRO] Quantidade insuficiente de imagens unicas apos limpeza:")
        print(f"      Fresh disponiveis : {len(fresh_pool)} (Necessarias: {max_fresh_req})")
        print(f"      Rotten disponiveis: {len(rotten_pool)} (Necessarias: {max_rotten_req})")
        sys.exit(1)

    sementes = preparar_sementes(NUM_GRUPOS, SEEDS, USAR_SEEDS_FIXAS)

    mapa_seeds_path = os.path.join(SETTINGS_DIR, "sementes_utilizadas.txt")
    with open(mapa_seeds_path, "w", encoding="utf-8") as fs:
        fs.write("GRUPO;SEMENTE\n")
        for idx, s in enumerate(sementes, 1):
            fs.write(f"{idx};{s}\n")

    total_txts = 0

    for idx_grupo, seed in enumerate(sementes, 1):
        rng = random.Random(seed)

        f_list = list(fresh_pool)
        r_list = list(rotten_pool)
        rng.shuffle(f_list)
        rng.shuffle(r_list)

        n_test_f = TEST_SPLIT["fresh"]
        n_test_r = TEST_SPLIT["rotten"]

        test_fresh = f_list[:n_test_f]
        test_rotten = r_list[:n_test_r]

        rem_fresh = f_list[n_test_f:]
        rem_rotten = r_list[n_test_r:]

        linhas_test = (
            formatar_linhas(test_rotten, 0, "rotten") +
            formatar_linhas(test_fresh, 1, "fresh")
        )
        rng.shuffle(linhas_test)

        for id_cenario, cfg in CENARIOS.items():
            f_tr_qtd = cfg["fresh_train"]
            r_tr_qtd = cfg["rotten_train"]
            f_val_qtd = cfg["fresh_val"]
            r_val_qtd = cfg["rotten_val"]

            c_train_f = rem_fresh[:f_tr_qtd]
            c_val_f = rem_fresh[f_tr_qtd : f_tr_qtd + f_val_qtd]

            c_train_r = rem_rotten[:r_tr_qtd]
            c_val_r = rem_rotten[r_tr_qtd : r_tr_qtd + r_val_qtd]

            linhas_train = (
                formatar_linhas(c_train_r, 0, "rotten") +
                formatar_linhas(c_train_f, 1, "fresh")
            )
            linhas_val = (
                formatar_linhas(c_val_r, 0, "rotten") +
                formatar_linhas(c_val_f, 1, "fresh")
            )

            rng.shuffle(linhas_train)
            rng.shuffle(linhas_val)

            prefixo = os.path.join(SETTINGS_DIR, f"setting{id_cenario}.{idx_grupo}")

            with open(f"{prefixo}-train.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(linhas_train) + "\n")
            with open(f"{prefixo}-val.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(linhas_val) + "\n")
            with open(f"{prefixo}-test.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(linhas_test) + "\n")

            total_txts += 3

    print(f"   [OK] {total_txts} arquivos .txt gerados em '{SETTINGS_DIR}/'.")
    print(f"   [OK] Registro de sementes salvo em: '{mapa_seeds_path}'")


# ==============================================================================
# PASSO 4: AUDITORIA POS-GERACAO (EXISTENCIA FISICA, VAZAMENTOS E METAS)
# ==============================================================================
def ler_registros_txt(arquivo_txt):
    registros = []
    with open(arquivo_txt, "r", encoding="utf-8") as f:
        for num_linha, linha in enumerate(f, 1):
            linha = linha.strip()
            if not linha or ";" not in linha:
                continue
            partes = linha.split(";")
            id_cls = partes[0].strip()
            nome_cls = partes[1].strip().lower()
            caminho_norm = normalizar_caminho(partes[2])
            cls = "fresh" if (id_cls == "1" or "fresh" in nome_cls) else "rotten"
            chave_nome = chave_identidade_nome(caminho_norm)
            registros.append((num_linha, caminho_norm, chave_nome, cls))
    return registros


def auditar_settings_gerados(mapa_phash):
    print(f"\n[4/4] Auditando todos os arquivos gerados em '{SETTINGS_DIR}/'...")
    erros = 0
    arquivos_faltando = 0

    for idx_grupo in range(1, NUM_GRUPOS + 1):
        for id_cenario, cfg in CENARIOS.items():
            prefixo = os.path.join(SETTINGS_DIR, f"setting{id_cenario}.{idx_grupo}")
            tr = ler_registros_txt(f"{prefixo}-train.txt")
            val = ler_registros_txt(f"{prefixo}-val.txt")
            ts = ler_registros_txt(f"{prefixo}-test.txt")

            cont = {"train": {"fresh": 0, "rotten": 0}, "val": {"fresh": 0, "rotten": 0}, "test": {"fresh": 0, "rotten": 0}}
            nomes_split = {"train": set(), "val": set(), "test": set()}
            hashes_split = {"train": set(), "val": set(), "test": set()}

            for nome_etapa, regs in [("train", tr), ("val", val), ("test", ts)]:
                for lin, cam, ch_nome, cls in regs:
                    if not os.path.isfile(cam):
                        arquivos_faltando += 1
                        erros += 1
                        print(f"   [ERRO] Arquivo ausente no disco ({prefixo}-{nome_etapa}.txt:{lin}): {cam}")
                    cont[nome_etapa][cls] += 1

                    if ch_nome in nomes_split[nome_etapa]:
                        erros += 1
                        print(f"   [ERRO] Nome duplicado interno em {prefixo}-{nome_etapa}.txt: {ch_nome}")
                    nomes_split[nome_etapa].add(ch_nome)

                    h = mapa_phash.get(cam.lower())
                    if h:
                        if h in hashes_split[nome_etapa]:
                            erros += 1
                            print(f"   [ERRO] pHash duplicado interno em {prefixo}-{nome_etapa}.txt: {h}")
                        hashes_split[nome_etapa].add(h)

            for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
                vaz_nome = nomes_split[s1] & nomes_split[s2]
                vaz_hash = hashes_split[s1] & hashes_split[s2]
                if vaz_nome or vaz_hash:
                    erros += 1
                    print(f"   [ERRO] Vazamento em {prefixo} ({s1}<->{s2}): Nomes={len(vaz_nome)}, pHash={len(vaz_hash)}")

            if (cont["train"]["fresh"] != cfg["fresh_train"] or
                cont["train"]["rotten"] != cfg["rotten_train"] or
                cont["val"]["fresh"] != cfg["fresh_val"] or
                cont["val"]["rotten"] != cfg["rotten_val"] or
                cont["test"]["fresh"] != TEST_SPLIT["fresh"] or
                cont["test"]["rotten"] != TEST_SPLIT["rotten"]):
                erros += 1
                print(f"   [ERRO] Divergencia de quantidade em {prefixo}: {cont}")

    if erros > 0:
        print(f"\n[FALHA] Auditoria detectou {erros} erro(s) ({arquivos_faltando} arquivo(s) ausente(s)).")
        sys.exit(1)

    print(f"   [OK] Zero vazamento de nome ou pHash nos {NUM_GRUPOS} grupos e {len(CENARIOS)} cenarios.")
    print("   [OK] 100% dos arquivos referenciados nos .txt existem fisicamente no disco.")
    print("\n================ RESUMO DA CONFIGURACAO VALIDADA (POR GRUPO) ================")
    for id_cen, cfg in sorted(CENARIOS.items()):
        soma_f = cfg["fresh_train"] + cfg["fresh_val"]
        soma_r = cfg["rotten_train"] + cfg["rotten_val"]
        print(
            f" CENARIO {id_cen}: Train+Val = {soma_f + soma_r:4d} "
            f"(Train: {cfg['fresh_train']:3d}F/{cfg['rotten_train']:3d}R | "
            f"Val: {cfg['fresh_val']:2d}F/{cfg['rotten_val']:2d}R) | "
            f"Test = {TEST_SPLIT['fresh'] + TEST_SPLIT['rotten']:3d} "
            f"({TEST_SPLIT['fresh']}F/{TEST_SPLIT['rotten']}R) [OK]"
        )
    print("=============================================================================")


if __name__ == "__main__":
    inicializar_pastas_limpas()
    descompactar_zip()
    fresh_limpos, rotten_limpos, mapa_phash = limpar_duplicatas_phash()
    gerar_arquivos_settings(fresh_limpos, rotten_limpos)
    auditar_settings_gerados(mapa_phash)