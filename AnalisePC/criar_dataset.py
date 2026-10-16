import os
import re
import shutil
import zipfile
import urllib.request
import ssl

# ==============================================================================
# 1. CONJUNTO DE TESTE EXATO (100 Fresh + 97 Rotten = 197 imagens)
# ==============================================================================
TEST_FRESH = {
    "FreshStrawberry (105)!.jpg", "FreshStrawberry (110)!.jpg", "FreshStrawberry (114)!.jpg",
    "FreshStrawberry (132)!.jpg", "FreshStrawberry (137)!.jpg", "FreshStrawberry (150)!.jpg",
    "FreshStrawberry (158)!.jpg", "FreshStrawberry (182)!.jpg", "FreshStrawberry (188)!.jpg",
    "FreshStrawberry (19)!.jpg", "FreshStrawberry (191)!.jpg", "FreshStrawberry (195)!.jpg",
    "FreshStrawberry (200)!.jpg", "FreshStrawberry (201)!.jpg", "FreshStrawberry (25)!.jpg",
    "FreshStrawberry (34)!.jpg", "FreshStrawberry (35)!.jpg", "FreshStrawberry (39)!.jpg",
    "FreshStrawberry (42)!.jpg", "FreshStrawberry (46)!.jpg", "FreshStrawberry (53)!.jpg",
    "FreshStrawberry (57)!.jpg", "FreshStrawberry (72)!.jpg", "FreshStrawberry (80)!.jpg",
    "FreshStrawberry (94)!.jpg", "freshStrawberry (103)1.jpg", "freshStrawberry (108)1.jpg",
    "freshStrawberry (113)1.jpg", "freshStrawberry (115)1.jpg", "freshStrawberry (130)1.jpg",
    "freshStrawberry (137)1.jpg", "freshStrawberry (138)1.jpg", "freshStrawberry (140)1.jpg",
    "freshStrawberry (144)1.jpg", "freshStrawberry (145)1.jpg", "freshStrawberry (151)1.jpg",
    "freshStrawberry (161)1.jpg", "freshStrawberry (17)1.jpg", "freshStrawberry (182)1.jpg",
    "freshStrawberry (183)1.jpg", "freshStrawberry (188)1.jpg", "freshStrawberry (19)1.jpg",
    "freshStrawberry (209)1.jpg", "freshStrawberry (216)1.jpg", "freshStrawberry (222)1.jpg",
    "freshStrawberry (276)1.jpg", "freshStrawberry (282)1.jpg", "freshStrawberry (296)1.jpg",
    "freshStrawberry (313)1.jpg", "freshStrawberry (319)1.jpg", "freshStrawberry (324)1.jpg",
    "freshStrawberry (328)1.jpg", "freshStrawberry (338)1.jpg", "freshStrawberry (342)1.jpg",
    "freshStrawberry (35)1.jpg", "freshStrawberry (351)1.jpg", "freshStrawberry (355)1.jpg",
    "freshStrawberry (356)1.jpg", "freshStrawberry (361)1.jpg", "freshStrawberry (368)1.jpg",
    "freshStrawberry (374)1.jpg", "freshStrawberry (382)1.jpg", "freshStrawberry (384)1.jpg",
    "freshStrawberry (391)1.jpg", "freshStrawberry (393)1.jpg", "freshStrawberry (404)1.jpg",
    "freshStrawberry (407)1.jpg", "freshStrawberry (415)1.jpg", "freshStrawberry (417)1.jpg",
    "freshStrawberry (432)1.jpg", "freshStrawberry (445)1.jpg", "freshStrawberry (450)1.jpg",
    "freshStrawberry (457)1.jpg", "freshStrawberry (47)1.jpg", "freshStrawberry (470)1.jpg",
    "freshStrawberry (472)1.jpg", "freshStrawberry (474)1.jpg", "freshStrawberry (48)1.jpg",
    "freshStrawberry (483)1.jpg", "freshStrawberry (506)1.jpg", "freshStrawberry (513)1.jpg",
    "freshStrawberry (523)1.jpg", "freshStrawberry (524)1.jpg", "freshStrawberry (528)1.jpg",
    "freshStrawberry (529)1.jpg", "freshStrawberry (533)1.jpg", "freshStrawberry (539)1.jpg",
    "freshStrawberry (543)1.jpg", "freshStrawberry (545)1.jpg", "freshStrawberry (553)1.jpg",
    "freshStrawberry (556)1.jpg", "freshStrawberry (56)1.jpg", "freshStrawberry (561)1.jpg",
    "freshStrawberry (591)1.jpg", "freshStrawberry (597)1.jpg", "freshStrawberry (62)1.jpg",
    "freshStrawberry (66)1.jpg", "freshStrawberry (80)1.jpg", "freshStrawberry (9)1.jpg",
    "freshStrawberry (98)1.jpg"
}

TEST_ROTTEN = {
    "RottenStrawberry (10)!.jpg", "RottenStrawberry (113)!.jpg", "RottenStrawberry (119)!.jpg",
    "RottenStrawberry (12)!.jpg", "RottenStrawberry (125)!.jpg", "RottenStrawberry (126)!.jpg",
    "RottenStrawberry (128)!.jpg", "RottenStrawberry (131)!.jpg", "RottenStrawberry (14)!.jpg",
    "RottenStrawberry (143)!.jpg", "RottenStrawberry (163)!.jpg", "RottenStrawberry (165)!.jpg",
    "RottenStrawberry (17)!.jpg", "RottenStrawberry (179)!.jpg", "RottenStrawberry (180)!.jpg",
    "RottenStrawberry (181)!.jpg", "RottenStrawberry (19)!.jpg", "RottenStrawberry (202)!.jpg",
    "RottenStrawberry (205)!.jpg", "RottenStrawberry (21)!.jpg", "RottenStrawberry (22)!.jpg",
    "RottenStrawberry (32)!.jpg", "RottenStrawberry (39)!.jpg", "RottenStrawberry (48)!.jpg",
    "RottenStrawberry (50)!.jpg", "RottenStrawberry (72)!.jpg", "RottenStrawberry (79)!.jpg",
    "RottenStrawberry (86)!.jpg", "RottenStrawberry (96)!.jpg", "rottenStrawberry (111)1.jpg",
    "rottenStrawberry (117)1.jpg", "rottenStrawberry (12)1.jpg", "rottenStrawberry (121)1.jpg",
    "rottenStrawberry (136)1.jpg", "rottenStrawberry (140)1.jpg", "rottenStrawberry (146)1.jpg",
    "rottenStrawberry (149)1.jpg", "rottenStrawberry (161)1.jpg", "rottenStrawberry (177)1.jpg",
    "rottenStrawberry (19)1.jpg", "rottenStrawberry (190)1.jpg", "rottenStrawberry (2)1.jpg",
    "rottenStrawberry (205)1.jpg", "rottenStrawberry (235)1.jpg", "rottenStrawberry (240)1.jpg",
    "rottenStrawberry (243)1.jpg", "rottenStrawberry (244)1.jpg", "rottenStrawberry (250)1.jpg",
    "rottenStrawberry (254)1.jpg", "rottenStrawberry (257)1.jpg", "rottenStrawberry (261)1.jpg",
    "rottenStrawberry (264)1.jpg", "rottenStrawberry (266)1 (1).jpg", "rottenStrawberry (272)1.jpg",
    "rottenStrawberry (28)1.jpg", "rottenStrawberry (290)1.jpg", "rottenStrawberry (295)1 (1).jpg",
    "rottenStrawberry (30)1.jpg", "rottenStrawberry (301)1 (1).jpg", "rottenStrawberry (312)1 (1).jpg",
    "rottenStrawberry (326)1.jpg", "rottenStrawberry (33)1.jpg", "rottenStrawberry (331)1.jpg",
    "rottenStrawberry (332)1.jpg", "rottenStrawberry (333)1 (1).jpg", "rottenStrawberry (334)1.jpg",
    "rottenStrawberry (34)1.jpg", "rottenStrawberry (351)1.jpg", "rottenStrawberry (357)1.jpg",
    "rottenStrawberry (369)1.jpg", "rottenStrawberry (371)1 (1).jpg", "rottenStrawberry (388)1.jpg",
    "rottenStrawberry (392)1.jpg", "rottenStrawberry (439)1.jpg", "rottenStrawberry (44)1.jpg",
    "rottenStrawberry (452)1.jpg", "rottenStrawberry (453)1.jpg", "rottenStrawberry (455)1.jpg",
    "rottenStrawberry (471)1.jpg", "rottenStrawberry (472)1.jpg", "rottenStrawberry (486)1.jpg",
    "rottenStrawberry (50)1.jpg", "rottenStrawberry (501)1.jpg", "rottenStrawberry (502)1.jpg",
    "rottenStrawberry (504)1.jpg", "rottenStrawberry (512)1.jpg", "rottenStrawberry (516)1.jpg",
    "rottenStrawberry (519)1.jpg", "rottenStrawberry (527)1.jpg", "rottenStrawberry (546)1.jpg",
    "rottenStrawberry (549)1.jpg", "rottenStrawberry (552)1.jpg", "rottenStrawberry (62)1.jpg",
    "rottenStrawberry (8)1.jpg", "rottenStrawberry (83)1.jpg", "rottenStrawberry (9)1.jpg",
    "rottenStrawberry (99)1.jpg"
}

# ==============================================================================
# 2. GERADOR HARDCODED EXATO DE TREINO_VALIDACAO (600 Fresh + 582 Rotten = 1182)
# Reconstruído matematicamente a partir de setting1-train_3.txt e setting1-val_3.txt
# ==============================================================================
def construir_conjuntos_treino_val():
    # --- FRESH (600 imagens) ---
    # No dataset completo (treino+val+teste) de morangos existem:
    # - FreshStrawberry (i)!.jpg de 10 a 209, exceto {18, 44, 51, 67, 69, 70, 91, 93, 98, 106, 118, 120, 121, 126, 133, 134, 143, 149, 153, 174, 179, 196} (alguns estão no val)
    # Vamos construir a lista total de 700 Fresh e remover os 100 do TEST_FRESH:
    missing_bang_fresh = {18, 44, 51, 67, 69, 70, 91, 93, 98, 118, 120, 121, 126, 133, 134, 143, 149, 153, 174, 179}
    all_fresh = {f"FreshStrawberry ({i})!.jpg" for i in range(10, 210) if i not in missing_bang_fresh}

    missing_one_fresh = {
        1, 18, 22, 24, 42, 46, 61, 79, 82, 90, 102, 118, 120, 124, 131, 147, 148, 156, 159, 172,
        190, 199, 213, 219, 221, 223, 227, 231, 233, 235, 237, 239, 241, 242, 243, 245, 247, 248,
        249, 251, 252, 253, 254, 255, 257, 259, 261, 263, 269, 270, 271, 275, 281, 283, 285, 287,
        288, 289, 293, 300, 303, 307, 315, 317, 321, 325, 327, 329, 333, 337, 347, 353, 389, 399,
        402, 419, 439, 447, 463, 468, 481, 512, 565, 569, 572, 595
    }
    # Nota: 389, 481 e 595 estão em train/val, então garantimos sua presença:
    missing_one_fresh -= {389, 481, 595}
    all_fresh |= {f"freshStrawberry ({i})1.jpg" for i in range(2, 605) if i not in missing_one_fresh}
    tr_val_fresh = all_fresh - TEST_FRESH

    # --- ROTTEN (582 imagens) ---
    missing_bang_rotten = {
        19, 39, 54, 57, 63, 68, 81, 91, 93, 113, 125, 135, 143, 162, 166, 180, 184, 188, 190
    }
    all_rotten = {f"RottenStrawberry ({i})!.jpg" for i in range(10, 210) if i not in missing_bang_rotten}
    all_rotten.add("RottenStrawberry (93)! (1).jpg")

    missing_one_rotten = {
        1, 2, 19, 22, 25, 27, 32, 35, 42, 45, 49, 64, 65, 66, 71, 75, 81, 85, 86, 90, 92, 109,
        116, 132, 144, 148, 166, 181, 197, 209, 220, 227, 230, 238, 247, 259, 266, 269, 276, 279,
        286, 291, 292, 295, 297, 299, 301, 304, 312, 321, 324, 330, 333, 338, 339, 348, 349, 350,
        353, 354, 359, 360, 361, 362, 363, 365, 366, 368, 371, 372, 377, 378, 380, 382, 386, 389,
        393, 397, 398, 405, 409, 411, 412, 415, 416, 417, 418, 419, 421, 422, 424, 425, 426, 427,
        435, 447, 454, 469, 481, 483, 492, 493, 496, 518, 531, 548, 551, 555, 563
    }
    missing_one_rotten -= {518}  # 518 está no val
    all_rotten |= {f"rottenStrawberry ({i})1.jpg" for i in range(3, 571) if i not in missing_one_rotten}

    # Os 11 arquivos de treino/val que têm sufixo " (1).jpg"
    dup_rotten_tr_val = [269, 286, 291, 297, 299, 304, 321, 324, 330, 338, 348, 350, 361, 363, 377, 380, 393, 405, 409, 412]
    for d in dup_rotten_tr_val:
        all_rotten.add(f"rottenStrawberry ({d})1 (1).jpg")

    tr_val_rotten = all_rotten - TEST_ROTTEN
    return tr_val_fresh, tr_val_rotten


def normalizar_nome_kaggle(nome_esperado):
    """
    Remove o sufixo ' (1)' antes da extensão para encontrar o arquivo correspondente
    dentro do zip original do Kaggle.
    Ex: 'rottenStrawberry (266)1 (1).jpg' -> 'rottenStrawberry (266)1.jpg'
    """
    return re.sub(r" \(1\)(\.[a-zA-Z]+)$", r"\1", nome_esperado)


def baixar_freshness44(destino_temp="temp_freshness44", zip_path="freshness44.zip"):
    os.makedirs(destino_temp, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/download/siavash93/freshness44"

    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        print(f"[1/3] Arquivo '{zip_path}' já encontrado localmente. Pulando download!")
    else:
        print(f"[1/3] A descarregar de {url} ...")
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("      Download concluído!")

    extract_dir = os.path.join(destino_temp, "extracted")
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        print("[2/3] A descompactar ficheiros...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

    return extract_dir


def garantir_dataset(base_dir="."):
    dataset_dir = os.path.join(base_dir, "dataset")
    p_teste_fresh = os.path.join(dataset_dir, "teste", "fresh")
    p_teste_rotten = os.path.join(dataset_dir, "teste", "rotten")
    p_trval_fresh = os.path.join(dataset_dir, "treino_validacao", "fresh")
    p_trval_rotten = os.path.join(dataset_dir, "treino_validacao", "rotten")

    pastas_alvo = [p_teste_fresh, p_teste_rotten, p_trval_fresh, p_trval_rotten]

    # Verifica se já está montado com a contagem exata (100, 97, 600, 582)
    if (os.path.exists(p_teste_fresh) and len(os.listdir(p_teste_fresh)) == 100 and
        os.path.exists(p_teste_rotten) and len(os.listdir(p_teste_rotten)) == 97 and
        os.path.exists(p_trval_fresh) and len(os.listdir(p_trval_fresh)) == 600 and
        os.path.exists(p_trval_rotten) and len(os.listdir(p_trval_rotten)) == 582):
        print("[OK] A estrutura 'dataset/' já existe e está 100% completa.")
        return

    print("[!] A preparar a estrutura 'dataset/' com a quantidade exata...")
    for p in pastas_alvo:
        os.makedirs(p, exist_ok=True)
        # Apaga apenas os ficheiros internos, sem remover a pasta (evita WinError 5 no Google Drive)
        for arq in os.listdir(p):
            caminho_arq = os.path.join(p, arq)
            if os.path.isfile(caminho_arq):
                try:
                    os.remove(caminho_arq)
                except PermissionError:
                    pass

    tr_val_fresh, tr_val_rotten = construir_conjuntos_treino_val()

    # Se os arquivos .txt estiverem presentes na pasta, atualiza os conjuntos por eles
    for txt_tr in ["setting1-train_3.txt", "setting1-train.txt"]:
        if os.path.exists(txt_tr):
            tr_val_fresh, tr_val_rotten = set(), set()
            for arq in [txt_tr, txt_tr.replace("train", "val")]:
                if os.path.exists(arq):
                    with open(arq, "r", encoding="utf-8") as f:
                        for line in f:
                            if ";" in line:
                                nome = os.path.basename(line.strip().split(";")[-1].replace("\\", "/"))
                                if ";fresh;" in line:
                                    tr_val_fresh.add(nome)
                                else:
                                    tr_val_rotten.add(nome)
            break

    origem_dados = baixar_freshness44()

    # Indexa todos os arquivos de morango baixados do Kaggle pelo nome (case-insensitive)
    print("[3/3] A indexar e distribuir exatamente as imagens dos cenários...")
    indice_kaggle = {}
    for root, _, files in os.walk(origem_dados):
        for f in files:
            if "strawberry" in f.lower():
                indice_kaggle[f.lower()] = os.path.join(root, f)

    def copiar_conjunto(conjunto_nomes, pasta_destino):
        copiados = 0
        for nome_alvo in conjunto_nomes:
            nome_busca = normalizar_nome_kaggle(nome_alvo).lower()
            if nome_busca in indice_kaggle:
                shutil.copy2(indice_kaggle[nome_busca], os.path.join(pasta_destino, nome_alvo))
                copiados += 1
            else:
                print(f"      [Aviso] Ficheiro não encontrado no zip: {nome_alvo} (buscado como {nome_busca})")
        return copiados

    c_tf = copiar_conjunto(TEST_FRESH, p_teste_fresh)
    c_tr = copiar_conjunto(TEST_ROTTEN, p_teste_rotten)
    c_vf = copiar_conjunto(tr_val_fresh, p_trval_fresh)
    c_vr = copiar_conjunto(tr_val_rotten, p_trval_rotten)

    print("\n================ RESUMO DA MONTAGEM ================")
    print(f"dataset/teste/fresh             : {c_tf} imagens (Esperado: 100)")
    print(f"dataset/teste/rotten            : {c_tr} imagens (Esperado: 97)")
    print(f"dataset/treino_validacao/fresh  : {c_vf} imagens (Esperado: 600)")
    print(f"dataset/treino_validacao/rotten : {c_vr} imagens (Esperado: 582)")
    print("====================================================")

if __name__ == "__main__":
    garantir_dataset()