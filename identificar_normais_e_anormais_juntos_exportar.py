
"""
Identifica instâncias da PTB-XL que apresentam simultaneamente:
    - NORM (normalidade)
    - Pelo menos um diagnóstico anormal (MI, STTC, CD ou HYP)

O algoritmo:
1. Localiza automaticamente os arquivos CSV.
2. Analisa os códigos SCP de cada ECG.
3. Identifica a coexistência de NORM e anormalidade.
4. Exporta as instâncias encontradas para CSV.
5. Exibe um resumo no terminal.

Não modifica os arquivos originais nem reclassifica os ECGs.
"""

import ast
from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PASTA = Path(__file__).resolve().parent

PASTA_SAIDA = PASTA / "resultados"

CLASSES_ANORMAIS = {"MI", "STTC", "CD", "HYP"}


# ============================================================
# 2. LOCALIZAÇÃO AUTOMÁTICA DOS ARQUIVOS
# ============================================================

def localizar_csv(nome):

    arquivos = sorted(
        arquivo
        for arquivo in PASTA.rglob("*.csv")
        if arquivo.name.lower().startswith(nome.lower())
        and "resultados" not in arquivo.relative_to(PASTA).parts
    )

    if not arquivos:
        raise FileNotFoundError(
            f"\nArquivo não encontrado: {nome}\n"
            f"Pasta pesquisada: {PASTA}\n"
            "Coloque o CSV na pasta do projeto ou em uma subpasta."
        )

    if len(arquivos) > 1:
        raise RuntimeError(
            f"\nForam encontrados vários arquivos para {nome}:\n"
            + "\n".join(str(a) for a in arquivos)
            + "\nMantenha apenas a versão desejada ou informe o caminho exato."
        )

    print(f"Arquivo localizado: {arquivos[0]}")

    return arquivos[0]


# ============================================================
# 3. INTERPRETAÇÃO DOS CÓDIGOS SCP
# ============================================================

def interpretar_codigos(valor):

    codigos = ast.literal_eval(valor)

    if not isinstance(codigos, dict):
        raise ValueError(
            f"Formato inválido em scp_codes: {valor}"
        )

    return codigos


# ============================================================
# 4. IDENTIFICAÇÃO DAS INSTÂNCIAS
# ============================================================

def identificar():

    print("=" * 65)
    print("ANÁLISE DE NORMALIDADE E ANORMALIDADE — PTB-XL")
    print("=" * 65)

    arquivo_ecg = localizar_csv("ptbxl_database")
    arquivo_scp = localizar_csv("scp_statements")

    ecgs = pd.read_csv(arquivo_ecg)

    scp = pd.read_csv(arquivo_scp)

    scp = scp.set_index(scp.columns[0])

    registros = []
    desconhecidos = set()

    total_norm = 0

    # --------------------------------------------------------
    # Percorrer todas as instâncias
    # --------------------------------------------------------

    for _, ecg in ecgs.iterrows():

        codigos = interpretar_codigos(ecg["scp_codes"])

        # Verificar presença explícita de NORM
        if "NORM" not in codigos:
            continue

        total_norm += 1

        anormais = []

        # ----------------------------------------------------
        # Procurar diagnósticos anormais na mesma instância
        # ----------------------------------------------------

        for codigo, confianca in codigos.items():

            if codigo not in scp.index:
                desconhecidos.add(codigo)
                continue

            info = scp.loc[codigo]

            diagnostico = info["diagnostic"]
            classe = info["diagnostic_class"]

            if (
                diagnostico == 1
                and classe in CLASSES_ANORMAIS
            ):

                anormais.append({
                    "codigo": codigo,
                    "confianca": confianca,
                    "descricao": info["description"],
                    "superclasse": classe
                })

        # Se não existem diagnósticos anormais,
        # não é uma instância com coexistência
        if not anormais:
            continue

        # ----------------------------------------------------
        # Registrar instância com NORM + anormalidade
        # ----------------------------------------------------

        registros.append({

            "ecg_id": ecg["ecg_id"],

            "patient_id": ecg["patient_id"],

            "filename_lr": ecg["filename_lr"],

            "filename_hr": ecg["filename_hr"],

            "norm_confianca": codigos["NORM"],

            "codigos_originais": str(codigos),

            "codigos_anormais": "; ".join(
                x["codigo"] for x in anormais
            ),

            "descricoes_anormais": "; ".join(
                str(x["descricao"]) for x in anormais
            ),

            "superclasses_anormais": "; ".join(
                sorted({
                    str(x["superclasse"])
                    for x in anormais
                })
            ),

            "confiancas_anormais": "; ".join(
                f'{x["codigo"]}={x["confianca"]}'
                for x in anormais
            ),

            "quantidade_codigos_anormais": len(anormais),

            "report": ecg["report"]

        })

    # ========================================================
    # 5. ORGANIZAÇÃO DOS RESULTADOS
    # ========================================================

    colunas = [
        "ecg_id",
        "patient_id",
        "filename_lr",
        "filename_hr",
        "norm_confianca",
        "codigos_originais",
        "codigos_anormais",
        "descricoes_anormais",
        "superclasses_anormais",
        "confiancas_anormais",
        "quantidade_codigos_anormais",
        "report"
    ]

    resultado = pd.DataFrame(
        registros,
        columns=colunas
    )

    # ========================================================
    # 6. EXPORTAÇÃO AUTOMÁTICA PARA CSV
    # ========================================================

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    destino = (
        PASTA_SAIDA
        / "instancias_NORM_e_anormal_juntos.csv"
    )

    resultado.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 7. RELATÓRIO NO TERMINAL
    # ========================================================

    print("\n" + "=" * 65)
    print("RESULTADOS")
    print("=" * 65)

    print(f"Total de ECGs analisados: {len(ecgs)}")

    print(f"ECGs contendo NORM: {total_norm}")

    print(
        "ECGs contendo NORM + diagnóstico anormal: "
        f"{len(resultado)}"
    )

    print(
        "ECGs contendo NORM sem diagnóstico anormal "
        f"das quatro superclasses: {total_norm - len(resultado)}"
    )

    print(
        "Códigos desconhecidos nos registros com NORM: "
        f"{sorted(desconhecidos)}"
    )

    if not resultado.empty:

        print("\nPrimeiras 10 instâncias identificadas:")

        print(
            resultado[
                [
                    "ecg_id",
                    "codigos_anormais",
                    "superclasses_anormais"
                ]
            ].head(10).to_string(index=False)
        )

        print("\nDistribuição por superclasse:")

        for classe in sorted(CLASSES_ANORMAIS):

            quantidade = resultado[
                "superclasses_anormais"
            ].apply(
                lambda valor: classe in valor.split("; ")
            ).sum()

            print(f"{classe}: {quantidade}")

    print("\n" + "=" * 65)
    print("EXPORTAÇÃO CONCLUÍDA")
    print("=" * 65)

    print(f"CSV salvo em:\n{destino}")

    return resultado


# ============================================================
# 8. EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    resultado = identificar()
