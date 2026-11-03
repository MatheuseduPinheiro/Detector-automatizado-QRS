import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks
from tqdm.auto import tqdm

# ============================================================
# 1. CONFIGURAÇÕES: TODAS AS AMOSTRAS
# ============================================================
PASTA = Path.cwd()
DESTINO = PASTA / 'Recortes-QRS'
DERIVACOES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
CLASSES = ['0', '1.1_MI', '1.2_STTC', '1.3_CD',
           '1.4_HYP', '1.5_FORM', '1.6_RHYTHM']
LIMIAR_QRS_MS = 120
SOBRESCREVER = False  # True apenas para reprocessar pastas completas
JANELA_ANTES_MS = 120
JANELA_DEPOIS_MS = 160

# ============================================================
# 2. ENCONTRAR ORIGEM SEM MODIFICÁ-LA
# ============================================================
candidatas = [p for p in PASTA.rglob('*')
              if p.is_dir() and p.name.casefold() == 'ecg_organizados'
              and DESTINO.resolve() not in p.resolve().parents]
if len(candidatas) != 1:
    raise RuntimeError(f'Esperada uma pasta ECG_Organizados; encontradas: {candidatas}')
ORIGEM = candidatas[0]
if ORIGEM.resolve() == DESTINO.resolve() or ORIGEM.resolve() in DESTINO.resolve().parents:
    raise RuntimeError('Origem e destino não podem coincidir ou se sobrepor.')

# ============================================================
# 3. EXCLUSÃO DAS 445 INSTÂNCIAS
# ============================================================
arquivos_exclusao = sorted((PASTA / 'resultados').glob('instancias_NORM_e_anormal_juntos*.csv'))
if len(arquivos_exclusao) != 1:
    raise RuntimeError('É necessário exatamente um CSV de exclusões em resultados/.')
exclusoes = pd.read_csv(arquivos_exclusao[0])
if 'ecg_id' not in exclusoes:
    raise ValueError('CSV de exclusões sem coluna ecg_id.')
ids_excluidos = set(pd.to_numeric(exclusoes.ecg_id, errors='raise').astype(int))
if len(ids_excluidos) != 445:
    raise ValueError(f'Esperados 445 IDs únicos excluídos; encontrados {len(ids_excluidos)}.')

# ============================================================
# 4. DETECÇÃO EXPLORATÓRIA DE QRS
# Se as funções já foram definidas no notebook, preservá-las.
# ============================================================
if not callable(globals().get('detectar_qrs')):
    def detectar_qrs(sinal, fs):
        sinal = np.asarray(sinal, dtype=float)
        if len(sinal) < 30 or not np.isfinite(sinal).all():
            return [], None, 'SINAL_INVALIDO'
        filtrado = sosfiltfilt(butter(3, [5, 25], btype='bandpass', fs=fs, output='sos'), sinal)
        energia = filtrado ** 2
        janela = max(1, int(.08 * fs))
        energia_suave = np.convolve(energia, np.ones(janela) / janela, mode='same')
        limiar = np.median(energia_suave) + .5 * np.std(energia_suave)
        candidatos, _ = find_peaks(
            energia_suave, distance=max(1, int(.25 * fs)), height=limiar,
            prominence=max(np.std(energia_suave) * .25, np.finfo(float).eps))
        picos = []
        for candidato in candidatos:
            margem = int(.06 * fs)
            a, b = max(0, candidato - margem), min(len(filtrado), candidato + margem + 1)
            if a < b:
                picos.append(int(a + np.argmax(np.abs(filtrado[a:b]))))
        selecionados = []
        for pico in sorted(set(picos)):
            if not selecionados or pico - selecionados[-1] >= int(.20 * fs):
                selecionados.append(pico)
            elif abs(filtrado[pico]) > abs(filtrado[selecionados[-1]]):
                selecionados[-1] = pico
        return selecionados, filtrado, ('DETECCAO_PRELIMINAR' if selecionados else 'PICO_R_NAO_DETECTADO')

if not callable(globals().get('estimar_limites')):
    def estimar_limites(sinal_filtrado, pico, fs):
        antes, depois = int(JANELA_ANTES_MS * fs / 1000), int(JANELA_DEPOIS_MS * fs / 1000)
        a, b = max(0, pico - antes), min(len(sinal_filtrado) - 1, pico + depois)
        if a == 0 or b == len(sinal_filtrado) - 1:
            return None, None, 'QRS_PROXIMO_DA_BORDA'
        amplitude = np.abs(sinal_filtrado)
        referencia = amplitude[pico]
        if referencia <= 0:
            return None, None, 'AMPLITUDE_INSUFICIENTE'
        limiar = .10 * referencia
        inicio = fim = int(pico)
        while inicio > a and amplitude[inicio] > limiar:
            inicio -= 1
        while fim < b and amplitude[fim] > limiar:
            fim += 1
        if inicio >= pico or fim <= pico:
            return None, None, 'LIMITES_INDEFINIDOS'
        if inicio == a or fim == b:
            return None, None, 'LIMITES_NAO_CONFIAVEIS'
        return inicio, fim, 'LIMITES_ESTIMADOS'

# ============================================================
# 5. IDENTIFICAR AMOSTRAS E PRESERVAR CAMINHOS ORIGINAIS
# ============================================================
def classe_do_caminho(rel):
    partes = rel.parts
    if len(partes) >= 3 and partes[0] == '0' and partes[1].startswith('registros'):
        return '0'
    if len(partes) >= 4 and partes[0] == '1' and partes[1] in CLASSES[1:] and partes[2].startswith('registros'):
        return partes[1]
    return None

registros = []
ignorados = {'excluidos': 0, 'sem_dat': 0, 'fora_estrutura': 0, 'nome_invalido': 0}
for hea in sorted(ORIGEM.rglob('*.hea')):
    rel = hea.parent.relative_to(ORIGEM)
    classe = classe_do_caminho(rel)
    if classe is None:
        ignorados['fora_estrutura'] += 1
        continue
    if not rel.parts[-1].isdigit():
        ignorados['nome_invalido'] += 1
        continue
    ecg_id = int(rel.parts[-1])
    # O ID na pasta deve corresponder ao registro WFDB
    m = re.fullmatch(r'(\d+)(?:_hr|_lr)?', hea.stem)
    if m is None or int(m.group(1)) != ecg_id:
        ignorados['nome_invalido'] += 1
        continue
    if ecg_id in ids_excluidos:
        ignorados['excluidos'] += 1
        continue
    dat = hea.with_suffix('.dat')
    if not dat.is_file():
        ignorados['sem_dat'] += 1
        continue
    registros.append({'id': ecg_id, 'classe': classe, 'hea': hea,
                      'dat': dat, 'rel': rel})

# Evita sobrescrever o mesmo destino caso haja mais de um registro na pasta.
destinos = [str(item['rel']).casefold() for item in registros]
if len(destinos) != len(set(destinos)):
    raise RuntimeError('Há mais de um par WFDB para a mesma pasta de amostra; revise antes de processar.')

print('Origem (somente leitura):', ORIGEM)
print('Destino:', DESTINO)
print('Registros elegíveis:', len(registros))
print('Ignorados:', ignorados)
if not registros:
    raise RuntimeError('Nenhuma amostra elegível encontrada na estrutura esperada.')

# ============================================================
# 6. PROCESSAR E SALVAR CADA AMOSTRA INDIVIDUALMENTE
# ============================================================
COLUNAS_QRS = ['ecg_id', 'derivacao', 'batimento', 'amostra_qrs',
               'amostra_original', 'tempo_relativo_ms', 'amplitude_mV',
               'duracao_qrs_ms', 'qrs_alargado']
COLUNAS_META = ['ecg_id', 'classe', 'derivacao', 'batimento', 'pico_r',
                'inicio_qrs', 'fim_qrs', 'duracao_ms', 'qrs_alargado',
                'situacao', 'motivo', 'fs_hz']

def processar_amostra(item):
    destino = DESTINO / item['rel']
    esperados = [item['dat'].name, item['hea'].name, 'instancias.csv'] + [f'{d}_QRS.csv' for d in DERIVACOES]
    if destino.exists() and not SOBRESCREVER:
        completos = all((destino / nome).is_file() for nome in esperados)
        return {'ecg_id': item['id'], 'classe': item['classe'],
                'situacao': 'JA_PROCESSADO' if completos else 'DESTINO_INCOMPLETO', 'qrs_extraidos': 0}

    registro = wfdb.rdrecord(str(item['hea'].with_suffix('')))
    fs = float(registro.fs)
    if registro.p_signal is None or fs <= 0:
        raise ValueError('Sinal físico ausente ou frequência inválida.')
    metadados = []
    por_derivacao = {}
    total = 0
    for derivacao in DERIVACOES:
        linhas = []
        def registrar(batimento=None, pico=None, inicio=None, fim=None,
                     duracao=None, alargado=None, situacao='REVISAR', motivo=''):
            metadados.append({'ecg_id': item['id'], 'classe': item['classe'],
                              'derivacao': derivacao, 'batimento': batimento,
                              'pico_r': pico, 'inicio_qrs': inicio, 'fim_qrs': fim,
                              'duracao_ms': duracao, 'qrs_alargado': alargado,
                              'situacao': situacao, 'motivo': motivo, 'fs_hz': fs})
        if derivacao not in registro.sig_name:
            registrar(motivo='DERIVACAO_AUSENTE')
        else:
            sinal = registro.p_signal[:, registro.sig_name.index(derivacao)]
            picos, filtrado, status = detectar_qrs(sinal, fs)
            if not picos:
                registrar(motivo=status)
            else:
                for numero, pico in enumerate(picos, 1):
                    inicio, fim, motivo = estimar_limites(filtrado, pico, fs)
                    if inicio is None or fim is None or not (0 <= inicio < pico < fim < len(sinal)):
                        registrar(batimento=numero, pico=pico, motivo=motivo)
                        continue
                    duracao = (fim - inicio) / fs * 1000
                    alargado = bool(duracao >= LIMIAR_QRS_MS)
                    for deslocamento, amplitude in enumerate(sinal[inicio:fim + 1]):
                        linhas.append({'ecg_id': item['id'], 'derivacao': derivacao,
                                      'batimento': numero, 'amostra_qrs': deslocamento,
                                      'amostra_original': inicio + deslocamento,
                                      'tempo_relativo_ms': deslocamento / fs * 1000,
                                      'amplitude_mV': amplitude, 'duracao_qrs_ms': duracao,
                                      'qrs_alargado': alargado})
                    registrar(numero, pico, inicio, fim, round(duracao, 2), alargado,
                              'ESTIMATIVA_EXPLORATORIA', 'REQUER_VALIDACAO')
                    total += 1
        por_derivacao[derivacao] = linhas

    # Montagem em diretório temporário: só publica pasta completa.
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.qrs_tmp_', dir=destino.parent) as tmp:
        temp = Path(tmp)
        shutil.copy2(item['dat'], temp / item['dat'].name)
        shutil.copy2(item['hea'], temp / item['hea'].name)
        for derivacao in DERIVACOES:
            pd.DataFrame(por_derivacao[derivacao], columns=COLUNAS_QRS).to_csv(
                temp / f'{derivacao}_QRS.csv', index=False)
        pd.DataFrame(metadados, columns=COLUNAS_META).to_csv(temp / 'instancias.csv', index=False)
        if destino.exists():
            if not SOBRESCREVER:
                return {'ecg_id': item['id'], 'classe': item['classe'],
                        'situacao': 'DESTINO_JA_EXISTE', 'qrs_extraidos': 0}
            shutil.rmtree(destino)  # apenas pasta de saída da amostra
        shutil.move(str(temp), str(destino))
    return {'ecg_id': item['id'], 'classe': item['classe'],
            'situacao': 'PROCESSADO', 'qrs_extraidos': total}

# ============================================================
# 7. EXECUTAR TODAS AS CLASSES, SEM AMOSTRAGEM
# ============================================================
DESTINO.mkdir(parents=True, exist_ok=True)
resumo = []
for item in tqdm(registros, desc='Extração QRS — todas as amostras'):
    try:
        resumo.append(processar_amostra(item))
    except Exception as exc:
        resumo.append({'ecg_id': item['id'], 'classe': item['classe'],
                       'situacao': 'ERRO', 'qrs_extraidos': 0, 'erro': str(exc)})
        print(f"Erro: classe {item['classe']}, ECG {item['id']}: {exc}")

# ============================================================
# 8. RESUMO POR CLASSE
# ============================================================
df_resumo = pd.DataFrame(resumo)
if not df_resumo.empty:
    tabela = (df_resumo.groupby(['classe', 'situacao'], as_index=False)
              .agg(amostras=('ecg_id', 'count'), qrs_extraidos=('qrs_extraidos', 'sum')))
    print('\nRESUMO POR CLASSE E SITUAÇÃO:')
    print(tabela.to_string(index=False))
    df_resumo.to_csv(DESTINO / 'resumo_processamento.csv', index=False)
print('\nECG_Organizados preservada:', ORIGEM)
print('Recortes salvos em:', DESTINO)
