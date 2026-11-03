"""Relatório interpretativo PRELIMINAR de pesquisa, baseado exclusivamente no JSON do fluxograma.

Não é laudo médico. Não faz inferência de diagnósticos, infarto ou normalidade global.
Uso:
  python 14_Gerador_Relatorio_Interpretativo.py --entrada ".../fluxograma/ecg_21836.json"
  python 14_Gerador_Relatorio_Interpretativo.py --entrada ".../fluxograma/ecg_21836.json" --saida ".../fluxograma"

Regras de proveniência: cada afirmação se ancora em um campo medido, comparação ou
regra pré-existente. SIM e NÃO indicam apenas satisfação lógica de um teste.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path

GRUPOS = [
    ("Frequência e ritmo", ("frequencia_bpm", "rr_medio_ms", "rr_cv_percentual", "pp_ms", "frequencia_atrial_bpm", "fracao_qrs_precedido_p")),
    ("Ondas e condução", ("p_duracao_ms", "p_amplitude_ii_mv", "pr_ms", "pr_variacao_ms", "qrs_ms")),
    ("Repolarização", ("qt_ms", "qtc_bazett_ms", "qtc_fridericia_ms", "st_segmento_ms", "tp_segmento_ms")),
]
ROTULOS = {
    "frequencia_bpm": ("Frequência ventricular estimada", "bpm"),
    "rr_medio_ms": ("Mediana RR", "ms"),
    "rr_cv_percentual": ("Coeficiente de variação RR", "%"),
    "pp_ms": ("Mediana PP", "ms"),
    "frequencia_atrial_bpm": ("Frequência atrial estimada", "bpm"),
    "fracao_qrs_precedido_p": ("Fração de QRS com P precedente estimada", ""),
    "p_duracao_ms": ("Duração da onda P em DII", "ms"),
    "p_amplitude_ii_mv": ("Amplitude de P em DII", "mV"),
    "pr_ms": ("Intervalo PR em DII", "ms"),
    "pr_variacao_ms": ("Amplitude da variação PR em DII", "ms"),
    "qrs_ms": ("Duração do QRS em DII", "ms"),
    "qt_ms": ("Intervalo QT em DII", "ms"),
    "qtc_bazett_ms": ("QTc Bazett em DII", "ms"),
    "qtc_fridericia_ms": ("QTc Fridericia em DII", "ms"),
    "st_segmento_ms": ("Duração do segmento ST em DII", "ms"),
    "tp_segmento_ms": ("Duração do segmento TP em DII", "ms"),
}
DERIVACOES = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
COLUNAS_DERIVACOES = (
    ("P_amplitude_mV", "P (mV)"), ("Q_amplitude_mV", "Q (mV)"),
    ("R_amplitude_mV", "R (mV)"), ("S_amplitude_mV", "S (mV)"),
    ("T_amplitude_mV", "T (mV)"), ("PR_intervalo_ms", "PR (ms)"),
    ("QRS_duracao_ms", "QRS (ms)"), ("QT_intervalo_ms", "QT (ms)"),
    ("ST_J_mV", "ST no J (mV)"), ("ST_J60_mV", "ST em J+60 (mV)"),
)

def numero(valor):
    if isinstance(valor, bool):
        return None
    try:
        n = float(valor)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError, OverflowError):
        return None


def fmt(valor, unidade=""):
    n = numero(valor)
    if n is None:
        return "Não disponível"
    v = f"{n:,.3f}".rstrip('0').rstrip('.')
    v = v.replace(',', '_').replace('.', ',').replace('_', '.')
    return v + (" " + unidade if unidade else "")


def validar(dados):
    if not isinstance(dados, dict) or not isinstance(dados.get('medidas'), dict) or not isinstance(dados.get('regras'), list):
        raise ValueError('JSON inválido: esperados objetos medidas e lista regras do ecg_fluxograma.py')
    if not isinstance(dados.get('comparacao_referencias', []), list):
        raise ValueError('JSON inválido: comparacao_referencias precisa ser lista')
    for regra in dados['regras']:
        if not isinstance(regra, dict) or not {'regra', 'resposta', 'pergunta'} <= regra.keys():
            raise ValueError('Regra do JSON incompleta; campos regra, resposta e pergunta são obrigatórios')


def analisar(dados):
    validar(dados)
    medidas = dados['medidas']
    regras = dados['regras']
    refs = dados.get('comparacao_referencias', [])
    registro = dados.get('registro', 'não informado')
    identificadas, nao_satisfeitas, indeterminadas = [], [], []
    for regra in regras:
        estado = str(regra.get('resposta', '')).upper().strip()
        item = {
            'codigo': str(regra['regra']),
            'pergunta': str(regra['pergunta']),
            'estado': estado,
            'resultado_regra': str(regra.get('resultado') or ''),
            'evidencia_original': str(regra.get('evidencia') or ''),
            'origem': 'ecg_fluxograma.py / regras',
        }
        if estado == 'SIM':
            identificadas.append(item)
        elif estado == 'NÃO':
            nao_satisfeitas.append(item)
        else:
            indeterminadas.append(item)
    comparacoes = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        comparacoes.append({k: ref.get(k) for k in ('parametro','medido','referencia','avaliacao','fonte','observacao')})
    medidas_organizadas = [
        {'grupo': nome, 'itens': [
            {'campo': k, 'rotulo': ROTULOS[k][0], 'valor': numero(medidas.get(k)), 'unidade': ROTULOS[k][1]}
            for k in campos]}
        for nome, campos in GRUPOS]
    derivacoes = medidas.get('derivacoes', {})
    if not isinstance(derivacoes, dict):
        derivacoes = {}
    res_derivacoes = {}
    for nome in DERIVACOES:
        d = derivacoes.get(nome, {})
        if not isinstance(d, dict):
            d = {}
        res_derivacoes[nome] = {
            'batimentos': d.get('batimentos'),
            'cobertura_p': numero(d.get('cobertura_p')),
            'p_polaridade': d.get('p_polaridade'),
            'medidas': {c: numero(d.get(c)) for c, _ in COLUNAS_DERIVACOES}
        }
    alertas = [
        'Este é um relatório interpretativo preliminar de pesquisa, não um laudo médico nem uma classificação diagnóstica validada.',
        'SIM/NÃO registram apenas o resultado lógico das regras F01–F14; NÃO não comprova doença, e SIM não comprova normalidade clínica.',
        'Resultados são medianas e podem ocultar eventos transitórios; derivações foram delineadas de modo independente e não necessariamente sincronizado.',
        'Uma onda P não detectada não significa ausência comprovada de P. O detector de P utiliza informação ancorada em picos R.',
        'Ritmo sinusal definitivo, eixo elétrico, diagnósticos de arritmias, bloqueios, isquemia e infarto não são estabelecidos por este módulo.',
        'Valores de ST em J e J+60 não devem ser tratados como equivalentes; avaliação clínica exige critérios por derivação e contexto.',
        'Comparações usam referências simplificadas do módulo existente e dependem de validação e revisão médica.'
    ]
    csv = dados.get('arquivo_csv')
    return {
        'registro': registro,
        'gerado_em_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'tipo': 'Relatório interpretativo preliminar de pesquisa — não é laudo clínico',
        'fonte': {'arquivo_fluxograma_json': None, 'arquivo_csv_original': csv},
        'perfil': dados.get('perfil', {}),
        'medidas_organizadas': medidas_organizadas,
        'derivacoes': res_derivacoes,
        'comparacoes': comparacoes,
        'evidencias_regras_satisfeitas': identificadas,
        'evidencias_regras_nao_satisfeitas': nao_satisfeitas,
        'regras_indeterminadas': indeterminadas,
        'contagem_regras': {'satisfeitas':len(identificadas),'nao_satisfeitas':len(nao_satisfeitas),'indeterminadas':len(indeterminadas)},
        'sintese': (
            f'Foram avaliadas {len(regras)} regras do fluxograma: {len(identificadas)} satisfeita(s), '
            f'{len(nao_satisfeitas)} não satisfeita(s) e {len(indeterminadas)} indeterminada(s). '
            'Os estados são resultados de testes exploratórios, não diagnósticos. '
            'Não é possível estabelecer laudo clínico conclusivo apenas a partir dessas evidências.'
        ),
        'hipoteses_clinicas_confirmadas': [],
        'limitacoes': alertas,
    }


def tabela(cabecalhos, linhas):
    th = ''.join(f'<th>{escape(str(c))}</th>' for c in cabecalhos)
    td = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in linha) + '</tr>' for linha in linhas)
    if not linhas:
        td = f'<tr><td colspan="{len(cabecalhos)}">Sem dados ou critérios disponíveis</td></tr>'
    return '<div class="scroll"><table><thead><tr>' + th + '</tr></thead><tbody>' + td + '</tbody></table></div>'


def criar_html(relatorio):
    esc = lambda x: escape(str(x))
    perfil=relatorio.get('perfil') or {}
    secoes_medidas = ''
    for grupo in relatorio['medidas_organizadas']:
        secoes_medidas += f'<h3>{esc(grupo["grupo"])}</h3>'
        secoes_medidas += tabela(['Parâmetro','Valor'], [[x['rotulo'],fmt(x['valor'],x['unidade'])] for x in grupo['itens']])
    refs = relatorio['comparacoes']
    tab_refs = tabela(['Parâmetro','Medido','Referência','Avaliação','Fonte','Observação'], [[str(x.get(k) if x.get(k) is not None else '—') for k in ('parametro','medido','referencia','avaliacao','fonte','observacao')] for x in refs])
    def tabela_regras(registros):
        return tabela(['ID','Critério','Estado','Resultado do algoritmo','Evidência original'], [
            [r['codigo'],r['pergunta'],r['estado'],r['resultado_regra'],r['evidencia_original']] for r in registros])
    derivacoes = []
    for d,info in relatorio['derivacoes'].items():
        derivacoes.append([d,info['batimentos'] if info['batimentos'] is not None else '—',
                           info['p_polaridade'] or '—']+
                          [fmt(info['medidas'][campo]) if info['medidas'][campo] is not None else '—' for campo,_ in COLUNAS_DERIVACOES])
    tab_der=tabela(['Derivação','Batimentos','Polaridade P']+[rotulo for _,rotulo in COLUNAS_DERIVACOES],derivacoes)
    avisos=''.join(f'<li>{esc(x)}</li>' for x in relatorio['limitacoes'])
    c=relatorio['contagem_regras']
    css='''body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f3f6fa;color:#25364a;max-width:1450px;margin:28px auto;padding:0 18px;line-height:1.5}section{background:white;border:1px solid #dfe6ee;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:1.7rem}h2{font-size:1.25rem}h3{font-size:1.08rem;margin-top:25px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid #e1e7ef;text-align:left;vertical-align:top}th{background:#edf2f8;white-space:nowrap}.tag{display:inline-block;padding:5px 9px;border-radius:6px;background:#e8edf3;margin-right:8px}.note{border-left:4px solid #aa8330;padding:10px 14px;background:#fffbf0}.small{font-size:13px;color:#526376}ul{padding-left:20px}@media print{body{background:white}section{break-inside:avoid}}'''
    return f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECG {esc(relatorio['registro'])} — Relatório interpretativo preliminar</title><style>{css}</style></head><body>
<h1>ECG {esc(relatorio['registro'])} — Relatório interpretativo preliminar</h1>
<p class="note"><strong>Uso exclusivo em pesquisa.</strong> Não é laudo médico, não confirma nem exclui doenças e não deve ser usado para decisões clínicas.</p>
<section><h2>1. Identificação e proveniência</h2><p>Registro: <strong>{esc(relatorio['registro'])}</strong> · Idade: {esc(perfil.get('idade_anos', 'não informada'))} anos · Sexo PTB-XL: {esc(perfil.get('sexo','não informado'))}</p><p class="small">Fonte das medidas: {esc(relatorio['fonte'].get('arquivo_csv_original') or 'não informada')}<br>Fonte das regras: {esc(relatorio['fonte'].get('arquivo_fluxograma_json') or 'não informada')}</p></section>
<section><h2>2. Síntese interpretativa</h2><p>{esc(relatorio['sintese'])}</p><span class="tag">Satisfeitas: {c['satisfeitas']}</span><span class="tag">Não satisfeitas: {c['nao_satisfeitas']}</span><span class="tag">Indeterminadas: {c['indeterminadas']}</span><p><strong>Hipóteses clínicas confirmadas automaticamente:</strong> nenhuma. Conclusão clínica depende de critérios adicionais e revisão médica.</p></section>
<section><h2>3. Medidas disponíveis (principalmente DII)</h2>{secoes_medidas}</section>
<section><h2>4. Comparação com referências do algoritmo</h2>{tab_refs}<p class="small">Os limites de triagem são os disponibilizados no módulo de referência; sua aplicação exige contexto e validação.</p></section>
<section><h2>5. Critérios satisfeitos</h2>{tabela_regras(relatorio['evidencias_regras_satisfeitas'])}</section>
<section><h2>6. Critérios não satisfeitos — exigem revisão</h2>{tabela_regras(relatorio['evidencias_regras_nao_satisfeitas'])}</section>
<section><h2>7. Critérios indeterminados</h2>{tabela_regras(relatorio['regras_indeterminadas'])}</section>
<section><h2>8. Medidas por derivação</h2>{tab_der}<p class="small">Medianas independentes por derivação; a mediana de uma coluna não necessariamente representa o mesmo batimento da mediana de outra.</p></section>
<section><h2>9. Limitações e validação necessária</h2><ul>{avisos}</ul></section>
</body></html>'''


def main():
    ap = argparse.ArgumentParser(description='Gera relatório interpretativo preliminar a partir do JSON do fluxograma')
    ap.add_argument('--entrada', required=True, help='Arquivo ecg_ID.json do ecg_fluxograma.py')
    ap.add_argument('--saida', default=None, help='Pasta de saída; padrão: mesma pasta do JSON de entrada')
    args = ap.parse_args()
    entrada = Path(args.entrada)
    if not entrada.is_file():
        ap.error(f'Arquivo JSON não encontrado: {entrada}. Execute primeiro ecg_fluxograma.py.')
    dados = json.loads(entrada.read_text(encoding='utf-8-sig'))
    rel = analisar(dados)
    rel['fonte']['arquivo_fluxograma_json'] = str(entrada.resolve())
    saida = Path(args.saida) if args.saida else entrada.parent
    saida.mkdir(parents=True, exist_ok=True)
    stem = f"ecg_{rel['registro']}_relatorio_interpretativo"
    destino_html = saida / (stem + '.html')
    destino_json = saida / (stem + '.json')
    destino_html.write_text(criar_html(rel), encoding='utf-8')
    destino_json.write_text(json.dumps(rel, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'HTML: {destino_html}\nJSON: {destino_json}')
    print(rel['sintese'])

if __name__ == '__main__':
    main()
