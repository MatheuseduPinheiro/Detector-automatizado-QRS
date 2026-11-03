"""Referências para triagem exploratória, SEM interpretação diagnóstica.
Os limiares simplificados abaixo são os que constavam do módulo anterior;
não se presume que sejam critérios completos da SBC.
"""
import math

SBC_2022 = 'SBC, Diretriz de Laudos Eletrocardiográficos, 2022 (verificar seção aplicável)'

def numero(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def perfil_ptbxl(age, sex):
    idade = numero(age)
    if idade is not None and not 0 <= idade <= 120: idade = None
    s = numero(sex)
    sexo = {0:'masculino', 1:'feminino'}.get(int(s)) if s is not None and s in (0, 1) else None
    return {'idade_anos': idade, 'sexo': sexo,
            'codificacao_sexo': 'PTB-XL: 0=masculino; 1=feminino'}

def avaliar_referencias(medidas, perfil):
    """Recebe medidas consolidadas do CSV 09. Não usa laudos PTB-XL como entrada."""
    idade, sexo = perfil.get('idade_anos'), perfil.get('sexo')
    adulto = idade is not None and idade >= 18
    out = []
    def add(nome, valor, ref, cond=None, fonte='Critério exploratório', obs=''):
        out.append({'parametro': nome, 'medido': valor, 'referencia': ref,
                    'avaliacao': 'INDETERMINADO' if cond is None else ('DENTRO' if cond else 'FORA'),
                    'fonte': fonte, 'observacao': obs})
    def valor(chave): return numero(medidas.get(chave))
    fc = valor('frequencia_bpm')
    add('Frequência cardíaca (bpm)', fc, '60–100 bpm (adulto em repouso)',
        60 <= fc <= 100 if adulto and fc is not None else None,
        'Faixa usual de repouso', 'Não classifica ritmo sinusal; contexto clínico desconhecido')
    pr = valor('pr_ms')
    add('PR (ms)', pr, '120–200 ms (adulto)', 120 <= pr <= 200 if adulto and pr is not None else None,
        'Faixa usual de triagem', 'Mediana de estimativas válidas; verificar batimentos e cobertura')
    qrs = valor('qrs_ms')
    add('QRS (ms)', qrs, '<120 ms (adulto)', qrs < 120 if adulto and qrs is not None else None,
        'Faixa usual de triagem', 'Não identifica sozinho bloqueio ou aberrância')
    for chave, rotulo in [('qtc_bazett_ms','QTc Bazett (ms)'),('qtc_fridericia_ms','QTc Fridericia (ms)')]:
        qtc = valor(chave)
        if chave == 'qtc_bazett_ms' and adulto and sexo in ('masculino','feminino'):
            superior = 450 if sexo == 'masculino' else 470
            add(rotulo, qtc, f'340–{superior} ms (triagem adulta, {sexo})',
                340 <= qtc <= superior if qtc is not None else None, SBC_2022,
                'Limites exploratórios herdados; Bazett é sensível à frequência cardíaca; rever QRS largo')
        else:
            add(rotulo, qtc, 'Sem critério implementado para este perfil/fórmula', None,
                SBC_2022, 'Não transportar automaticamente os limites de Bazett para Fridericia ou pediatria')
    for chave,rotulo in [('pp_ms','PP (ms)'),('rr_medio_ms','RR (ms)'),
                         ('tp_segmento_ms','TP (ms)'),('st_segmento_ms','ST duração (ms)'),
                         ('pr_segmento_ms','PR segmento (ms)'),('p_duracao_ms','P duração (ms)'),
                         ('p_amplitude_ii_mv','P amplitude DII (mV)')]:
        add(rotulo, valor(chave), 'Parâmetro descritivo; sem referência universal aplicada', None,
            'Parâmetro calculado pelo módulo 09', 'Duração/amplitude não permitem diagnóstico isolado')
    add('ST por derivação', None, 'Ponto J, contiguidade, morfologia e contexto', None,
        SBC_2022, 'ST J+60 não é intercambiável com limiares clínicos medidos no ponto J')
    add('Ondas Q/R/S e T', None, 'Critérios específicos por derivação e morfologia', None,
        SBC_2022, 'Não implementados como classificação clínica')
    return out
