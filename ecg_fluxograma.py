"""Fluxograma F01–F14 a partir do CSV de parâmetros do algoritmo 09.
Uso: python ecg_fluxograma.py --base "CAMINHO/PTB-XL" --ecg-id 21836 \
         --medidas-dir "CAMINHO/ImagensECGSaida" --saida "CAMINHO/resultados"
NÃO é um laudo médico. NUNCA deduz ausência de onda P por falha de detecção.
"""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from referencias_ecg import perfil_ptbxl, avaliar_referencias, numero

DERIVACOES = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
CAMPOS = ['FC_bpm','RR_intervalo_ms','PP_intervalo_ms','PR_intervalo_ms',
          'PR_segmento_ms','ST_segmento_ms','TP_segmento_ms','P_duracao_ms',
          'QRS_duracao_ms','QT_intervalo_ms','QTc_Bazett_ms','QTc_Fridericia_ms',
          'P_amplitude_mV','Q_amplitude_mV','R_amplitude_mV','S_amplitude_mV',
          'T_amplitude_mV','ST_J_mV','ST_J60_mV','QRS_amplitude_dominante_mV',
          'QRS_max_mV','QRS_min_mV','P_pico_s','R_pico_s']

def valores(g, col):
    if col not in g.columns: return np.array([], dtype=float)
    a = pd.to_numeric(g[col], errors='coerce').to_numpy(dtype=float)
    return a[np.isfinite(a)]

def med(g,col):
    x=valores(g,col)
    return round(float(np.median(x)),3) if x.size else None

def cobertura(g,col):
    return round(len(valores(g,col))/len(g),3) if len(g) else 0.0

def registro_medidas(df):
    if not {'derivacao','batimento'}.issubset(df.columns):
        raise ValueError('CSV do 09 precisa das colunas derivacao e batimento')
    dados=df.copy()
    dados['derivacao']=dados['derivacao'].astype(str).str.strip().str.upper().replace({'DI':'I','DII':'II','DIII':'III'})
    # Usar DII para intervalos de ritmo e apresentar medidas individuais de TODAS as derivações.
    grupos={d:dados.loc[dados.derivacao==d].sort_values('batimento') for d in DERIVACOES}
    d2=grupos['II']
    if d2.empty: raise ValueError('CSV sem derivação II; não é possível avaliar o ritmo')
    rr=valores(d2,'RR_intervalo_ms')
    pp=valores(d2,'PP_intervalo_ms')
    pr=valores(d2,'PR_intervalo_ms')
    rr=rr[rr>0]; pp=pp[pp>0]; pr=pr[pr>0]
    fc=60000/np.median(rr) if rr.size else None
    fc=round(float(fc),2) if fc is not None else med(d2,'FC_bpm')
    pp_med=round(float(np.median(pp)),2) if pp.size else None
    # Sem deduzir P ausente: a cobertura é somente evidência de detecção.
    cp=cobertura(d2,'P_pico_s')
    fracao=None
    if 'P_pico_s' in d2 and 'R_pico_s' in d2:
        p=valores(d2,'P_pico_s')
        r=valores(d2,'R_pico_s')
        if p.size and r.size:
            fracao=round(float(np.mean([np.any((rp-p>=0.08)&(rp-p<=0.35)) for rp in r])),3)
    derivacoes={}
    for d,g in grupos.items():
        derivacoes[d]={'batimentos':len(g),'cobertura_p':cobertura(g,'P_amplitude_mV'),
                       'p_polaridade': None,
                       **{k:med(g,k) for k in CAMPOS if k not in ('P_pico_s','R_pico_s')}}
        if 'P_polaridade' in g and len(g):
            pcat=g['P_polaridade'].dropna().astype(str).str.lower()
            pcat=pcat[pcat.isin(['positiva','negativa','bifasica','baixa amplitude'])]
            if len(pcat)>=3 and len(pcat)/len(g)>=.7 and pcat.value_counts(normalize=True).iloc[0]>=.8:
                derivacoes[d]['p_polaridade']=str(pcat.mode().iloc[0])
    return {'frequencia_bpm':fc, 'rr_medio_ms':round(float(np.median(rr)),2) if rr.size else None,
            'rr_cv_percentual':round(float(100*np.std(rr)/np.mean(rr)),2) if rr.size>=3 else None,
            'pp_ms':pp_med, 'frequencia_atrial_bpm':round(float(60000/np.median(pp)),2) if pp.size else None,
            'pr_ms':med(d2,'PR_intervalo_ms'), 'pr_segmento_ms':med(d2,'PR_segmento_ms'),
            'st_segmento_ms':med(d2,'ST_segmento_ms'),'tp_segmento_ms':med(d2,'TP_segmento_ms'),
            'p_duracao_ms':med(d2,'P_duracao_ms'), 'qrs_ms':med(d2,'QRS_duracao_ms'),
            'qt_ms':med(d2,'QT_intervalo_ms'),'qtc_bazett_ms':med(d2,'QTc_Bazett_ms'),
            'qtc_fridericia_ms':med(d2,'QTc_Fridericia_ms'),
            'p_amplitude_ii_mv':med(d2,'P_amplitude_mV'),
            'p_detectada':True if cp>=.8 and len(d2)>=3 else None,
            'cobertura_p':cp,'fracao_qrs_precedido_p':fracao,
            'pr_variacao_ms':round(float(np.max(pr)-np.min(pr)),2) if pr.size>=3 else None,
            'n_batimentos_dii':len(d2), 'derivacoes':derivacoes,
            'cobertura':{c:cobertura(d2,c) for c in CAMPOS if c not in ('P_pico_s','R_pico_s')}}

def avaliacao(codigo, pergunta, condicao, sim, nao, evidencia):
    return {'regra':codigo,'pergunta':pergunta,
            'resposta':'INDETERMINADO' if condicao is None else ('SIM' if condicao else 'NÃO'),
            'resultado':'Não foi possível avaliar com segurança' if condicao is None else sim if condicao else nao,
            'evidencia':evidencia}

def regras(m, perfil, referencias):
    d=m['derivacoes']; out=[]; adulto=perfil.get('idade_anos') is not None and perfil['idade_anos']>=18
    def add(c,p,test,sim,nao,e):out.append(avaliacao(c,p,test,sim,nao,e))
    cv=m['rr_cv_percentual']; fc=m['frequencia_bpm']
    add('F01','RR aproximadamente regular?',cv<10 if cv is not None else None,
        'Baixa variabilidade RR pelo limiar exploratório','Variabilidade RR elevada pelo limiar exploratório',f'CV RR={cv}%; 10% é limiar operacional, não diagnóstico')
    add('F02','FC em faixa usual de repouso adulto (60–100 bpm)?',60<=fc<=100 if adulto and fc is not None else None,
        'FC dentro da faixa de repouso','FC fora da faixa de repouso',f'FC={fc} bpm; idade={perfil.get("idade_anos")}')
    # NÃO inferir eixo pelo pico dominante; estimar polaridade líquida requer integração QRS.
    add('F03','Eixo frontal normal (DI e aVF)?',None,'Quadrante compatível','Eixo não habitual',
        'Amplitude dominante do QRS não equivale à área/amplitude líquida; eixo não implementado')
    add('F04','Onda P identificada de forma consistente?',m['p_detectada'],
        'P identificada em >=80% dos batimentos delineados','Ausência de P confirmada',
        f'Cobertura DII={m["cobertura_p"]:.0%}; detecção incompleta não demonstra ausência de P')
    p2=d['II'].get('p_polaridade'); avr=d['AVR'].get('p_polaridade')
    add('F05','P positiva DII e negativa aVR?',p2=='positiva' and avr=='negativa' if p2 and avr else None,
        'Polaridades compatíveis com origem sinusal (critério parcial)','Polaridades não concordantes',f'P DII={p2}; P aVR={avr}')
    pdur=m['p_duracao_ms']; a=d['II'].get('P_amplitude_mV')
    add('F06','P <120 ms (triagem)?',pdur<120 if adulto and pdur is not None else None,
        'P não alargada por este critério','P alargada por este critério',f'P={pdur} ms; critério simplificado adulto')
    add('F07','|P| DII <=0,25 mV?',abs(a)<=.25 if adulto and a is not None else None,
        'Amplitude dentro deste critério','Amplitude acima deste critério',f'P DII={a} mV')
    associacao=m['fracao_qrs_precedido_p']
    # Associação estimada em DII e ancorada em R; não comprova condução 1:1.
    add('F08','Associação temporal P–R >=90%?',associacao>=.9 if m['p_detectada'] and associacao is not None else None,
        'Associação temporal frequente (não confirma 1:1)','Associação temporal insuficiente',f'Proporção={associacao}; janela P até R 80–350 ms')
    pr=m['pr_ms']
    add('F09','PR 120–200 ms (adulto)?',120<=pr<=200 if adulto and pr is not None else None,
        'PR na faixa habitual','PR fora da faixa habitual',f'PR mediano={pr} ms')
    add('F10','PR constante entre os batimentos?',None,'PR estável','PR variável',
        f'Variação PR observada={m["pr_variacao_ms"]} ms; sem limiar de estabilidade validado; revisar série')
    qrs=m['qrs_ms']
    add('F11','QRS <120 ms (adulto)?',qrs<120 if adulto and qrs is not None else None,
        'QRS estreito pelo limiar','QRS alargado pelo limiar',f'QRS={qrs} ms')
    st={x:d[x].get('ST_J60_mV') for x in DERIVACOES}
    add('F12','Segmento ST sem alteração clínica?',None,'ST sem alteração','Alteração do ST',
        'Valores J+60 por derivação (mV): '+', '.join(f'{k}={v}' for k,v in st.items())+'; requer ponto J, contiguidade e morfologia')
    add('F13','Onda T sem alteração clínica?',None,'T sem alteração','T alterada',
        'Amplitudes/polaridades exportadas; critérios morfológicos e distribuição por derivação ainda não implementados')
    ref=next((x for x in referencias if x['parametro']=='QTc Bazett (ms)'),None)
    test=(ref['avaliacao']=='DENTRO') if ref and ref['avaliacao'] in ('DENTRO','FORA') else None
    add('F14','QTc Bazett dentro do limite exploratório?',test,
        'QTc dentro da faixa de triagem','QTc fora da faixa de triagem: revisão necessária',
        f'Bazett={m["qtc_bazett_ms"]} ms; Fridericia={m["qtc_fridericia_ms"]} ms; referência={ref.get("referencia") if ref else "indisponível"}')
    return out

def renderizar(resultado, destino):
    esc=lambda x:html.escape(str(x))
    css='''body{font:15px system-ui,Arial;margin:25px auto;max-width:1450px;padding:0 18px;color:#243349;background:#f5f7fb}section{background:white;padding:20px;margin:16px 0;border-radius:12px;border:1px solid #e0e7ed}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e1e7ed;vertical-align:top}th{background:#eef2f8;position:sticky;top:0}.scroll{overflow:auto}.avisos{border-left:4px solid #9c7422;padding:12px}.sim{color:#176b48}.nao{color:#9a2938}.ind{color:#755a12}'''
    r=resultado
    linhas=''.join(f'<tr><td>{esc(x["regra"])}</td><td>{esc(x["pergunta"])}</td><td class="{dict(SIM="sim",NÃO="nao",INDETERMINADO="ind")[x["resposta"]]}">{esc(x["resposta"])}</td><td>{esc(x["resultado"])}</td><td>{esc(x["evidencia"])}</td></tr>' for x in r['regras'])
    refs=''.join('<tr>'+''.join('<td>'+esc(x.get(k,''))+'</td>' for k in ['parametro','medido','referencia','avaliacao','fonte','observacao'])+'</tr>' for x in r['comparacao_referencias'])
    d=r['medidas']['derivacoes']
    cols=['P_amplitude_mV','Q_amplitude_mV','R_amplitude_mV','S_amplitude_mV','T_amplitude_mV','PR_intervalo_ms','ST_segmento_ms','TP_segmento_ms','PP_intervalo_ms','RR_intervalo_ms','QT_intervalo_ms','QTc_Bazett_ms','QTc_Fridericia_ms','ST_J_mV','ST_J60_mV']
    header='<tr><th>Derivação</th><th>Batimentos</th>'+''.join('<th>'+esc(c)+'</th>' for c in cols)+'</tr>'
    dr=''.join('<tr><td>'+esc(k)+'</td><td>'+esc(v['batimentos'])+'</td>'+''.join('<td>'+esc(v.get(c) if v.get(c) is not None else '—')+'</td>' for c in cols)+'</tr>' for k,v in d.items())
    geral=''.join('<tr><td>'+esc(k)+'</td><td>'+esc(v if v is not None else '—')+'</td></tr>' for k,v in r['medidas'].items() if k not in ('derivacoes','cobertura'))
    pagina=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECG {esc(r['registro'])} — Fluxograma</title><style>{css}</style></head><body><h1>ECG {esc(r['registro'])} — Fluxograma e parâmetros</h1><p class="avisos">Protótipo de pesquisa, sem validação clínica. SIM/NÃO indicam somente o resultado lógico da regra; não confirmam ou excluem doença. Medianas podem ocultar eventos transitórios. Derivações delineadas independentemente; as linhas de batimentos não são necessariamente sincronizadas entre derivações.</p><section><h2>Perfil PTB-XL</h2><p>Idade: {esc(r['perfil'].get('idade_anos'))} | Sexo: {esc(r['perfil'].get('sexo'))}</p><p>Os rótulos diagnósticos da PTB-XL não foram usados nas regras.</p></section><section><h2>Medidas de ritmo (DII)</h2><div class="scroll"><table><tr><th>Medida</th><th>Valor</th></tr>{geral}</table></div></section><section><h2>Comparação com referências</h2><div class="scroll"><table><tr><th>Parâmetro</th><th>Medido</th><th>Referência</th><th>Estado</th><th>Fonte</th><th>Nota</th></tr>{refs}</table></div></section><section><h2>Regras F01–F14</h2><div class="scroll"><table><tr><th>Regra</th><th>Pergunta</th><th>Resposta</th><th>Resultado</th><th>Evidência</th></tr>{linhas}</table></div></section><section><h2>Medianas por derivação</h2><p>Valores em mV ou ms conforme sufixo da coluna; — significa indisponível.</p><div class="scroll"><table>{header}{dr}</table></div></section></body></html>'''
    destino.write_text(pagina,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description='Fluxograma a partir dos CSVs do algoritmo 09')
    ap.add_argument('--base',required=True,help='Diretório PTB-XL com ptbxl_database.csv')
    ap.add_argument('--ecg-id',type=int,required=True,help='ID no ptbxl_database.csv')
    ap.add_argument('--medidas-dir',default=None,help='Diretório com CSV de medidas gerado pelo script 09; padrão: ../ImagensECGSaida')
    ap.add_argument('--csv',default=None,help='Caminho explícito para o CSV de 12 derivações')
    ap.add_argument('--saida',default='resultados_ecg',help='Pasta dos resultados HTML/JSON')
    args=ap.parse_args()
    base=Path(args.base)
    meta_path=base/'ptbxl_database.csv'
    if not meta_path.is_file():raise FileNotFoundError(f'Metadados ausentes: {meta_path}')
    meta=pd.read_csv(meta_path,low_memory=False)
    if 'ecg_id' not in meta:raise ValueError('ptbxl_database.csv sem ecg_id')
    linhas=meta[pd.to_numeric(meta['ecg_id'],errors='coerce')==args.ecg_id]
    if len(linhas)!=1:raise ValueError(f'ecg_id={args.ecg_id}: esperado 1 registro, encontrados {len(linhas)}')
    linha=linhas.iloc[0]
    registro=f'{args.ecg_id}_hr'
    pasta_medidas=Path(args.medidas_dir) if args.medidas_dir else base.parent/'ImagensECGSaida'
    csv=Path(args.csv) if args.csv else pasta_medidas/f'{registro}_medidas_12_derivacoes.csv'
    if not csv.is_file():raise FileNotFoundError(f'CSV não encontrado: {csv}. Execute primeiro o algoritmo 09 para o mesmo registro.')
    medidas=registro_medidas(pd.read_csv(csv,low_memory=False))
    perfil=perfil_ptbxl(linha.get('age'),linha.get('sex'))
    refs=avaliar_referencias(medidas,perfil)
    resultado={'registro':args.ecg_id,'arquivo_csv':str(csv),'perfil':perfil,
               'comparacao_referencias':refs,'medidas':medidas,
               'regras':regras(medidas,perfil,refs),
               'conclusao':'Triagem de pesquisa não validada; revisão do sinal necessária.'}
    pasta=Path(args.saida);pasta.mkdir(parents=True,exist_ok=True)
    stem=f'ecg_{args.ecg_id}'
    (pasta/f'{stem}.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    renderizar(resultado,pasta/f'{stem}.html')
    print('Arquivos gerados:',pasta/f'{stem}.html',pasta/f'{stem}.json')
    for r in resultado['regras']:print(r['regra'],r['resposta'])

if __name__=='__main__':main()
