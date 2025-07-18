import os
import json
import time
import pathlib
from google.generativeai import GenerativeModel
import google.generativeai as genai
#from google import genai
#from google.genai import types
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# Caminho do diretório coms os processos
DIRETORIO_PROCESSOS = pathlib.Path('data')

# Nome da chave que conterá a análise FIRAC no JSON
CAMPO_FIRAC = 'analise_firac'

# Modelo Gemini
MODELO = 'models/gemini-2.5-pro'

# Prompt Base para análise FIRAC
PROMPT_TEMPLATE = """
# FUNÇÃO E OBJETIVO
Atue como Procurador Municipal Especializado com expertise em análise processual estratégica. 
Realize análise FIRAC completa e estruturada do processo judicial para subsidiar tomada de decisão da Procuradoria Municipal e 
alimentar sistema de ciência de dados jurídicos. Seja extenso, objetivo e preciso na análise.
# ESTRUTURA: JSON SCHEMA COMPLETO
'''json
{
  "metadados": {
    "data_analise": "string",
    "versao_analise": "1.0",
    "analista_responsavel": "ia_gemini",
    "tempo_processamento": "string",
    "llm_tokens": "string",    
    "custo_estimado": "string"
  },
  "identificacao_processual": {
    "processo": "string",
    "juizo": "string",
    "vara": "string",
    "municipio_parte": "ativo|passivo|interessado|litisconsorte",
    "posicao_municipal": "requerente|requerido|terceiro_interessado|assistente",
    "fase_processual": "conhecimento|execução|recursal|arquivado|cumprimento_sentenca",
    "status_julgamento": "em_andamento|julgado_procedente|julgado_improcedente|parcialmente_procedente|extinto_sem_merito|extinto_com_merito",
    "grau_jurisdicao": "primeira_instancia|segunda_instancia|tribunais_superiores",
    "data_distribuicao": "string",
    "data_ultima_movimentacao": "string"
  },
  "classificacao_especializada": {
    "area_responsavel": "fiscal|administrativa|servidor_publico|ambiental|judicial|patrimonial|saude|divida_ativa|previdenciaria|trabalhista|contratual|regulatoria|trabalhista",
    "subarea": "string",
    "materia_principal": "string",
    "materias_conexas": ["string"]
  },
  "analise_urgencia": {
    "criterios_urgencia": {
      "impacto_financeiro_direto": "baixo|medio|alto|critico",
      "risco_acoes_identicas": "baixo|medio|alto|critico",
      "repercussao_politica": "baixa|media|alta|critica",
      "impacto_operacional": "baixo|medio|alto|paralisante"
    },
    "justificativa_urgencia": "string",
    "prazo_limite_acao": "string"
  },
  "fatos_juridicos": {
    "resumo_executivo": "string",
    "cronologia_detalhada": [
      {
        "data": "string",
        "evento": "string",
        "fonte_documento": "string",
        "relevancia_juridica": "baixa|media|alta|essencial",
        "impacto_caso": "positivo|neutro|negativo",
        "observacoes": "string"
      }
    ],
    "fatos_incontroversos": [
      {
        "fato": "string",
        "fundamento": "string",
        "implicacao_juridica": "string"
      }
    ],
    "fatos_controvertidos": [
      {
        "fato": "string",
        "posicao_municipio": "string",
        "posicao_adversa": "string",
        "onus_probatorio": "municipio|adversario|compartilhado",
        "dificuldade_comprovacao": "baixa|media|alta",
        "como_impugnar": "string"
      }
    ],
    "questoes_preliminares": [
      {
        "questao": "string",
        "posicionamento_sugerido": "string",
        "fundamentacao": "string"
      }
    ]
  },
  "questoes_juridicas": {
    "questao_principal": {
      "descricao": "string",
      "natureza": "direito_material|direito_processual|mista"
    },
    "questoes_secundarias": [
      {
        "descricao": "string",
        "relacao_principal": "string",
        "impacto_resultado": "determinante|relevante|acessorio"
      }
    ],
    "teses_municipio": [
      {
        "tese": "string",
        "fundamentacao_legal": "string",
        "solidez_juridica": "fragil|razoavel|solida|consolidada",
        "precedentes_favoraveis": ["string"]
      }
    ],
    "teses_adversarias": [
      {
        "tese": "string",
        "contra_argumentacao": "string",
        "vulnerabilidades_identificadas": ["string"],
        "precedentes_desfavoraveis": ["string"]
      }
    ]
  },
  "direito_aplicavel": {
    "normas_primarias": [
      {
        "dispositivo": "string",
        "texto_relevante": "string",
        "interpretacao": "string",
        "favorabilidade_municipio": "muito_favoravel|favoravel|neutro|desfavoravel|muito_desfavoravel"
      }
    ],
    "normas_secundarias": ["string"],
    "principios_aplicaveis": ["string"],
    "regulamentacao_municipal": ["string"]
  },
  "orientacao_jurisprudencial": {
    "palavras_chave_busca": ["string"],
    "temas_jurisprudenciais": ["string"],
    "sumulas_aplicaveis": ["string"],
    "precedentes_stf": ["string"],
    "precedentes_stj": ["string"],
    "query_sugerida_jurisprudencia": "string",
    "estrategia_pesquisa": "string"
  },
  "analise_riscos_detalhada": {
    "probabilidade_exito": {
      "percentual_estimado": "string",
      "classificacao": "muito_baixa|baixa|media|alta|muito_alta",
      "fatores_positivos": ["string"],
      "fatores_negativos": ["string"],
      "cenarios": {
        "otimista": "string",
        "realista": "string", 
        "pessimista": "string"
      }
    },
    "impacto_financeiro_detalhado": {
      "valor_causa": "string",
      "valor_estimado_condenacao": "string",
      "categoria_valor": "ate_10k|10k_50k|50k_200k|200k_1mi|1mi_5mi|acima_5mi",
      "custos_processuais": "string",
      "honorarios_estimados": "string",
      "multas_astreintes": "string",
      "impacto_orcamentario": "insignificante|baixo|medio|alto|critico",
      "fonte_recursos": "string"
    },
    "risco_multiplicacao": {
      "potencial_acoes_similares": "baixo|medio|alto|explosivo",
      "estimativa_numerica": "string",
      "valor_total_potencial": "string",
      "medidas_preventivas": ["string"]
    },
  },
  "resultado_julgamento": {
    "foi_julgado": "sim|nao",
    "data_julgamento": "string",
    "resultado_detalhado": "procedente|improcedente|parcialmente_procedente|extinto_sem_merito|extinto_com_merito|nao_aplicavel",
    "tese_vencedora": {
      "descricao": "string",
      "fundamentos_decisao": ["string"],
      "ratio_decidendi": "string",
      "obiter_dicta": ["string"]
    },
    "onus_municipio": [
      {
        "tipo_obrigacao": "obrigacao_fazer|obrigacao_nao_fazer|obrigacao_pagar|obrigacao_entregar|declaratoria",
        "descricao_detalhada": "string",
        "prazo_cumprimento": "string",
        "valor_monetario": "string",
        "periodicidade": "unica|mensal|anual|continuada",
        "indexador": "string",
        "multa_descumprimento": "string"
      }
    ],
    "recursos_possiveis": {
      "recurso_cabivel": "apelacao|embargos|recurso_especial|recurso_extraordinario|nao_aplicavel",
      "prazo_recursal": "string",
      "chances_reforma": "baixa|media|alta",
      "estrategia_recursal": "string"
    },
    "execucao_julgado": {
      "forma_cumprimento": "string",
      "cronograma_sugerido": "string",
      "riscos_descumprimento": ["string"]
    }
  },
  "recomendacao_estrategica": {
    "acao_primaria": "contestar|recorrer|cumprir_voluntariamente|negociar_acordo|inacao_monitorada|medida_cautelar",
    "acoes_complementares": ["string"],
    "justificativa_detalhada": "string",
    "prioridade_institucional": "baixa|media|alta|urgente|estrategica",
    "cronograma_acao": {
      "prazo_imediato": "string",
      "prazo_medio": "string",
      "prazo_longo": "string"
    },
    "provas_complementares": [
      {
        "tipo_prova": "string",
        "dificuldade_obtencao": "baixa|media|alta",
        "impacto_resultado": "baixo|medio|alto",
        "responsavel_obtencao": "string"
      }
    ],
    "medidas_preventivas": ["string"]
  },
  "impacto_institucional": {
    "repercussao_geral": {
      "nivel": "baixa|media|alta|critica",
      "ambito": "local|regional|nacional",
      "midiatica": "baixa|media|alta",
      "politica": "baixa|media|alta"
    },
    "potencial_multiplicador": {
      "existe": "sim|nao",
      "estimativa_casos": "string",
      "areas_risco": ["string"],
      "valor_total_risco": "string"
    },
    "areas_municipio_afetadas": ["string"],
    "politicas_publicas_impactadas": ["string"],
    "necessidade_alteracao_normativa": {
      "necessaria": "sim|nao",
      "tipo_norma": "string",
      "urgencia": "baixa|media|alta"
    }
  },
  "ciencia_dados": {
    "tags_classificacao": ["string"],
    "categoria_principal": "string",
    "subcategorias": ["string"],
    "correlacoes_identificadas": ["string"],
    "padroes_detectados": ["string"]
  },
  "observacoes_tecnicas": {
    "alertas_importantes": ["string"],
    "recomendacoes_assessoria": ["string"],
    "pontos_atencao": ["string"],
    "documentos_anexar": ["string"]
  }
}
'''
# REGRAS DE PROCESSAMENTO ESPECÍFICAS
## ANÁLISE DE URGÊNCIA
1. **Impacto Financeiro Direto**: Valores acima de R$ 200k = Alto | R$ 1mi = Crítico
2. **Risco Ações Idênticas**: >10 casos similares = Alto | >50 casos = Crítico
3. **Repercussão Política**: Eleições, mídia, gestão = Alta/Crítica
4. **Impacto Operacional**: Paralisa serviços = Alto/Crítico
## ORIENTAÇÃO JURISPRUDENCIAL
1. **Palavras-chave**: Máximo 8 termos objetivos para busca
2. **Temas**: Questões específicas para pesquisa doutrinária
3. **Query Sugerida**: String de busca pronta para tribunais
4. **Estratégia**: Metodologia de pesquisa direcionada
## ANÁLISE FIRAC DETALHADA
1. **FATOS**: Cronologia completa com relevância jurídica classificada
2. **ISSUES**: Questões principal e secundárias com precedentes
3. **RULES**: Normas aplicáveis com interpretação favorabilidade
4. **ANALYSIS**: Correlação fatos-normas com cenários probabilísticos
5. **CONCLUSION**: Recomendação estratégica fundamentada
## CIÊNCIA DE DADOS
1. **Tags**: Máximo 10 tags objetivas para ML
2. **Categorização**: Hierárquica para clustering
3. **Indicadores**: Métricas quantificáveis para dashboards
4. **Padrões**: Identificação de regularidades para automação
# GARANTIA DE JSON VÁLIDO
- CRUCIAL: Resposta DEVE ser JSON válido e NADA MAIS
- Use aspas duplas para strings e propriedades
- Mantenha estrutura exata do schema
- Preencha todos os campos obrigatórios
- Use "string" para campos sem informação
# DADOS DO PROCESSO
<processo_dados>
{conteudo_json}
</processo_dados>
Execute análise FIRAC completa e estruturada conforme schema acima. Retorne EXCLUSIVAMENTE JSON válido.
"""

def configura_api_gemini():
    """Configura a autenticação da API do Gemini usando a variável de ambiente."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError('Variável de ambiente GEMINI_API_KEY não está definida')
    genai.configure(api_key=api_key)

# Função atualizada para Procuradoria Municipal
def generate_firac_analysis_procuradoria_extended(
    processo_json: dict,
    municipio_name: str = None,
    urgencia_custom: dict = None
):
    """
    Realiza análise FIRAC extensiva para Procuradoria Municipal com foco em ciência de dados.
    
    Args:
        processo_json: Dicionário contendo dados do processo judicial
        municipio_name: Nome do município (opcional)
        urgencia_custom: Critérios customizados de urgência (opcional)
    Returns:
        dict: Análise FIRAC estruturada completa
    """

    # Converter o JSON para string
    if isinstance(processo_json, dict):
        processo_str = json.dumps(processo_json, ensure_ascii=False, indent=2)
    else:
        processo_str = str(processo_json)

    prompt = PROMPT_TEMPLATE
    
    # Contexto municipal específico
    if municipio_name:
        contexto_municipal = f"""
    # CONTEXTO MUNICIPAL ESPECÍFICO
    Município: {municipio_name.upper()}
    Foque nas peculiaridades e precedentes específicos deste município.
    Considere histórico jurisprudencial local e políticas públicas municipais.
    """
        prompt = prompt.replace("# DADOS DO PROCESSO", contexto_municipal + "# DADOS DO PROCESSO")
    
    # Critérios de urgência customizados
    if urgencia_custom:
        criterios_custom = f"""
    # CRITÉRIOS DE URGÊNCIA CUSTOMIZADOS
    {json.dumps(urgencia_custom, ensure_ascii=False, indent=2)}
    """
        prompt = prompt.replace("# DADOS DO PROCESSO", criterios_custom + "# DADOS DO PROCESSO")
    
    prompt = prompt.replace("{conteudo_json}", processo_str)
    
    return prompt

def processar_arquivo(caminho_arquivo:pathlib.Path, model: GenerativeModel):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        processo = json.load(f)
    
    #if CAMPO_FIRAC in processo:
    #    print(f'Já análisado: {caminho_arquivo.name}')
    #    return
    
    prompt = generate_firac_analysis_procuradoria_extended(processo_json=processo, municipio_name='Natal')
    #prompt = PROMPT_TEMPLATE.replace("{conteudo_json}", json.dumps(processo['peticao_inicial_llm'], ensure_ascii=False))

    inicio = time.time()
    response = model.generate_content(prompt)
    fim = time.time()

    firac_output = response.text.strip()

    try:
        firac_json = json.loads(firac_output)
    except json.JSONDecodeError:
        print(f'Erro ao interpretar JSON retornado para {caminho_arquivo.name}')
        print(firac_output)
        return
    
    # Adiciona metadados de uso ao JSON
    usage = response.usage_metadata
    print(firac_json)
    if usage:
        firac_json['metadados']['tempo_processamento'] = f"{fim - inicio:.2f} segundos"
        #firac_json['metadados']['proomt_token_count'] = str(usage.promt_token_count)
        firac_json['metadados']['llm_tokens'] = str(usage.total_token_count)
        firac_json['metadados']['custo_estimado'] = str(usage)

    processo[CAMPO_FIRAC] = firac_json

    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        json.dump(processo, f, indent=2, ensure_ascii=False)
    print(f'Análise FIRAC salva em: {caminho_arquivo.name}')
    #time.sleep(1.5)

if __name__ == '__main__':
    configura_api_gemini()
    model = genai.GenerativeModel(MODELO,
                                   generation_config={
                                       'response_mime_type':'application/json',
                                       'temperature':0.1,
                                       'max_output_tokens':16384
                                   })

    arquivos = list(DIRETORIO_PROCESSOS.glob("*.json"))
    print(f'Encontrados {len(arquivos)} arquivos para análise.')

    for arquivo in arquivos:
        processar_arquivo(arquivo, model)
    
    print(f'Processamento concluido')
