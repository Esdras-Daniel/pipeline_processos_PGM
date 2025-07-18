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
Você é um especialista em Direito, Linguística, Ciências Cognitivas e Ciências Sociais. Sua tarefa é ler atentamente todos os documentos fornecidos sobre um processo jurídico e elaborar uma análise completa, crítica e fundamentada utilizando o modelo FIRAC para a contestação, com base apenas nas informações constantes nos autos.
Consulte todos os documentos fornecidos na íntegra. Eles podem conter informações contraditórias. Por isso, faça uma leitura holística para captar todos os pontos controvertidos e todas as questões jurídicas em sua profundidade e totalidade.
Sua resposta deve ser estruturada estritamente no seguinte formato JSON, com os campos preenchidos com base nas informações dos autos. Não inclua comentários ou explicações fora do JSON:

{
  "dados_processo": {
    "tribunal": "",
    "tipo_acao_ou_recurso": "",
    "numero_processo": "",
    "relator": "",
    "data_julgamento": ""
  },
  "fatos": "[Descreva detalhadamente todos os fatos relevantes do caso, com inferência lógica]",
  "problema_juridico": {
    "questao_central": "[Delimite a principal questão jurídica a ser resolvida]",
    "pontos_controvertidos": [
      "[Ponto controverso 1]",
      "[Ponto controverso 2]"
    ]
  },
  "direito_aplicavel": "[Liste e interprete as normas legais invocadas nos autos]",
  "analise_e_aplicacao": {
    "argumentos_e_provas_do_autor": "[Avaliação dos argumentos e provas apresentadas]",
    "aplicacao_da_norma": "[Análise jurídica dos fatos com base nas normas aplicáveis]"
  },
  "conclusao": "[Informe se o caso foi julgado ou não. Se sim, indique a ratio decidendi. Se não, indique possíveis encaminhamentos.]"
}

Abaixo está o conteúdo do processo:
{conteudo_json}
"""

def configura_api_gemini():
    """Configura a autenticação da API do Gemini usando a variável de ambiente."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError('Variável de ambiente GEMINI_API_KEY não está definida')
    genai.configure(api_key=api_key)

def processar_arquivo(caminho_arquivo:pathlib.Path, model: GenerativeModel):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        processo = json.load(f)
    
    if CAMPO_FIRAC in processo:
        print(f'Já análisado: {caminho_arquivo.name}')
        return
    
    prompt = PROMPT_TEMPLATE.replace("{conteudo_json}", json.dumps(processo['peticao_inicial_llm'], ensure_ascii=False))

    response = model.generate_content(prompt)
    firac_output = response.text.strip()

    try:
        firac_json = json.loads(firac_output)
    except json.JSONDecodeError:
        print(f'Erro ao interpretar JSON retornado para {caminho_arquivo.name}:')
        print(firac_output)
        return

    processo[CAMPO_FIRAC] = firac_json

    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        json.dump(processo, f, indent=2, ensure_ascii=False)
    print(f'Análise FIRAC salva em: {caminho_arquivo.name}')
    #time.sleep(1.5)

if __name__ == '__main__':
    configura_api_gemini()
    model = genai.GenerativeModel(MODELO,
                                   generation_config={
                                       'response_mime_type':'application/json'
                                   })

    arquivos = list(DIRETORIO_PROCESSOS.glob("*.json"))
    print(f'Encontrados {len(arquivos)} arquivos para análise.')

    for arquivo in arquivos:
        processar_arquivo(arquivo, model)
    
    print(f'Processamento concluido')
