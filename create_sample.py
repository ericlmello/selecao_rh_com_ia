"""
Este script lê os ficheiros JSON brutos e grandes, processa-os para um formato
de tabela limpo e cria uma amostra pequena e otimizada para ser usada em
ambientes com memória limitada, como o Render.
"""

import os
import pandas as pd
import json

# --- CONFIGURAÇÕES ---
# Define o número de registos de "prospects" que queremos na nossa amostra final.
SAMPLE_SIZE = 500

# Define os nomes das pastas de origem (com os ficheiros grandes) e de destino (com a amostra).
SOURCE_DATA_FOLDER = 'data_original'
DESTINATION_DATA_FOLDER = 'data'

# --- FUNÇÕES DE PROCESSAMENTO (do seu script original) ---

def carregar_json_bruto(caminho):
    """Carrega um ficheiro JSON bruto."""
    print(f"  Lendo ficheiro bruto: {caminho}")
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def processar_jobs(data):
    """Processa e achata os dados de vagas."""
    records = []
    for vaga_id, conteudo in data.items():
        info = conteudo.get('informacoes_basicas', {})
        perfil = conteudo.get('perfil_vaga', {})
        record = {
            'vaga_id': vaga_id, 'data_requisicao': info.get('data_requicisao'),
            'limite_contratacao': info.get('limite_esperado_para_contratacao'),
            'titulo_vaga': info.get('titulo_vaga'), 'cliente': info.get('cliente'),
            'solicitante_cliente': info.get('solicitante_cliente'),
            'empresa_divisao': info.get('empresa_divisao'),
            'analista_responsavel': info.get('analista_responsavel'),
            'tipo_contratacao': info.get('tipo_contratacao'), 'pais': perfil.get('pais'),
            'estado': perfil.get('estado'), 'cidade': perfil.get('cidade')
        }
        records.append(record)
    return pd.DataFrame(records)

def processar_prospects(data):
    """Processa e achata os dados de prospects."""
    records = []
    for vaga_id, conteudo in data.items():
        lista = conteudo.get('prospects', [])
        for item in lista:
            record = {'vaga_id': vaga_id}
            record.update(item)
            records.append(record)
    return pd.DataFrame(records)

def processar_applicants(data):
    """Processa e achata os dados de candidatos."""
    records = []
    for candidato_id, conteudo in data.items():
        infos_basicas = conteudo.get('infos_basicas', {})
        info_pessoais = conteudo.get('informacoes_pessoais', {})
        info_profissionais = conteudo.get('informacoes_profissionais', {})
        formacao_idiomas = conteudo.get('formacao_e_idiomas', {})
        record = {
            'candidato_id': candidato_id, 'cargo_atual': conteudo.get('cargo_atual'),
            'nome': infos_basicas.get('nome'), 'email': infos_basicas.get('email'),
            'telefone': infos_basicas.get('telefone'), 'idade': info_pessoais.get('idade'),
            'cidade': info_pessoais.get('cidade'), 'estado': info_pessoais.get('estado'),
            'nacionalidade': info_pessoais.get('nacionalidade'),
            'estado_civil': info_pessoais.get('estado_civil'),
            'experiencia_anos': info_profissionais.get('experiencia_total_anos'),
            'ultimo_cargo': info_profissionais.get('ultimo_cargo'),
            'ultima_empresa': info_profissionais.get('ultima_empresa'),
            'setor_atuacao': info_profissionais.get('setor_atuacao'),
            'nivel_educacao': formacao_idiomas.get('nivel_educacao'),
            'curso': formacao_idiomas.get('curso'),
            'instituicao': formacao_idiomas.get('instituicao'),
            'idiomas': formacao_idiomas.get('idiomas'),
            'habilidades': conteudo.get('habilidades')
        }
        records.append(record)
    return pd.DataFrame(records)

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    print("="*50)
    print("INICIANDO SCRIPT DE CRIAÇÃO DE AMOSTRA DE DADOS")
    print("="*50)

    # Verifica se a pasta de origem existe
    if not os.path.exists(SOURCE_DATA_FOLDER):
        print(f"\nERRO: A pasta de origem '{SOURCE_DATA_FOLDER}' não foi encontrada.")
        print("Por favor, crie esta pasta e coloque os seus ficheiros JSON grandes dentro dela.")
        exit()

    # Cria a pasta de destino se não existir
    os.makedirs(DESTINATION_DATA_FOLDER, exist_ok=True)

    # 1. Carrega e processa todos os ficheiros
    print("\n[PASSO 1 de 3] Processando ficheiros JSON brutos...")
    raw_jobs = carregar_json_bruto(os.path.join(SOURCE_DATA_FOLDER, 'jobs.json'))
    raw_prospects = carregar_json_bruto(os.path.join(SOURCE_DATA_FOLDER, 'prospects.json'))
    raw_applicants = carregar_json_bruto(os.path.join(SOURCE_DATA_FOLDER, 'applicants.json'))

    df_jobs_full = processar_jobs(raw_jobs)
    df_prospects_full = processar_prospects(raw_prospects)
    df_applicants_full = processar_applicants(raw_applicants)
    print("  -> Ficheiros processados com sucesso!")

    # 2. Cria a amostra
    print(f"\n[PASSO 2 de 3] Criando amostra de {SAMPLE_SIZE} registros...")
    df_prospects_sample = df_prospects_full.head(SAMPLE_SIZE).copy()

    # 3. Filtra os outros dataframes com base na amostra
    valid_job_ids = df_prospects_sample['vaga_id'].astype(str).unique()
    valid_candidate_ids = df_prospects_sample['codigo'].astype(str).unique()

    df_jobs_sample = df_jobs_full[df_jobs_full['vaga_id'].astype(str).isin(valid_job_ids)]
    df_applicants_sample = df_applicants_full[df_applicants_full['candidato_id'].astype(str).isin(valid_candidate_ids)]
    print("  -> Amostra e filtragem concluídas.")

    # 4. Salva os novos ficheiros de amostra
    print("\n[PASSO 3 de 3] Salvando ficheiros de amostra...")
    output_paths = {
        'jobs': os.path.join(DESTINATION_DATA_FOLDER, 'jobs.json'),
        'prospects': os.path.join(DESTINATION_DATA_FOLDER, 'prospects.json'),
        'applicants': os.path.join(DESTINATION_DATA_FOLDER, 'applicants.json')
    }

    df_jobs_sample.to_json(output_paths['jobs'], orient='records', lines=True, force_ascii=False)
    df_prospects_sample.to_json(output_paths['prospects'], orient='records', lines=True, force_ascii=False)
    df_applicants_sample.to_json(output_paths['applicants'], orient='records', lines=True, force_ascii=False)
    
    print(f"  -> Ficheiros salvos na pasta '{DESTINATION_DATA_FOLDER}'!")
    print("\n" + "="*50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*50)
    print(f"\nResumo da Amostra:")
    print(f"  - Prospects: {len(df_prospects_sample)} registros")
    print(f"  - Jobs:      {len(df_jobs_sample)} registros")
    print(f"  - Applicants:{len(df_applicants_sample)} registros")
