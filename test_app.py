'''Scipt de testes''
import sys
import os
import pytest
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# -------------------------------------------


# ------------------ SETUP DO AMBIENTE DE TESTE (FIXTURE) ------------------

@pytest.fixture(scope='module')
def client():
    """
    Cria e configura um cliente de teste para a aplicação Flask.
    Esta 'fixture' garante que a aplicação seja completamente inicializada
    UMA VEZ por sessão de teste, o que é eficiente e correto.
    
    CORREÇÃO: Esta fixture agora cria os arquivos de dados e modelo necessários
    para garantir que o ambiente de teste seja autocontido e não dependa
    da execução prévia de 'app.py'.
    """
    # --- 1. SETUP DO AMBIENTE DE TESTE ---
    # Importa componentes do app APENAS AQUI para evitar problemas de escopo.
    from app import app, RecommenderModel

    # Define os caminhos para os arquivos de teste
    test_data_dir = 'test_data'
    test_model_dir = 'test_models'
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(test_model_dir, exist_ok=True)

    # Sobrescreve a configuração do app para usar os caminhos de teste
    app.config['TEST_DATA_PATHS'] = {
        'jobs': os.path.join(test_data_dir, 'jobs.csv'),
        'prospects': os.path.join(test_data_dir, 'prospects.csv'),
        'applicants': os.path.join(test_data_dir, 'applicants.csv')
    }
    app.config['TEST_MODEL_PATH'] = os.path.join(test_model_dir, 'test_model.pth')

    # Cria arquivos de dados DUMMY para os testes
    pd.DataFrame({
        'vaga_id': ['4534', '123'], 'titulo_vaga': ['Engenheiro de Teste', 'Dev'], 'descricao': ['Testar', 'Codar'], 'cliente': ['Empresa A', 'B']
    }).to_csv(app.config['TEST_DATA_PATHS']['jobs'], index=False)
    
    # CORREÇÃO: Garante que todas as listas (colunas) tenham o mesmo tamanho.
    pd.DataFrame({
        'candidato_id': ['11132', '456'], 
        'nome': ['Candidato Teste', 'Outro'], 
        'cargo_atual': ['Analista', 'Pleno'], 
        'campo_extra_cv_pt': ['Python', 'Java e Go'], 
        'campo_extra_cv_en': ['Python experience', 'Java and Go'], 
        'habilidades': ['SQL', 'Docker, K8s']
    }).to_csv(app.config['TEST_DATA_PATHS']['applicants'], index=False)

    # CORREÇÃO: Adiciona um segundo valor para 'comentario' para igualar o tamanho das outras colunas
    pd.DataFrame({
        'vaga_id': ['4534', '123'], 
        'codigo': ['11132', '456'], 
        'comentario': ['', 'Bom candidato'], 
        'situacao_candidado': ['pendente', 'aprovado']
    }).to_csv(app.config['TEST_DATA_PATHS']['prospects'], index=False)

    # Cria um modelo DUMMY
    torch.save(RecommenderModel().state_dict(), app.config['TEST_MODEL_PATH'])
    
    # Monkeypatch: Altera o objeto Config em tempo de execução para os testes
    # Isso faz com que initialize_components use os caminhos de teste
    from app import Config
    Config.DATA_PATHS = app.config['TEST_DATA_PATHS']
    Config.MODEL_PATH = app.config['TEST_MODEL_PATH']

    # --- 2. INICIALIZAÇÃO DO APP ---
    from app import initialize_components, init_db
    
    # Inicializa o banco de dados e os componentes
    init_db()
    if not initialize_components():
        pytest.fail("A inicialização dos componentes da aplicação falhou. Verifique o app.py e os caminhos dos dados.")

    # Ativa o modo de teste do Flask
    app.config['TESTING'] = True

    # --- 3. EXECUÇÃO DOS TESTES ---
    with app.test_client() as test_client:
        yield test_client

    # --- 4. LIMPEZA APÓS OS TESTES ---
    # Remove os arquivos e diretórios de teste
    for path in app.config['TEST_DATA_PATHS'].values():
        os.remove(path)
    os.remove(app.config['TEST_MODEL_PATH'])
    os.rmdir(test_data_dir)
    os.rmdir(test_model_dir)


# --------------------- TESTES DE INTEGRAÇÃO (API ENDPOINTS) ---------------------


def test_predict_success_scenario(client):
    """Testa um cenário de SUCESSO para a rota /predict."""
    print("Executando teste de integração para: POST /predict (Sucesso)")
    payload = {'job_id': '4534', 'candidate_id': '11132'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'success_probability' in data
    assert data['job_id'] == '4534'
    assert data['recommendation'] is not None

def test_predict_failure_for_invalid_id(client):
    """Testa um cenário de FALHA para a rota /predict com IDs inválidos."""
    print("Executando teste de integração para: POST /predict (Falha)")
    payload = {'job_id': '99999', 'candidate_id': 'id_invalido'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 400  # A aplicação retorna 400 para IDs inválidos
    data = response.get_json()
    assert 'error' in data
    assert 'ID inválido' in data['error']


# --------------------------- TESTES UNITÁRIOS (FUNÇÕES ISOLADAS) ---------------------------

def test_unit_calculate_metrics_perfect_score():
    """Testa a função 'calculate_metrics' em um cenário de acerto perfeito."""
    from app import calculate_metrics
    print("\nExecutando teste unitário para: calculate_metrics (Pontuação Perfeita)")
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    metrics = calculate_metrics(y_true, y_pred, threshold=0.5)
    assert metrics['accuracy'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1_score'] == 1.0

