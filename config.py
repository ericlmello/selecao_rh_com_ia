'''
class Config:
    # Caminhos dos arquivos
    DATA_PATHS = {
        'jobs': 'C:\\Users\\e178454\\Desktop\\partic\\FIAP ML\\selecao\\data\\jobs.csv',
        'prospects': 'C:\\Users\\e178454\\Desktop\\partic\\FIAP ML\\selecao\\data\\prospects.csv',
        'applicants': 'C:\\Users\\e178454\\Desktop\\partic\\FIAP ML\\selecao\\data\\applicants.csv'
    }
    
    # Configurações do modelo
    MODEL_PATH = 'model.pth'
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 1
    
    # Configurações de treinamento
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32'''
    
import os

# --- 1. DETECÇÃO DE AMBIENTE E DEFINIÇÃO DE CAMINHOS ---
# Detecta se está a rodar no Render. Se não, assume ambiente local.
IS_ON_RENDER = os.environ.get('RENDER')
BASE_PATH = '/app/persistent_data' if IS_ON_RENDER else '.'

# Define os caminhos dinamicamente a partir do caminho base
MODELS_DIR = os.path.join(BASE_PATH, 'models')
DATA_DIR = os.path.join(BASE_PATH, 'data')
MLRUNS_DIR = os.path.join(BASE_PATH, 'mlruns')
DATABASE_PATH = os.path.join(BASE_PATH, 'predictions.db')

# --- 2. CLASSE DE CONFIGURAÇÃO PRINCIPAL ---
class Config:
    """
    Classe de configuração que usa os caminhos dinâmicos definidos acima.
    """
    # Caminhos dos arquivos de dados
    DATA_PATHS = {
        'jobs': os.path.join(DATA_DIR, 'jobs.csv'),
        'prospects': os.path.join(DATA_DIR, 'prospects.csv'),
        'applicants': os.path.join(DATA_DIR, 'applicants.csv')
    }
    
    # Caminhos do modelo
    MODEL_PATH = os.path.join(MODELS_DIR, 'recommender_model.pth')
    MODEL_DIR = MODELS_DIR
    
    # Hiperparâmetros do modelo e treinamento
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32

# --- 3. CRIAÇÃO DE DIRETÓRIOS ---
# Garante que os diretórios necessários existam assim que este arquivo for importado.
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)
