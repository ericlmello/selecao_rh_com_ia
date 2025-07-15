# config.py
import os

# --- 1. DETECÇÃO DE AMBIENTE E DEFINIÇÃO DE CAMINHOS ---
# A forma mais robusta de encontrar o disco no Render é usar a variável de ambiente que ele fornece.
RENDER_DISK_PATH = os.environ.get('RENDER_DISK_PATH')

# Se a variável RENDER_DISK_PATH existir, usa-a como caminho base.
# Senão (em ambiente local), usa o diretório atual ('.').
BASE_PATH = RENDER_DISK_PATH if RENDER_DISK_PATH else '.'

# Define os caminhos dinamicamente a partir do caminho base
DATA_DIR = os.path.join(BASE_PATH, 'data')
MODELS_DIR = os.path.join(BASE_PATH, 'models')
MLRUNS_DIR = os.path.join(BASE_PATH, 'mlruns')
DATABASE_PATH = os.path.join(BASE_PATH, 'predictions.db')


# --- 2. CLASSE DE CONFIGURAÇÃO PRINCIPAL ---
class Config:
    """
    Classe de configuração que usa os caminhos dinâmicos definidos acima
    e armazena os hiperparâmetros do projeto.
    """
    # --- Configuração de Dados ---
    # IDs dos ficheiros .zip no Google Drive
    GDRIVE_ZIP_FILE_IDS = {
        'applicants': '1Z0dOk8FMjazQo03PuUeNGZOW-rxtpzmO',
        'prospects': '17RkgTlckZ6ItDqgsDCT8H5HZn_OwmxQO',
        'jobs': '1h8Lk5LM8VE5TF80mngCcbsQ14qA2rbw_'
    }
    
    # Caminhos onde os zips serão guardados temporariamente
    ZIP_OUTPUT_PATHS = {
        'applicants': os.path.join(BASE_PATH, 'applicants.zip'),
        'prospects': os.path.join(BASE_PATH, 'prospects.zip'),
        'jobs': os.path.join(BASE_PATH, 'jobs.zip')
    }
    
    # Caminhos onde os ficheiros de dados estarão após a extração
    DATA_PATHS = {
        'jobs': os.path.join(DATA_DIR, 'jobs.json'),
        'prospects': os.path.join(DATA_DIR, 'prospects.json'),
        'applicants': os.path.join(DATA_DIR, 'applicants.json')
    }
    
    # --- Configurações do Modelo ---
    MODEL_PATH = os.path.join(MODELS_DIR, 'recommender_model.pth')
    MODEL_DIR = MODELS_DIR
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 1
    
    # --- Configurações de Treinamento ---
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32

    # --- Configurações da Aplicação ---
    SECRET_KEY = os.environ.get('SECRET_KEY', 'uma-chave-secreta-de-desenvolvimento')
    DEBUG = RENDER_DISK_PATH is None # Ativa o debug apenas se não estiver no Render


# --- 3. CRIAÇÃO DE DIRETÓRIOS ---
# Garante que os diretórios necessários existam assim que este ficheiro for importado.
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)
except OSError as e:
    print(f"Erro ao criar diretórios: {e}")
    raise

