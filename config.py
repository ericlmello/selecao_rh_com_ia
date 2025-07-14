import os

# --- 1. DETECÇÃO DE AMBIENTE E DEFINIÇÃO DE CAMINHOS ---
# Detecta se está a rodar no Render. Se não, assume ambiente local.
IS_ON_RENDER = os.environ.get('RENDER')
BASE_PATH = '/app/persistent_data' if IS_ON_RENDER else '.'

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
    
    # Caminhos onde os ficheiros CSV estarão após a extração
    DATA_PATHS = {
        'jobs': os.path.join(DATA_DIR, 'jobs.csv'),
        'prospects': os.path.join(DATA_DIR, 'prospects.csv'),
        'applicants': os.path.join(DATA_DIR, 'applicants.csv')
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


# --- 3. CRIAÇÃO DE DIRETÓRIOS ---
# Garante que os diretórios necessários existam assim que este arquivo for importado.
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

