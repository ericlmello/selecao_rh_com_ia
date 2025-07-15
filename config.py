# config.py
import os

# --- 1. DETECÇÃO DE AMBIENTE E DEFINIÇÃO DE CAMINHOS ---
# Detecta se está a rodar no ambiente do Render.
# A variável 'RENDER' é definida automaticamente pela plataforma Render.
IS_ON_RENDER = os.environ.get('RENDER') is not None

# Define o caminho base para todos os ficheiros.
# No Render, usa o disco persistente montado em '/app/persistent_data'.
# Localmente, usa o diretório atual do projeto ('.').
BASE_PATH = '/app/persistent_data' if IS_ON_RENDER else '.'

# Define os caminhos dinamicamente a partir do caminho base
DATA_DIR = os.path.join(BASE_PATH, 'data')
MODELS_DIR = os.path.join(BASE_PATH, 'models')
MLRUNS_DIR = os.path.join(BASE_PATH, 'mlruns')
DATABASE_PATH = os.path.join(BASE_PATH, 'predictions.db')


# --- 2. CLASSE DE CONFIGURAÇÃO PRINCIPAL ---
class Config:
    """
    Classe de configuração que centraliza todos os parâmetros do projeto.
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
        'jobs': os.path.join(DATA_DIR, 'jobs.json'), # Alterado para .json
        'prospects': os.path.join(DATA_DIR, 'prospects.json'), # Alterado para .json
        'applicants': os.path.join(DATA_DIR, 'applicants.json') # Alterado para .json
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
    # Chave secreta para a aplicação Flask. É importante para a segurança.
    SECRET_KEY = os.environ.get('SECRET_KEY', 'uma-chave-secreta-de-desenvolvimento')
    
    # Ativa o modo de debug apenas em ambiente local
    DEBUG = not IS_ON_RENDER


# --- 3. CRIAÇÃO DE DIRETÓRIOS ---
# Garante que os diretórios necessários existam assim que este ficheiro for importado.
# Esta é uma forma limpa e segura de garantir que as pastas estão prontas.
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)
except OSError as e:
    print(f"Erro ao criar diretórios: {e}")
    # Em caso de erro, a aplicação irá falhar, o que é preferível a continuar
    # com uma configuração de ficheiros incorreta.
