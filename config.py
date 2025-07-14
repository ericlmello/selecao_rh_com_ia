import os
import tempfile

# --- 1. DETECÇÃO DE AMBIENTE E DEFINIÇÃO DE CAMINHOS ---
# Detecta se está a rodar no Render
IS_ON_RENDER = os.environ.get('RENDER') is not None

if IS_ON_RENDER:
    # No Render, usa diretório temporário (efêmero)
    BASE_PATH = '/tmp/app_data'
else:
    # Ambiente local
    BASE_PATH = '.'

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
    
    # --- Configurações específicas do Render ---
    # No Render, os dados são baixados a cada inicialização
    DOWNLOAD_DATA_ON_START = IS_ON_RENDER
    
    # Configurações do Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Configurações de debug
    DEBUG = not IS_ON_RENDER  # Debug apenas em ambiente local

# --- 3. CRIAÇÃO DE DIRETÓRIOS ---
def ensure_directories():
    """
    Garante que os diretórios necessários existam.
    Função separada para melhor controle de erros.
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(MLRUNS_DIR, exist_ok=True)
        
        if IS_ON_RENDER:
            print(f"Diretórios criados no Render: {BASE_PATH}")
        else:
            print(f"Diretórios criados localmente: {BASE_PATH}")
            
    except Exception as e:
        print(f"Erro ao criar diretórios: {e}")
        # Em caso de erro, tenta usar diretório temporário do sistema
        global BASE_PATH, DATA_DIR, MODELS_DIR, MLRUNS_DIR, DATABASE_PATH
        BASE_PATH = tempfile.mkdtemp()
        DATA_DIR = os.path.join(BASE_PATH, 'data')
        MODELS_DIR = os.path.join(BASE_PATH, 'models')
        MLRUNS_DIR = os.path.join(BASE_PATH, 'mlruns')
        DATABASE_PATH = os.path.join(BASE_PATH, 'predictions.db')
        
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(MLRUNS_DIR, exist_ok=True)
        
        print(f"Usando diretório temporário: {BASE_PATH}")

# --- 4. INICIALIZAÇÃO ---
# Chama a função para criar os diretórios
ensure_directories()

# --- 5. VARIÁVEIS DE AMBIENTE ADICIONAIS PARA O RENDER ---
if IS_ON_RENDER:
    # Port binding para o Render
    PORT = int(os.environ.get('PORT', 5000))
    
    # Configurações de produção
    os.environ['FLASK_ENV'] = 'production'
    
    print(f"Configuração do Render ativada. Port: {PORT}")
    print(f"Diretório base: {BASE_PATH}")
else:
    PORT = 5000
    print("Configuração local ativada")

