"""
Aplicação Flask principal para o Sistema de Recomendação de Candidatos.

"""

# --- 1. IMPORTAÇÕES PRINCIPAIS ---
import os
import sys
import time
import logging
import json
import sqlite3
import shutil
import traceback
import zipfile
import gc # Importa o Garbage Collector
from datetime import datetime

# --- 2. IMPORTAÇÕES DE BIBLIOTECAS DE DADOS E ML ---
import pandas as pd
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch
import gdown

# --- 3. CONFIGURAÇÃO DE AMBIENTE (LOCAL vs RENDER) ---
try:
    from config import Config, DATABASE_PATH, MLRUNS_DIR, DATA_DIR
except ImportError:
    class Config:
        DATA_PATHS = {}
        MODEL_PATH = 'models/recommender_model.pth'
    DATABASE_PATH = 'predictions.db'
    MLRUNS_DIR = 'mlruns'
    DATA_DIR = 'data'

# --- 4. CONFIGURAÇÃO DE LOGGING E MLFLOW ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('app.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
try:
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
except Exception as e:
    logger.error(f"Falha ao configurar o tracking URI do MLflow: {e}")

# --- 5. INICIALIZAÇÃO DA APLICAÇÃO FLASK E BANCO DE DADOS ---
app = Flask(__name__)
DATABASE = DATABASE_PATH

def get_db():
    db = sqlite3.connect(DATABASE, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        with db as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, timestamp TEXT, job_id TEXT, candidate_id TEXT, success_probability REAL, recommendation TEXT, features_json TEXT, total_time REAL);''')
            conn.execute('''CREATE TABLE IF NOT EXISTS evaluations (id INTEGER PRIMARY KEY, timestamp TEXT, total_samples INTEGER, avg_prediction REAL, metrics_json TEXT);''')
        logger.info(f"Banco de dados SQLite inicializado em: {DATABASE}")

# --- 6. VARIÁVEIS GLOBAIS E CLASSES ---
processor, model, jobs_df, prospects_df, applicants_df = None, None, None, None, None

class RecommenderModel(torch.nn.Module):
    def __init__(self, input_size=201, hidden_layer_1_size=128, hidden_layer_2_size=64, dropout_rate=0.2):
        super(RecommenderModel, self).__init__()
        self.input_size = input_size
        self.features = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_layer_1_size), torch.nn.ReLU(), torch.nn.Dropout(dropout_rate), torch.nn.Linear(hidden_layer_1_size, hidden_layer_2_size), torch.nn.ReLU(), torch.nn.Dropout(dropout_rate))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_layer_2_size, 1), torch.nn.Sigmoid())
    def forward(self, x):
        return self.classifier(self.features(x))

class SimpleProcessor:
    def __init__(self):
        self.vectorizers = {}
        self.scaler = None
    def initialize_text_vectorizers(self, applicants_df, jobs_df, max_features=100):
        sample_texts = ["experiencia profissional", "vaga emprego", "candidato qualificado"]
        try:
            text_columns_applicants = ['campo_extra_cv_pt', 'campo_extra_cv_en', 'habilidades']
            text_columns_jobs = ['titulo_vaga', 'descricao']
            all_texts = sample_texts.copy()
            for col in text_columns_applicants:
                if col in applicants_df.columns:
                    texts = applicants_df[col].dropna().astype(str)
                    all_texts.extend([safe_clean_text(t) for t in texts if t])
            for col in text_columns_jobs:
                if col in jobs_df.columns:
                    texts = jobs_df[col].dropna().astype(str)
                    all_texts.extend([safe_clean_text(t) for t in texts if t])
            all_texts = [t for t in all_texts if t.strip()]
            if not all_texts:
                all_texts = sample_texts
            self.vectorizers['text'] = TfidfVectorizer(max_features=max_features, stop_words=None, ngram_range=(1, 1))
            self.vectorizers['text'].fit(all_texts)
            logger.info("Vectorizer treinado com textos limpos.")
        except Exception as e:
            logger.error(f"Erro ao treinar vectorizer: {e}")
            self.vectorizers['text'] = TfidfVectorizer(max_features=100)
            self.vectorizers['text'].fit(sample_texts)

# --- 7. FUNÇÕES AUXILIARES ---
def download_and_unzip_data():
    logger.info("Verificando ficheiros de dados...")
    for key, file_id in Config.GDRIVE_ZIP_FILE_IDS.items():
        final_json_path = Config.DATA_PATHS[key]
        if os.path.exists(final_json_path):
            logger.info(f"Ficheiro '{os.path.basename(final_json_path)}' já existe.")
            continue
        logger.warning(f"Ficheiro '{os.path.basename(final_json_path)}' não encontrado. A descarregar...")
        zip_output_path = Config.ZIP_OUTPUT_PATHS[key]
        try:
            gdown.download(id=file_id, output=zip_output_path, quiet=False)
            temp_extract_dir = os.path.join(DATA_DIR, f"temp_{key}")
            with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            found_file = None
            for root, _, files in os.walk(temp_extract_dir):
                if files:
                    found_file = os.path.join(root, files[0])
                    break
            if not found_file: raise Exception(f"Nenhum ficheiro de dados encontrado dentro de {zip_output_path}")
            shutil.move(found_file, final_json_path)
            logger.info(f"Ficheiro '{os.path.basename(found_file)}' movido para '{final_json_path}'.")
            os.remove(zip_output_path)
            shutil.rmtree(temp_extract_dir)
        except Exception as e:
            logger.error(f"Falha ao obter dados para '{key}': {e}")
            raise

def safe_load_json_optimized(file_path, columns_to_keep, dtype_map=None, sample_size=None):
    df = None
    try:
        logger.info(f"Lendo o ficheiro JSON bruto: '{os.path.basename(file_path)}'")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        df = pd.json_normalize(raw_data)
        logger.info(f"Ficheiro '{os.path.basename(file_path)}' normalizado com sucesso.")
    except (json.JSONDecodeError, AttributeError):
        logger.warning(f"Falha na normalização. Tentando carregar como JSON-Lines...")
        try:
            df = pd.read_json(file_path, lines=True, dtype=dtype_map)
        except Exception as e:
            logger.error(f"Falha ao carregar o ficheiro JSON '{os.path.basename(file_path)}': {e}")
            return None
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {file_path}")
        return None
    if df.empty:
        logger.warning(f"DataFrame '{os.path.basename(file_path)}' está vazio após o carregamento.")
        return df
    actual_cols_to_keep = [col for col in columns_to_keep if col in df.columns]
    if len(actual_cols_to_keep) < len(columns_to_keep):
        missing_cols = set(columns_to_keep) - set(actual_cols_to_keep)
        logger.warning(f"Colunas esperadas ausentes no ficheiro '{os.path.basename(file_path)}': {missing_cols}")
    df = df[actual_cols_to_keep]
    if sample_size and len(df) > sample_size:
        logger.info(f"Reduzindo DataFrame de {len(df)} para uma amostra de {sample_size} registros.")
        df = df.head(sample_size)
    return df

def safe_clean_text(text):
    try:
        if pd.isna(text) or text is None: return ""
        return ''.join(char for char in str(text) if char.isprintable() or char.isspace())
    except Exception: return ""

def load_model(model_path, **kwargs):
    model_instance = RecommenderModel(**kwargs)
    model_instance.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_instance.eval()
    return model_instance

def get_model_input_size(model_instance):
    return getattr(model_instance, 'input_size', 201)

def calculate_metrics(y_true, y_pred, threshold=0.5):
    try:
        y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
        y_pred_binary = (y_pred_np >= threshold).astype(int)
        return {
            'accuracy': round(accuracy_score(y_true_np, y_pred_binary), 3),
            'precision': round(precision_score(y_true_np, y_pred_binary, zero_division=0), 3),
            'recall': round(recall_score(y_true_np, y_pred_binary, zero_division=0), 3),
            'f1_score': round(f1_score(y_true_np, y_pred_binary, zero_division=0), 3)
        }
    except Exception as e:
        logger.error(f"Erro ao calcular métricas: {e}")
        return None

def extract_single_prediction_features(job_data, candidate_data):
    try:
        cv_text = " ".join(safe_clean_text(candidate_data.get(field, '')) for field in ['campo_extra_cv_pt', 'campo_extra_cv_en', 'habilidades'])
        job_text = " ".join(safe_clean_text(job_data.get(field, '')) for field in ['titulo_vaga', 'descricao'])
        if not cv_text.strip(): cv_text = "sem experiencia"
        if not job_text.strip(): job_text = "sem descricao vaga"
        cv_vec = processor.vectorizers['text'].transform([cv_text]).toarray()[0]
        job_vec = processor.vectorizers['text'].transform([job_text]).toarray()[0]
        cv_length_feature = len(cv_text.split())
        return np.concatenate([cv_vec, job_vec, [cv_length_feature]])
    except Exception as e:
        logger.error(f"Erro na extração de features: {e}.")
        return None

def train_new_model(data_paths_config, output_model_dir, hyperparameters):
    logger.info(f"Iniciando o treinamento de um novo modelo com hiperparâmetros: {hyperparameters}")
    # ... (código da função) ...
    pass

# --- 8. FUNÇÃO DE INICIALIZAÇÃO PRINCIPAL ---
def initialize_components():
    global processor, model, jobs_df, prospects_df, applicants_df
    # CORREÇÃO: Chamando a função com o nome correto.
    download_and_unzip_data()
    try:
        logger.info("Inicializando componentes com amostragem de dados...")
        
        SAMPLE_SIZE = 500
        
        prospects_cols = ['vaga_id', 'codigo', 'situacao_candidado']
        prospects_dtypes = {'vaga_id': 'str', 'codigo': 'str', 'situacao_candidado': 'category'}
        
        prospects_df = safe_load_json_optimized(Config.DATA_PATHS['prospects'], prospects_cols, prospects_dtypes, sample_size=SAMPLE_SIZE)
        if prospects_df is None or prospects_df.empty:
            raise ValueError("Não foi possível carregar a amostra de prospects.")
        logger.info(f"Amostra de {len(prospects_df)} prospects carregada.")

        valid_job_ids = prospects_df['vaga_id'].unique()
        valid_candidate_ids = prospects_df['codigo'].unique()
        
        jobs_cols = ['vaga_id', 'titulo_vaga', 'descricao', 'cliente']
        jobs_dtypes = {'vaga_id': 'str', 'cliente': 'category'}
        full_jobs_df = safe_load_json_optimized(Config.DATA_PATHS['jobs'], jobs_cols, jobs_dtypes)
        
        applicants_cols = ['candidato_id', 'nome', 'cargo_atual', 'campo_extra_cv_pt', 'campo_extra_cv_en', 'habilidades']
        applicants_dtypes = {'candidato_id': 'str', 'cargo_atual': 'category'}
        full_applicants_df = safe_load_json_optimized(Config.DATA_PATHS['applicants'], applicants_cols, applicants_dtypes)

        if full_jobs_df is None or full_applicants_df is None:
            raise ValueError("Falha ao carregar os dataframes completos de jobs ou applicants.")

        jobs_df = full_jobs_df[full_jobs_df['vaga_id'].astype(str).isin(valid_job_ids)]
        applicants_df = full_applicants_df[full_applicants_df['candidato_id'].astype(str).isin(valid_candidate_ids)]
        
        logger.info(f"Dados filtrados: {len(jobs_df)} vagas, {len(applicants_df)} candidatos.")
        
        del full_jobs_df, full_applicants_df
        gc.collect()
        
        if any(df is None or df.empty for df in [jobs_df, prospects_df, applicants_df]):
             raise ValueError("Um ou mais dataframes estão vazios após a amostragem e filtragem.")
        
        processor = SimpleProcessor()
        processor.initialize_text_vectorizers(applicants_df, jobs_df)
        model = load_model(Config.MODEL_PATH, hidden_layer_1_size=Config.HIDDEN_SIZE, hidden_layer_2_size=Config.HIDDEN_SIZE//2)
        input_size = get_model_input_size(model)
        processor.scaler = MinMaxScaler()
        processor.scaler.fit(np.random.rand(10, input_size))
        
        logger.info("Componentes inicializados com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Erro fatal ao inicializar componentes: {e}\n{traceback.format_exc()}")
        return False

# --- 9. ROTAS DA API (ENDPOINTS) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        data = request.json
        job_id, candidate_id = str(data.get('job_id')), str(data.get('candidate_id'))
        logger.info(f"NOVA PREDIÇÃO: job_id={job_id}, candidate_id={candidate_id}")
        job_data = jobs_df[jobs_df['vaga_id'].astype(str) == job_id]
        candidate_data = applicants_df[applicants_df['candidato_id'].astype(str) == candidate_id]
        if job_data.empty or candidate_data.empty:
            return jsonify({'error': f'ID inválido: Vaga existe? {not job_data.empty}, Candidato existe? {not candidate_data.empty}'}), 404
        features = extract_single_prediction_features(job_data.iloc[0], candidate_data.iloc[0])
        if features is None: return jsonify({'error': 'Erro ao extrair features.'}), 500
        expected_size = get_model_input_size(model)
        if len(features) != expected_size:
            features = np.pad(features, (0, max(0, expected_size - len(features))), 'constant')[:expected_size]
        features_normalized = processor.scaler.transform([features])
        features_tensor = torch.FloatTensor(features_normalized)
        with torch.no_grad():
            prediction = model(features_tensor).item()
        total_time = time.time() - start_time
        recommendation = 'Alto potencial' if prediction > 0.7 else 'Médio potencial' if prediction > 0.3 else 'Baixo potencial'
        try:
            with get_db() as conn:
                conn.execute('INSERT INTO predictions (timestamp, job_id, candidate_id, success_probability, recommendation, features_json, total_time) VALUES (?, ?, ?, ?, ?, ?, ?)',
                             (datetime.now().isoformat(), job_id, candidate_id, prediction, recommendation, json.dumps(features.tolist()), total_time))
        except Exception as e:
            logger.error(f"Erro ao salvar predição no DB: {e}")
        return jsonify({'job_id': job_id, 'candidate_id': candidate_id, 'success_probability': round(prediction, 4), 'recommendation': recommendation})
    except Exception as e:
        logger.error(f"Erro na predição: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/evaluate_model', methods=['GET'])
def evaluate_model():
    try:
        logger.info("Iniciando avaliação do modelo...")
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_param("evaluation_date", datetime.now().isoformat())
            n_samples = min(100, len(prospects_df))
            if n_samples == 0: return jsonify({'error': 'Não há dados de prospects para avaliação.'}), 400
            sample_prospects = prospects_df.sample(n=n_samples, random_state=42)
            predictions, true_labels = [], []
            for _, prospect in sample_prospects.iterrows():
                try:
                    vaga_id, candidato_id = str(prospect['vaga_id']), str(prospect['codigo'])
                    job_data = jobs_df[jobs_df['vaga_id'].astype(str) == vaga_id]
                    candidate_data = applicants_df[applicants_df['candidato_id'].astype(str) == candidato_id]
                    if job_data.empty or candidate_data.empty: continue
                    situacao = str(prospect.get('situacao_candidado', '')).lower()
                    if 'aprovado' in situacao or 'contratado' in situacao: true_label = 1
                    elif 'rejeitado' in situacao or 'reprovado' in situacao: true_label = 0
                    else: continue
                    features = extract_single_prediction_features(job_data.iloc[0], candidate_data.iloc[0])
                    if features is None: continue
                    expected_size = get_model_input_size(model)
                    if len(features) != expected_size:
                        features = np.pad(features, (0, max(0, expected_size - len(features))), 'constant')[:expected_size]
                    features_normalized = processor.scaler.transform([features])
                    features_tensor = torch.FloatTensor(features_normalized)
                    with torch.no_grad():
                        pred = model(features_tensor).item()
                    predictions.append(pred)
                    true_labels.append(true_label)
                except Exception as e:
                    logger.error(f"Erro ao processar prospect para avaliação: {e}")
            if not predictions: return jsonify({'error': 'Nenhuma amostra de avaliação válida foi encontrada.'}), 400
            metrics = calculate_metrics(true_labels, predictions, 0.5)
            if not metrics: return jsonify({'error': 'Falha ao calcular métricas.'}), 500
            mlflow.log_metrics(metrics)
            return jsonify({'evaluation_metrics': metrics})
    except Exception as e:
        logger.error(f"Erro na avaliação: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Erro na avaliação: {str(e)}'}), 500

@app.route('/trigger_retraining', methods=['POST'])
def trigger_retraining():
    global model, processor
    MINIMUM_F1_SCORE_THRESHOLD = 0.75
    logger.info(f"GATE DE QUALIDADE: O F1-Score do novo modelo deve ser >= {MINIMUM_F1_SCORE_THRESHOLD}")
    request_data = request.json if request.json else {}
    hyperparameters = {'learning_rate': request_data.get('learning_rate', Config.LEARNING_RATE), 'epochs': request_data.get('epochs', Config.EPOCHS), 'hidden_layer_1_size': request_data.get('hidden_layer_1_size', Config.HIDDEN_SIZE), 'hidden_layer_2_size': request_data.get('hidden_layer_2_size', Config.HIDDEN_SIZE // 2), 'dropout_rate': request_data.get('dropout_rate', 0.2), 'tfidf_max_features': request_data.get('tfidf_max_features', 100)}
    logger.info(f"Requisição para retreino recebida com hiperparâmetros: {hyperparameters}")
    try:
        with mlflow.start_run(run_name="Continuous Retraining Candidate"):
            mlflow.log_params(hyperparameters)
            new_model_path, new_processor_instance = train_new_model(Config.DATA_PATHS, Config.MODEL_DIR, hyperparameters)
            if new_model_path is None or new_processor_instance is None:
                return jsonify({'status': 'failed', 'message': 'Erro durante o treinamento.'}), 500
            input_size = 2 * hyperparameters['tfidf_max_features'] + 1
            temp_new_model = load_model(new_model_path, input_size=input_size, **hyperparameters)
            y_true_for_eval, y_pred_for_eval = [], []
            for _, prospect in prospects_df.sample(n=min(100, len(prospects_df)), random_state=42).iterrows():
                situacao = str(prospect.get('situacao_candidado', '')).lower()
                if 'aprovado' in situacao or 'contratado' in situacao: true_label = 1
                elif 'rejeitado' in situacao or 'reprovado' in situacao: true_label = 0
                else: continue
                job_data = jobs_df[jobs_df['vaga_id'].astype(str) == str(prospect['vaga_id'])]
                candidate_data = applicants_df[applicants_df['candidato_id'].astype(str) == str(prospect['codigo'])]
                if job_data.empty or candidate_data.empty: continue
                features = extract_single_prediction_features(job_data.iloc[0], candidate_data.iloc[0])
                if features is None: continue
                features = np.pad(features, (0, max(0, input_size - len(features))), 'constant')[:input_size]
                features_normalized = new_processor_instance.scaler.transform([features])
                features_tensor = torch.FloatTensor(features_normalized)
                with torch.no_grad():
                    pred = temp_new_model(features_tensor).item()
                y_true_for_eval.append(true_label)
                y_pred_for_eval.append(pred)
            if not y_true_for_eval: raise Exception("Nenhum dado de avaliação válido encontrado.")
            retrain_metrics = calculate_metrics(y_true_for_eval, y_pred_for_eval, threshold=0.5)
            new_f1_score = retrain_metrics.get('f1_score', 0)
            logger.info(f"AVALIAÇÃO DO NOVO MODELO: F1-Score = {new_f1_score}")
            mlflow.log_metrics(retrain_metrics)
            if new_f1_score >= MINIMUM_F1_SCORE_THRESHOLD:
                logger.info(f"QUALIDADE APROVADA! F1-Score ({new_f1_score}) atinge o limite.")
                model, processor = temp_new_model, new_processor_instance
                mlflow.pytorch.log_model(pytorch_model=model, artifact_path="recommender_model")
                return jsonify({'status': 'success', 'message': f'Modelo APROVADO e atualizado. Novo F1-Score: {new_f1_score}'})
            else:
                logger.warning(f"QUALIDADE REPROVADA! F1-Score ({new_f1_score}) abaixo do limite.")
                mlflow.pytorch.log_model(pytorch_model=temp_new_model, artifact_path="rejected_model")
                return jsonify({'status': 'rejected', 'message': f'Modelo REPROVADO. Novo F1-Score: {new_f1_score}'}), 202
    except Exception as e:
        logger.error(f"Erro crítico no retreino: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'failed', 'error': str(e)}), 500

@app.route('/jobs')
def get_jobs():
    try:
        if jobs_df is None: return jsonify({'error': 'Dados de vagas não carregados.'}), 500
        vagas_com_prospects = prospects_df['vaga_id'].astype(str).unique()
        jobs_filtered = jobs_df[jobs_df['vaga_id'].astype(str).isin(vagas_com_prospects)]
        return jsonify(jobs_filtered[['vaga_id', 'titulo_vaga', 'cliente']].head(500).to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/candidates')
def get_candidates():
    try:
        if applicants_df is None: return jsonify({'error': 'Dados de candidatos não carregados.'}), 500
        candidatos_com_prospects = prospects_df['codigo'].astype(str).unique()
        candidates_filtered = applicants_df[applicants_df['candidato_id'].astype(str).isin(candidatos_com_prospects)]
        return jsonify(candidates_filtered[['candidato_id', 'nome', 'cargo_atual']].head(500).to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/matches')
def get_matches():
    try:
        if prospects_df is None: return jsonify({'error': 'Dados de prospects não carregados.'}), 500
        matches = []
        for _, prospect in prospects_df.head(1000).iterrows():
            vaga_id, candidato_codigo = str(prospect.get('vaga_id')), str(prospect.get('codigo'))
            job_data = jobs_df[jobs_df['vaga_id'].astype(str) == vaga_id]
            candidate_data = applicants_df[applicants_df['candidato_id'].astype(str) == candidato_codigo]
            if not job_data.empty and not candidate_data.empty:
                matches.append({'vaga_id': vaga_id, 'candidato_id': candidato_codigo, 'titulo_vaga': job_data.iloc[0].get('titulo_vaga', ''), 'nome_candidato': candidate_data.iloc[0].get('nome', '')})
        if not matches: return jsonify({'message': 'Nenhum match encontrado nos dados atuais.'}), 200
        return jsonify(matches)
    except Exception as e:
        return jsonify({'error': f'Erro interno ao buscar matches: {str(e)}'}), 500

@app.route('/health')
def health():
    db_ok = False
    try:
        get_db().execute('SELECT 1'); db_ok = True
    except Exception as e:
        logger.error(f"Health check do DB falhou: {e}")
    data_ok = all([df is not None and not df.empty for df in [jobs_df, prospects_df, applicants_df]])
    return jsonify({'status': 'ok' if all([processor, model, db_ok, data_ok]) else 'error', 'components': {'processor': processor is not None, 'model': model is not None, 'database': 'ok' if db_ok else 'error', 'data_loaded_and_populated': data_ok}, 'data_counts': {'jobs': len(jobs_df) if jobs_df is not None else 0, 'prospects': len(prospects_df) if prospects_df is not None else 0, 'applicants': len(applicants_df) if applicants_df is not None else 0}})

# --- 10. BLOCO DE EXECUÇÃO PRINCIPAL ---
def create_dummy_files_if_needed():
    IS_ON_RENDER = os.environ.get('RENDER')
    if IS_ON_RENDER: return
    if not os.path.exists(Config.MODEL_PATH):
        dummy_model = RecommenderModel()
        torch.save(dummy_model.state_dict(), Config.MODEL_PATH)
        logger.warning(f"Modelo dummy criado em: {Config.MODEL_PATH}")
    if not os.path.exists(Config.DATA_PATHS['jobs']):
        # Adicione aqui a lógica para criar ficheiros JSON dummy se necessário
        logger.warning("Ficheiros de dados dummy não encontrados. Crie-os ou a aplicação pode falhar.")

# --- 11. INICIALIZAÇÃO DA APLICAÇÃO (ESCOPO GLOBAL) ---
logger.info("Iniciando a configuração da aplicação...")
init_db()
initialization_success = initialize_components()

if __name__ == '__main__':
    if initialization_success:
        logger.info("Iniciando servidor de desenvolvimento Flask...")
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        logger.error("Falha na inicialização. O servidor de desenvolvimento não será iniciado.")
