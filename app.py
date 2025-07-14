# -*- coding: utf-8 -*-
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
from datetime import datetime
import traceback
from pathlib import Path

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

# --- 3. CONFIGURAÇÃO DE AMBIENTE (LOCAL vs RENDER) ---
# Importa a configuração dinâmica do config.py
# Este arquivo agora lida com os caminhos para ambiente local e de produção (Render)
try:
    from config import Config, DATABASE_PATH, MLRUNS_DIR
except ImportError:
    # Fallback caso config.py não seja encontrado
    print("AVISO: 'config.py' não encontrado. A aplicação pode não funcionar corretamente.")
    # Definições mínimas para evitar que a aplicação quebre na inicialização
    class Config:
        DATA_PATHS = {}
        MODEL_PATH = 'models/recommender_model.pth'
        MODEL_DIR = 'models'
    DATABASE_PATH = 'predictions.db'
    MLRUNS_DIR = 'mlruns'


# --- 4. CONFIGURAÇÃO DE LOGGING E MLFLOW ---
# Configura o logging para registrar informações em um arquivo e no console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configura o MLflow para usar o diretório correto (local ou persistente no Render)
try:
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    logger.info(f"MLflow tracking URI configurado para: {MLRUNS_DIR}")
except Exception as e:
    logger.error(f"Falha ao configurar o tracking URI do MLflow: {e}")


# --- 5. INICIALIZAÇÃO DA APLICAÇÃO FLASK E BANCO DE DADOS ---
app = Flask(__name__)
DATABASE = DATABASE_PATH

def get_db():
    """Abre uma nova conexão com o banco de dados."""
    db = sqlite3.connect(DATABASE, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Inicializa o banco de dados criando as tabelas necessárias."""
    with app.app_context():
        db = get_db()
        with db as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    job_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    success_probability REAL NOT NULL, recommendation TEXT,
                    features_json TEXT, total_time REAL
                );
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    total_samples INTEGER, avg_prediction REAL, metrics_json TEXT
                );
            ''')
        logger.info(f"Banco de dados SQLite inicializado em: {DATABASE}")

# --- 6. VARIÁVEIS GLOBAIS E CLASSES DE MODELO/PROCESSADOR ---
# Serão inicializadas pela função initialize_components.
processor = None
model = None
jobs_df = None
prospects_df = None
applicants_df = None

class RecommenderModel(torch.nn.Module):
    def __init__(self, input_size=201, hidden_layer_1_size=128, hidden_layer_2_size=64, dropout_rate=0.2):
        super(RecommenderModel, self).__init__()
        self.input_size = input_size
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_layer_1_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_layer_1_size, hidden_layer_2_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_2_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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
                    texts = applicants_df[col].dropna().astype(str).head(100)
                    all_texts.extend([safe_clean_text(t) for t in texts if t])
            for col in text_columns_jobs:
                if col in jobs_df.columns:
                    texts = jobs_df[col].dropna().astype(str).head(100)
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

def safe_clean_text(text):
    """Limpa texto removendo caracteres problemáticos."""
    try:
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        clean_chars = [char for char in text if char.isprintable() or char.isspace()]
        return ''.join(clean_chars)
    except Exception as e:
        logger.error(f"Erro ao limpar texto: {e}")
        return ""

def safe_load_csv_with_surrogate_fix(file_path):
    """Carrega CSV com fallback de encoding."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"Falha no UTF-8 para {file_path}. Tentando latin-1...")
        return pd.read_csv(file_path, encoding='latin-1')
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar {file_path}: {e}")
        return None

def clean_dataframe_surrogates(df):
    """Limpa caracteres problemáticos de um DataFrame."""
    if df is None:
        return None
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).apply(safe_clean_text)
    return df

def get_model_input_size(model_instance):
    """Determina o tamanho de entrada do modelo."""
    if hasattr(model_instance, 'input_size'):
        return model_instance.input_size
    logger.warning("Não foi possível determinar input_size do modelo, usando valor padrão: 201")
    return 201

def load_model(model_path, **kwargs):
    """Carrega o modelo PyTorch salvo."""
    model_instance = RecommenderModel(**kwargs)
    try:
        model_instance.load_state_dict(torch.load(model_path))
        model_instance.eval()
        logger.info(f"Modelo carregado com sucesso de: {model_path}")
        return model_instance
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo de {model_path}: {e}")
        raise

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calcula e retorna um dicionário com métricas de classificação."""
    try:
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        y_pred_binary = (y_pred_np >= threshold).astype(int)
        return {
            'accuracy': round(accuracy_score(y_true_np, y_pred_binary), 3),
            'precision': round(precision_score(y_true_np, y_pred_binary, zero_division=0), 3),
            'recall': round(recall_score(y_true_np, y_pred_binary, zero_division=0), 3),
            'f1_score': round(f1_score(y_true_np, y_pred_binary, zero_division=0), 3),
            'threshold': threshold
        }
    except Exception as e:
        logger.error(f"Erro ao calcular métricas: {e}")
        return None

def extract_single_prediction_features(prospect, job_data, candidate_data):
    """Extrai features para uma única predição."""
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

def extract_text_from_candidate(candidate_data):
    """Extrai texto do candidato."""
    return " ".join(safe_clean_text(candidate_data.get(field, '')) for field in ['campo_extra_cv_pt', 'campo_extra_cv_en', 'habilidades'])

def extract_text_from_job(job_data):
    """Extrai texto da vaga."""
    return " ".join(safe_clean_text(job_data.get(field, '')) for field in ['titulo_vaga', 'descricao'])

def train_new_model(data_paths_config, output_model_dir, hyperparameters):
    """
    Função simulada para treinar um novo modelo com hiperparâmetros configuráveis.
    """
    logger.info("Iniciando o treinamento de um novo modelo...")
    logger.info(f"Hiperparâmetros de treino: {hyperparameters}")
    
    learning_rate = hyperparameters.get('learning_rate', 0.001)
    epochs = hyperparameters.get('epochs', 5)
    hidden_layer_1_size = hyperparameters.get('hidden_layer_1_size', 128)
    hidden_layer_2_size = hyperparameters.get('hidden_layer_2_size', 64)
    dropout_rate = hyperparameters.get('dropout_rate', 0.2)
    tfidf_max_features = hyperparameters.get('tfidf_max_features', 100)

    try:
        new_jobs_df = safe_load_csv_with_surrogate_fix(data_paths_config['jobs'])
        new_applicants_df = safe_load_csv_with_surrogate_fix(data_paths_config['applicants'])

        if any(df is None for df in [new_jobs_df, new_applicants_df]):
            raise Exception("Falha ao carregar novos dados para retreino.")

        temp_processor = SimpleProcessor()
        temp_processor.initialize_text_vectorizers(new_applicants_df, new_jobs_df, max_features=tfidf_max_features)
        
        input_size_for_training = 2 * tfidf_max_features + 1
        temp_processor.scaler = MinMaxScaler()
        temp_processor.scaler.fit(np.random.rand(10, input_size_for_training))

        new_model = RecommenderModel(
            input_size=input_size_for_training,
            hidden_layer_1_size=hidden_layer_1_size,
            hidden_layer_2_size=hidden_layer_2_size,
            dropout_rate=dropout_rate
        )
        
        optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCELoss()

        logger.info(f"Simulando treinamento do modelo por {epochs} épocas...")
        for epoch in range(epochs):
            dummy_features = torch.randn(10, input_size_for_training)
            dummy_labels = torch.randint(0, 2, (10, 1)).float()
            optimizer.zero_grad()
            outputs = new_model(dummy_features)
            loss = criterion(outputs, dummy_labels)
            loss.backward()
            optimizer.step()
        logger.info("Simulação de treinamento concluída.")

        model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        full_output_path = os.path.join(output_model_dir, model_filename)
        torch.save(new_model.state_dict(), full_output_path)
        logger.info(f"Novo modelo salvo em: {full_output_path}")

        return full_output_path, temp_processor

    except Exception as e:
        logger.error(f"Erro durante o treinamento do novo modelo: {e}")
        logger.error(f"Traceback completo do treinamento: {traceback.format_exc()}")
        return None, None

# --- 8. FUNÇÃO DE INICIALIZAÇÃO PRINCIPAL ---

def initialize_components():
    """Inicializa todos os componentes necessários: dados, processador e modelo."""
    global processor, model, jobs_df, prospects_df, applicants_df
    try:
        logger.info("Inicializando componentes da aplicação...")
        
        jobs_df = safe_load_csv_with_surrogate_fix(Config.DATA_PATHS['jobs'])
        prospects_df = safe_load_csv_with_surrogate_fix(Config.DATA_PATHS['prospects'])
        applicants_df = safe_load_csv_with_surrogate_fix(Config.DATA_PATHS['applicants'])
        
        if any(df is None for df in [jobs_df, prospects_df, applicants_df]):
            raise FileNotFoundError("Falha crítica: Um ou mais arquivos de dados não puderam ser carregados.")
        
        jobs_df = clean_dataframe_surrogates(jobs_df)
        prospects_df = clean_dataframe_surrogates(prospects_df)
        applicants_df = clean_dataframe_surrogates(applicants_df)
        
        processor = SimpleProcessor()
        processor.initialize_text_vectorizers(applicants_df, jobs_df)
        
        # Usa os hiperparâmetros da config para carregar o modelo
        model = load_model(Config.MODEL_PATH, hidden_layer_1_size=Config.HIDDEN_SIZE, hidden_layer_2_size=Config.HIDDEN_SIZE//2)
        input_size = get_model_input_size(model)
        
        processor.scaler = MinMaxScaler()
        dummy_data = np.random.rand(10, input_size)
        processor.scaler.fit(dummy_data)
        
        logger.info("Componentes inicializados com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Erro ao inicializar componentes: {e}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
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
        job_id = str(data.get('job_id'))
        candidate_id = str(data.get('candidate_id'))
        
        logger.info(f"NOVA PREDIÇÃO INICIADA - job_id={job_id}, candidate_id={candidate_id}")
        
        job_data = jobs_df[jobs_df['vaga_id'].astype(str) == job_id]
        candidate_data = applicants_df[applicants_df['candidato_id'].astype(str) == candidate_id]

        if job_data.empty or candidate_data.empty:
            error_msg = f'ID inválido: job_id={job_id} (existe: {not job_data.empty}), candidate_id={candidate_id} (existe: {not candidate_data.empty})'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 404
            
        features = extract_single_prediction_features(None, job_data.iloc[0], candidate_data.iloc[0])
        if features is None:
            return jsonify({'error': 'Erro ao extrair features.'}), 500
        
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
            db = get_db()
            with db as conn:
                conn.execute(
                    'INSERT INTO predictions (timestamp, job_id, candidate_id, success_probability, recommendation, features_json, total_time) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (datetime.now().isoformat(), job_id, candidate_id, prediction, recommendation, json.dumps(features.tolist()), total_time)
                )
            logger.info("Predição salva no banco de dados SQLite.")
        except Exception as e:
            logger.error(f"Erro ao salvar predição no banco de dados: {e}")

        result = {
            'job_id': job_id,
            'candidate_id': candidate_id,
            'success_probability': round(prediction, 4),
            'recommendation': recommendation
        }
        logger.info(f"Predição concluída: probabilidade={prediction:.4f}, tempo={total_time:.3f}s.")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}.")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/evaluate_model', methods=['GET'])
def evaluate_model():
    global model
    try:
        logger.info("Iniciando avaliação do modelo...")
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_param("evaluation_date", datetime.now().isoformat())
            
            n_samples = min(100, len(prospects_df))
            if n_samples == 0:
                return jsonify({'error': 'Não há dados de prospects para avaliação.'}), 400
            
            sample_prospects = prospects_df.sample(n=n_samples, random_state=42)
            predictions = []
            true_labels = []
            
            for _, prospect in sample_prospects.iterrows():
                try:
                    vaga_id = str(prospect['vaga_id'])
                    candidato_id = str(prospect['codigo'])
                    
                    job_data = jobs_df[jobs_df['vaga_id'].astype(str) == vaga_id]
                    candidate_data = applicants_df[applicants_df['candidato_id'].astype(str) == candidato_id]

                    if job_data.empty or candidate_data.empty:
                        continue

                    situacao = str(prospect.get('situacao_candidado', '')).lower()
                    if 'aprovado' in situacao or 'contratado' in situacao:
                        true_label = 1
                    elif 'rejeitado' in situacao or 'reprovado' in situacao:
                        true_label = 0
                    else:
                        continue
                    
                    features = extract_single_prediction_features(prospect, job_data.iloc[0], candidate_data.iloc[0])
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
                    continue
            
            if not predictions or not true_labels:
                return jsonify({'error': 'Nenhuma amostra de avaliação válida foi encontrada.'}), 400

            metrics = calculate_metrics(true_labels, predictions, 0.5)
            if not metrics:
                 return jsonify({'error': 'Falha ao calcular métricas.'}), 500

            mlflow.log_metric('eval_accuracy', metrics.get('accuracy', 0))
            mlflow.log_metric('eval_f1_score', metrics.get('f1_score', 0))
            
            logger.info(f"Avaliação concluída: {len(predictions)} amostras.")
            return jsonify({'evaluation_metrics': metrics})
            
    except Exception as e:
        logger.error(f"Erro na avaliação: {e}.\n{traceback.format_exc()}")
        return jsonify({'error': f'Erro na avaliação: {str(e)}'}), 500

@app.route('/trigger_retraining', methods=['POST'])
def trigger_retraining():
    global model, processor
    MINIMUM_F1_SCORE_THRESHOLD = 0.75
    logger.info(f"GATE DE QUALIDADE: O F1-Score do novo modelo deve ser >= {MINIMUM_F1_SCORE_THRESHOLD}")

    request_data = request.json if request.json else {}
    hyperparameters = {
        'learning_rate': request_data.get('learning_rate', Config.LEARNING_RATE),
        'epochs': request_data.get('epochs', Config.EPOCHS),
        'hidden_layer_1_size': request_data.get('hidden_layer_1_size', Config.HIDDEN_SIZE),
        'hidden_layer_2_size': request_data.get('hidden_layer_2_size', Config.HIDDEN_SIZE // 2),
        'dropout_rate': request_data.get('dropout_rate', 0.2),
        'tfidf_max_features': request_data.get('tfidf_max_features', 100)
    }

    logger.info(f"Requisição para retreino contínuo recebida com hiperparâmetros: {hyperparameters}")

    try:
        with mlflow.start_run(run_name="Continuous Retraining Candidate"):
            mlflow.log_params(hyperparameters)
            
            new_model_path, new_processor_instance = train_new_model(Config.DATA_PATHS, Config.MODEL_DIR, hyperparameters)

            if new_model_path is None or new_processor_instance is None:
                mlflow.log_param("retraining_status", "Failed - Training Error")
                return jsonify({'status': 'failed', 'message': 'Erro durante o treinamento do novo modelo.'}), 500

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
                
                features = extract_single_prediction_features(prospect, job_data.iloc[0], candidate_data.iloc[0])
                if features is None: continue
                
                features = np.pad(features, (0, max(0, input_size - len(features))), 'constant')[:input_size]
                features_normalized = new_processor_instance.scaler.transform([features])
                features_tensor = torch.FloatTensor(features_normalized)
                
                with torch.no_grad():
                    pred = temp_new_model(features_tensor).item()
                
                y_true_for_eval.append(true_label)
                y_pred_for_eval.append(pred)

            if not y_true_for_eval:
                 raise Exception("Nenhum dado de avaliação válido encontrado para o novo modelo.")

            retrain_metrics = calculate_metrics(y_true_for_eval, y_pred_for_eval, threshold=0.5)
            new_f1_score = retrain_metrics.get('f1_score', 0)
            
            logger.info(f"AVALIAÇÃO DO NOVO MODELO: F1-Score = {new_f1_score}")
            mlflow.log_metrics(retrain_metrics)

            if new_f1_score >= MINIMUM_F1_SCORE_THRESHOLD:
                logger.info(f"QUALIDADE APROVADA! F1-Score ({new_f1_score}) atinge o limite.")
                mlflow.log_param("retraining_status", "Success and Deployed")
                model, processor = temp_new_model, new_processor_instance
                mlflow.pytorch.log_model(pytorch_model=model, artifact_path="recommender_model")
                return jsonify({'status': 'success', 'message': f'Modelo APROVADO e atualizado. Novo F1-Score: {new_f1_score}'})
            else:
                logger.warning(f"QUALIDADE REPROVADA! F1-Score ({new_f1_score}) abaixo do limite.")
                mlflow.log_param("retraining_status", "Success but Rejected")
                mlflow.pytorch.log_model(pytorch_model=temp_new_model, artifact_path="rejected_model")
                return jsonify({'status': 'rejected', 'message': f'Modelo REPROVADO. Novo F1-Score: {new_f1_score}'}), 202

    except Exception as e:
        logger.error(f"Erro crítico no retreino contínuo: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'failed', 'error': f'Erro no retreino contínuo: {str(e)}'}), 500

@app.route('/jobs')
def get_jobs():
    try:
        vagas_com_prospects = prospects_df['vaga_id'].astype(str).unique()
        jobs_filtered = jobs_df[jobs_df['vaga_id'].astype(str).isin(vagas_com_prospects)]
        jobs_list = jobs_filtered[['vaga_id', 'titulo_vaga', 'cliente']].head(500).to_dict(orient='records')
        return jsonify(jobs_list)
    except Exception as e:
        logger.error(f'Erro ao buscar vagas: {str(e)}.')
        return jsonify({'error': str(e)}), 500

@app.route('/candidates')
def get_candidates():
    try:
        candidatos_com_prospects = prospects_df['codigo'].astype(str).unique()
        candidates_filtered = applicants_df[applicants_df['candidato_id'].astype(str).isin(candidatos_com_prospects)]
        candidates_list = candidates_filtered[['candidato_id', 'nome', 'cargo_atual']].head(500).to_dict(orient='records')
        return jsonify(candidates_list)
    except Exception as e:
        logger.error(f'Erro ao buscar candidatos: {str(e)}.')
        return jsonify({'error': str(e)}), 500

@app.route('/matches')
def get_matches():
    try:
        max_matches = 1000
        prospects_sample = prospects_df.head(max_matches)
        matches = []
        for _, prospect in prospects_sample.iterrows():
            vaga_id = str(prospect['vaga_id'])
            candidato_codigo = str(prospect['codigo'])
            job_data = jobs_df[jobs_df['vaga_id'].astype(str) == vaga_id]
            candidate_data = applicants_df[applicants_df['candidato_id'].astype(str) == candidato_codigo]
            if not job_data.empty and not candidate_data.empty:
                matches.append({
                    'vaga_id': vaga_id,
                    'candidato_id': candidato_codigo,
                    'titulo_vaga': safe_clean_text(job_data.iloc[0]['titulo_vaga']),
                    'nome_candidato': safe_clean_text(candidate_data.iloc[0]['nome']),
                    'situacao': safe_clean_text(prospect.get('situacao_candidado', 'N/A'))
                })
        return jsonify(matches)
    except Exception as e:
        logger.error(f'Erro ao buscar matches: {str(e)}.')
        return jsonify({'error': f'Erro ao buscar matches: {str(e)}'}), 500

@app.route('/health')
def health():
    db_ok = False
    try:
        db = get_db()
        db.execute('SELECT 1')
        db_ok = True
    except Exception as e:
        logger.error(f"Health check do banco de dados falhou: {e}")
    status = {
        'status': 'ok' if all([processor, model, db_ok, jobs_df is not None]) else 'error',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'processor': processor is not None,
            'model': model is not None,
            'database': 'ok' if db_ok else 'error',
            'data_loaded': all([d is not None for d in [jobs_df, prospects_df, applicants_df]])
        }
    }
    return jsonify(status)

# --- 10. BLOCO DE EXECUÇÃO PRINCIPAL ---

def initialize_persistent_data():
    """Copia os dados iniciais para o disco persistente do Render."""
    IS_ON_RENDER = os.environ.get('RENDER')
    if not IS_ON_RENDER:
        return
    local_data_dir = 'data'
    # Garante que o diretório de dados persistente exista
    persistent_data_dir = os.path.dirname(Config.DATA_PATHS['jobs'])
    os.makedirs(persistent_data_dir, exist_ok=True)
    
    if os.path.exists(local_data_dir):
        for filename in os.listdir(local_data_dir):
            local_path = os.path.join(local_data_dir, filename)
            persistent_path = os.path.join(persistent_data_dir, filename)
            if not os.path.exists(persistent_path):
                shutil.copy2(local_path, persistent_path)
                logger.info(f"Copiado dado inicial '{filename}' para o disco.")

def create_dummy_files_if_needed():
    """Cria arquivos de exemplo para desenvolvimento local."""
    IS_ON_RENDER = os.environ.get('RENDER')
    if IS_ON_RENDER:
        return
    
    if not os.path.exists(Config.MODEL_PATH):
        dummy_model = RecommenderModel(hidden_layer_1_size=Config.HIDDEN_SIZE, hidden_layer_2_size=Config.HIDDEN_SIZE//2)
        torch.save(dummy_model.state_dict(), Config.MODEL_PATH)
        logger.warning(f"Modelo dummy criado em: {Config.MODEL_PATH}")

    if not os.path.exists(Config.DATA_PATHS['jobs']):
        pd.DataFrame({
            'vaga_id': ['4534'], 'titulo_vaga': ['Engenheiro de Teste'], 'descricao': ['Testar software'], 'cliente': ['Empresa Teste']
        }).to_csv(Config.DATA_PATHS['jobs'], index=False)
        pd.DataFrame({
            'candidato_id': ['11132'], 'nome': ['Candidato Teste'], 'cargo_atual': ['Analista'], 'campo_extra_cv_pt': ['Python'], 'campo_extra_cv_en': [''], 'habilidades': ['SQL']
        }).to_csv(Config.DATA_PATHS['applicants'], index=False)
        pd.DataFrame({
            'vaga_id': ['4534'], 'codigo': ['11132'], 'comentario': [''], 'situacao_candidado': ['pendente']
        }).to_csv(Config.DATA_PATHS['prospects'], index=False)
        logger.warning("Arquivos de dados dummy criados.")

if __name__ == '__main__':
    logger.info("Iniciando aplicação Flask...")
    
    initialize_persistent_data()
    create_dummy_files_if_needed()
    init_db()

    if initialize_components():
        logger.info("Aplicação pronta!")
        # app.run é ideal para desenvolvimento local. O Gunicorn será usado no Render.
        app.run(debug=False, host='127.0.0.1', port=5000)
    else:
        logger.error("Falha na inicialização. Verifique os erros acima.")
