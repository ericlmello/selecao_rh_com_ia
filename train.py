'''
Treino do modelo
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_processing import DataProcessor
from model import JobMatchingModel, save_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Defina as configurações diretamente aqui
INPUT_SIZE = 201  # Tamanho do vetor de features após pré-processamento (atualizado baseado nos dados reais)
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
MODEL_PATH = 'model.pth'

FEATURES = {
    'candidate': ['academic_level', 'english_level', 'spanish_level', 'technical_skills', 'work_area'],
    'job': ['is_sap', 'professional_level', 'required_languages', 'technical_competencies', 'client'],
    'interaction': ['comments_length', 'match_technical', 'match_language']
}

TARGET = 'hired'

# Caminhos dos arquivos - CORRIGIDO PARA .csv
DATA_PATHS = {
    'jobs': 'C:\\Users\\e178454\\Desktop\\partic\\FIAP ML\\selecao\\data\\jobs.csv',
    'prospects': 'C:\\Users\\e178454\\Desktop\\partic\\FIAP ML\\selecao\\data\\prospects.csv',
    'applicants': 'C:\\Users\\e178454\\Desktop\\partic\\FIAP ML\\selecao\\data\\applicants.csv'
}

def train_model():
    # Inicialize o processor primeiro para extrair os dados
    processor = DataProcessor(DATA_PATHS)
    
    # Carregue e processe os dados
    print("Loading and preprocessing data...")
    jobs, prospects, applicants = processor.load_data()
    X, y = processor.extract_features(jobs, prospects, applicants)
    
    # Calcula automaticamente o tamanho das features
    actual_input_size = X.shape[1]
    print(f"Tamanho das features detectado: {actual_input_size}")
    
    # Inicialize o modelo com o tamanho correto
    model = JobMatchingModel(actual_input_size, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, test_loader = processor.create_dataloaders(X, y, batch_size=BATCH_SIZE)
    
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(test_loader.dataset)} samples")
    print("Starting training...")
    
    # Loop de treinamento
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        all_targets = []
        all_predictions = []
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            predicted = (outputs > 0.5).float()
            all_targets.extend(targets.numpy())
            all_predictions.extend(predicted.numpy())
        
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_predictions)
        train_prec = precision_score(all_targets, all_predictions, zero_division=0)
        train_rec = recall_score(all_targets, all_predictions, zero_division=0)
        train_f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        # Validação
        model.eval()
        test_loss = 0.0
        test_targets = []
        test_predictions = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * features.size(0)
                predicted = (outputs > 0.5).float()
                test_targets.extend(targets.numpy())
                test_predictions.extend(predicted.numpy())
        
        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(test_targets, test_predictions)
        test_prec = precision_score(test_targets, test_predictions, zero_division=0)
        test_rec = recall_score(test_targets, test_predictions, zero_division=0)
        test_f1 = f1_score(test_targets, test_predictions, zero_division=0)
        
        # Imprimir as métricas
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}')
        print(f'Train Precision: {train_prec:.4f} | Test Precision: {test_prec:.4f}')
        print(f'Train Recall: {train_rec:.4f} | Test Recall: {test_rec:.4f}')
        print(f'Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}')
        print('-' * 50)
    
    # Salve o modelo treinado
    save_model(model, MODEL_PATH)
    print(f'\nModel saved to {MODEL_PATH}')
    
    return model

if __name__ == '__main__':
    train_model()
