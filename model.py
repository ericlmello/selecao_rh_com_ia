import torch
import torch.nn as nn
import os

class JobMatchingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(JobMatchingModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

def save_model(model, path):
    """Salva o modelo treinado"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'output_size': model.output_size
    }, path)
    print(f"Modelo salvo em: {path}")

def safe_print_model(message):
    """Print seguro para mensagens do modelo"""
    try:
        clean_message = ''.join(char for char in str(message) if ord(char) < 65536)
        clean_message = clean_message.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        print(clean_message)
    except:
        print("Mensagem nao pode ser exibida devido a caracteres problematicos")

def load_model(path):
    """Carrega o modelo - versão específica para seu formato de state_dict"""
    try:
        print(f"🔄 Carregando modelo de: {path}")
        
        # Verificar se arquivo existe
        if not os.path.exists(path):
            print(f"❌ Arquivo não encontrado: {path}")
            raise FileNotFoundError(f"Modelo não encontrado em {path}")
        
        # Carregar o checkpoint
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint carregado: {type(checkpoint)}")
        
        # Seu modelo foi salvo como OrderedDict (state_dict direto)
        if hasattr(checkpoint, 'keys') and 'fc1.weight' in checkpoint:
            print("📋 Detectado: State dict direto (OrderedDict)")
            
            # Extrair dimensões da estrutura do modelo salvo
            fc1_weight = checkpoint['fc1.weight']
            fc1_bias = checkpoint['fc1.bias'] 
            fc2_weight = checkpoint['fc2.weight']
            
            input_size = fc1_weight.shape[1]    # 201
            hidden_size = fc1_weight.shape[0]   # 128
            hidden2_size = fc2_weight.shape[0]  # 64 (verificação)
            output_size = 1  # Sempre 1 para classificação binária
            
            print(f"📏 Dimensões detectadas:")
            print(f"   Input size: {input_size}")
            print(f"   Hidden size: {hidden_size}")
            print(f"   Hidden2 size: {hidden2_size}")
            print(f"   Output size: {output_size}")
            
            # Validar estrutura
            expected_layers = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
            missing_layers = [layer for layer in expected_layers if layer not in checkpoint]
            
            if missing_layers:
                print(f"⚠️ Camadas faltando: {missing_layers}")
            else:
                print("✅ Todas as camadas presentes")
            
            # Criar modelo com as dimensões corretas
            model = JobMatchingModel(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size
            )
            
            # Carregar os pesos
            model.load_state_dict(checkpoint)
            print("✅ Pesos carregados com sucesso!")
            
        # Caso seja um dicionário com metadata (formato alternativo)
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("📋 Detectado: Dicionário com metadata")
            
            input_size = checkpoint.get('input_size', 201)
            hidden_size = checkpoint.get('hidden_size', 128)
            output_size = checkpoint.get('output_size', 1)
            
            print(f"📏 Parâmetros do checkpoint: input={input_size}, hidden={hidden_size}, output={output_size}")
            
            model = JobMatchingModel(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Pesos carregados com sucesso!")
            
        else:
            print("❌ Formato de checkpoint não reconhecido")
            print(f"Tipo: {type(checkpoint)}")
            if hasattr(checkpoint, 'keys'):
                print(f"Chaves: {list(checkpoint.keys())[:10]}")
            raise ValueError("Formato de checkpoint inválido")
        
        # Configurar modelo para inferência
        model.eval()
        
        # Garantir que atributos estão disponíveis para o app.py
        if not hasattr(model, 'input_size'):
            model.input_size = input_size
        if not hasattr(model, 'hidden_size'):
            model.hidden_size = hidden_size
        if not hasattr(model, 'output_size'):
            model.output_size = output_size
            
        print(f"🎯 Modelo pronto!")
        print(f"   Arquitetura: {model.input_size} → {model.hidden_size} → {model.hidden_size//2} → {model.output_size}")
        print(f"   Parâmetros: {sum(p.numel() for p in model.parameters())}")
        print(f"   Modo: {'Treino' if model.training else 'Avaliação'}")
        
        return model
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("🆘 Criando modelo com pesos aleatórios...")
        
        # Fallback: modelo com arquitetura padrão
        model = JobMatchingModel(input_size=201, hidden_size=128, output_size=1)
        model.eval()
        
        # Adicionar atributos necessários
        model.input_size = 201
        model.hidden_size = 128
        model.output_size = 1
        
        print("⚠️ ATENÇÃO: Usando modelo com pesos NÃO treinados!")
        return model
