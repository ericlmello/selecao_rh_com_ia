#!/bin/bash
set -e

echo "🚀 Iniciando ML ATS..."

mkdir -p /app/data /app/models /app/logs

# Criar dados exemplo se não existirem
if [ ! -f "/app/data/jobs.json" ]; then
    echo " Criando dados exemplo..."
    python -c "
import json, os
os.makedirs('/app/data', exist_ok=True)

jobs = [{'vaga_id': 1, 'titulo_vaga': 'Dev Python', 'descricao': 'Vaga Python', 'cliente': 'TechCorp'}]
applicants = [{'candidato_id': 1, 'nome': 'João Silva', 'cargo_atual': 'Dev', 'habilidades': 'Python'}]
prospects = [{'vaga_id': 1, 'codigo': 1, 'situacao_candidado': 'aprovado', 'comentario': 'OK'}]

with open('/app/data/jobs.json', 'w') as f: json.dump(jobs, f)
with open('/app/data/applicants.json', 'w') as f: json.dump(applicants, f)
with open('/app/data/prospects.json', 'w') as f: json.dump(prospects, f)
print('✅ Dados criados!')
"
fi

# Criar modelo dummy se não existir
if [ ! -f "/app/models/model.pth" ]; then
    echo " Criando modelo dummy..."
    python -c "
import torch, torch.nn as nn, os
class Model(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(201, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): 
        return self.sigmoid(self.fc2(torch.relu(self.fc1(x))))
os.makedirs('/app/models', exist_ok=True)
torch.save(Model().state_dict(), '/app/models/model.pth')
print(' Modelo criado!')
"
fi

echo " Pronto! Executando: $@"
exec "$@"