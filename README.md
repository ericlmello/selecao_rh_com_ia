Sistema de Recomendação de Candidatos com IA
Este projeto é uma aplicação web completa que utiliza Machine Learning para automatizar e otimizar o processo de seleção de candidatos, calculando a probabilidade de sucesso entre perfis e vagas de emprego.

A solução é exposta através de uma API RESTful robusta, inclui monitoramento de modelos com MLflow, um portão de qualidade para retreino contínuo e está pronta para deploy na nuvem com o Render.

📋 Funcionalidades Principais
Matching com IA: Utiliza um modelo de rede neural (PyTorch) para analisar semanticamente currículos e descrições de vagas, gerando uma pontuação de compatibilidade.

API RESTful Completa: Oferece endpoints para predições, avaliação de modelo, retreino, consulta de dados e verificação de saúde do sistema.

Download Automático de Dados: A aplicação descarrega e descompacta os ficheiros de dados necessários a partir do Google Drive na inicialização, mantendo o repositório leve.

Monitoramento com MLflow: Integração nativa com MLflow para registar execuções, parâmetros e métricas de performance do modelo.

Retreino com Portão de Qualidade: Permite o retreino de novos modelos com hiperparâmetros personalizáveis e só promove um novo modelo para produção se a sua performance (F1-Score) atingir um limite mínimo predefinido.

Pronto para Deploy: Configurado para deploy fácil na plataforma Render usando um ficheiro render.yaml e Gunicorn como servidor de produção.

Testes Automatizados: Inclui uma suíte de testes com Pytest para garantir a qualidade e a estabilidade do código.

🛠️ Tecnologias Utilizadas
Backend: Flask, Gunicorn

Machine Learning: PyTorch, Scikit-learn

Manipulação de Dados: Pandas, NumPy

Monitoramento: MLflow

Utilitários: gdown (para downloads do Google Drive)

Banco de Dados: SQLite

🚀 Como Executar o Projeto Localmente
Pré-requisitos
Python 3.9+

Git

1. Clonar o Repositório
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

2. Criar e Ativar um Ambiente Virtual
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. Instalar as Dependências
pip install -r requirements.txt

4. Configurar os Dados
O projeto está configurado para descarregar os dados do Google Drive automaticamente. Certifique-se de que os IDs dos seus ficheiros estão corretos no ficheiro config.py.

5. Executar a Aplicação
python app.py

A aplicação estará disponível em http://127.0.0.1:5000.

⚙️ Endpoints da API
A seguir, os principais endpoints disponíveis:

Rota

Método

Descrição

/predict

POST

Retorna a predição de compatibilidade para uma vaga e um candidato.

/evaluate_model

GET

Avalia a performance do modelo atual e regista as métricas no MLflow.

/trigger_retraining

POST

Inicia o processo de retreino de um novo modelo.

/health

GET

Verifica a saúde da aplicação e dos seus componentes.

/jobs

GET

Lista as vagas disponíveis.

/candidates

GET

Lista os candidatos disponíveis.

/matches

GET

Retorna combinações válidas de vaga-candidato.

Exemplo de Requisição com curl
# Fazer uma predição
curl -X POST -H "Content-Type: application/json" -d '{"job_id": "4534", "candidate_id": "11132"}' http://127.0.0.1:5000/predict

# Acionar o retreino
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:5000/trigger_retraining

📈 Monitoramento com MLflow
Para visualizar os resultados dos seus experimentos de avaliação e retreino:

Inicie a Interface do MLflow: No terminal, na pasta do projeto, execute:

mlflow ui

Acesse no Navegador: Abra http://127.0.0.1:5000 (ou a porta indicada).

Na interface, pode comparar execuções, visualizar métricas como accuracy e f1_score, e analisar os modelos guardados como artefactos.

✅ Testes Automatizados
Para garantir a qualidade do código, a suíte de testes pode ser executada com um único comando:

pytest -v

☁️ Deploy no Render
Este projeto está configurado para deploy contínuo no Render.

Suba o seu código para um repositório no GitHub.

No painel do Render, crie um novo serviço do tipo Blueprint.

Selecione o seu repositório. O Render irá ler o ficheiro render.yaml e configurar tudo automaticamente.

A cada git push para a sua branch principal, um novo deploy será acionado.
