# Sistema de Recomendação de Candidatos com IA

Este projeto é uma aplicação web completa que utiliza Machine Learning para automatizar e otimizar o processo de seleção de candidatos, calculando a probabilidade de sucesso entre perfis e vagas de emprego.

A solução foi desenhada para ser eficiente e escalável, utilizando um fluxo de trabalho em duas etapas: um **script de pré-processamento offline** para tratar e criar uma amostra de dados, e uma **aplicação Flask** leve que consome esses dados para fazer predições em tempo real.

## Funcionalidades Principais

* **Pré-processamento de Dados Robusto:** Um script (`create_sample.py`) descarrega, extrai e processa ficheiros JSON complexos e aninhados, transformando-os em tabelas limpas e estruturadas.
* **Otimização de Memória:** O script gera uma amostra de dados pequena e otimizada, permitindo que a aplicação funcione em ambientes com memória limitada, como o plano gratuito do Render.
* **Matching com IA:** A aplicação principal (`app.py`) utiliza um modelo de rede neural (PyTorch) para analisar a compatibilidade entre os perfis da amostra.
* **API RESTful Completa:** Oferece endpoints para predições, avaliação de modelo, retreino, consulta de dados e verificação de saúde do sistema.
* **Ciclo de Vida MLOps Completo:** Inclui registo em banco de dados, monitoramento de métricas com MLflow e um portão de qualidade para o retreino de modelos.
* **Pronto para Deploy:** Configurado para deploy fácil na plataforma Render usando um ficheiro `render.yaml` e Gunicorn como servidor de produção.

## Processamento de Dados e Otimização

Para garantir a performance e a viabilidade do projeto em ambientes de nuvem, foi implementado um passo crucial de pré-processamento de dados.

### A Limitação Técnica

* **GitHub:** A plataforma tem um limite estrito para o tamanho dos ficheiros (geralmente 100MB), o que torna inviável subir as bases de dados originais e grandes.
* **Render (Plano Gratuito):** O ambiente de produção oferece uma memória RAM limitada (512MB). Carregar e processar os ficheiros JSON grandes e complexos diretamente na aplicação excederia este limite, causando o erro "Ran out of memory".

### A Solução: `create_sample.py`

Para contornar estas limitações, o script `create_sample.py` realiza um tratamento offline dos dados. Ele é o coração da nossa estratégia de otimização.

1.  **Download e Extração:** O script primeiro descarrega os ficheiros `.zip` do Google Drive e extrai os ficheiros JSON brutos.
2.  **Tratamento do JSON:** Utiliza funções de processamento personalizadas (`processar_jobs`, `processar_prospects`, etc.) para navegar pela estrutura complexa e "aninhada" dos ficheiros JSON originais. Ele "achata" os dados, transformando-os em tabelas simples e limpas (DataFrames).
3.  **Análise de Qualidade:** Após o tratamento, o script verifica automaticamente cada tabela em busca de dados em falta (nulos) e gera um resumo estatístico (`describe()`), garantindo a integridade dos dados.
4.  **Criação da Amostra:** Por fim, ele seleciona uma amostra representativa e pequena dos dados (500 registos) e salva-a em formato otimizado.

O resultado é um conjunto de dados leve, limpo e pronto para ser usado, que é enviado para o GitHub e consumido pela aplicação principal.

## Ciclo de Vida da Aplicação: Do Dado à Decisão

O sistema foi desenhado com um ciclo de vida completo em mente, garantindo não apenas a execução, mas também a auditoria, o monitoramento e a evolução contínua do modelo.

### A. Registro e Auditoria (Banco de Dados SQLite)

Para garantir a rastreabilidade e a transparência, todas as operações importantes são registadas num banco de dados SQLite.

* **Registo de Predições:** Sempre que a rota `/predict` é chamada, um registo detalhado é salvo na tabela `predictions`. Isto inclui o timestamp, os IDs da vaga e do candidato, a probabilidade de sucesso calculada, a recomendação final e o tempo total da predição. Isto cria um histórico auditável de todas as recomendações feitas pelo sistema.
* **Registo de Avaliações:** Da mesma forma, quando a rota `/evaluate_model` é executada, um resumo da avaliação é guardado na tabela `evaluations`, incluindo as métricas de performance (Acurácia, F1-Score, etc.).

### B. Monitoramento e Avaliação (MLflow & Testes)

A qualidade do modelo é constantemente monitorada.

* **Avaliação de Performance:** A rota `/evaluate_model` permite que a eficácia do modelo em produção seja verificada a qualquer momento. Ela usa um conjunto de dados de teste para calcular métricas chave (Acurácia, Precisão, Recall, F1-Score) e as regista no **MLflow**.
* **Visualização com MLflow:** Ao executar o comando `mlflow ui` localmente, é possível aceder a um painel de controlo visual para comparar as métricas de diferentes execuções, analisar a degradação do modelo ao longo do tempo e tomar decisões informadas sobre a necessidade de retreino.
* **Testes Automatizados:** A suíte de testes **Pytest** valida a integridade do código e a lógica de negócio, garantindo que novas alterações não quebrem funcionalidades existentes.

### C. Evolução Contínua (Retreino do Modelo)

O modelo não é estático. A rota `/trigger_retraining` permite a sua evolução contínua.

* **Retreino Sob Demanda:** Esta rota `POST` inicia o processo de treinamento de um novo modelo. É possível enviar hiperparâmetros personalizados (como taxa de aprendizado ou número de épocas) no corpo da requisição para experimentar diferentes arquiteturas.
* **Portão de Qualidade (Quality Gate):** Um novo modelo só é promovido para produção se a sua performance for superior a um limite mínimo predefinido (ex: F1-Score >= 0.75). Se o novo modelo não atingir este patamar de qualidade, ele é descartado e o modelo antigo é mantido, garantindo que a performance do sistema nunca seja degradada.

## Tecnologias Utilizadas

* **Backend:** Flask, Gunicorn
* **Machine Learning:** PyTorch, Scikit-learn
* **Manipulação de Dados:** Pandas, NumPy
* **Utilitários:** gdown (para downloads do Google Drive)
* **Banco de Dados:** SQLite

## Fluxo de Trabalho e Execução

O projeto opera em duas fases distintas:

### **Fase 1: Preparação dos Dados (Executar uma única vez, localmente)**

Esta fase é realizada pelo script `create_sample.py`. O seu objetivo é pegar nos seus dados brutos e grandes e criar uma amostra pequena e limpa que será usada pela aplicação.

1.  **Pré-requisitos:** Tenha o Python e o `pip` instalados.
2.  **Instale as dependências necessárias para o script:**
    ```
    pip install pandas gdown
    ```
3.  **Execute o script:** No seu terminal, na pasta raiz do projeto, execute:
    ```
    python create_sample.py
    ```
4.  **Resultado:** O script irá criar uma nova pasta `data/` contendo os ficheiros `jobs.json`, `prospects.json` e `applicants.json` em formato otimizado e com um tamanho reduzido. São estes os ficheiros que devem ser enviados para o GitHub.

### **Fase 2: Executar a Aplicação Principal**

Depois de gerar a amostra de dados, a aplicação principal pode ser executada tanto localmente como no Render.

1.  **Instale todas as dependências do projeto:**
    ```
    pip install -r requirements.txt
    ```
2.  **Execute a aplicação Flask:**
    ```
    python app.py
    ```
3.  **Aceda:** A aplicação estará disponível em `http://127.0.0.1:5000`.

## Endpoints da API

A seguir, os principais endpoints disponíveis na aplicação:

| Rota | Método | Descrição |
| :--- | :--- | :--- |
| `/` | `GET` | Exibe a interface principal da aplicação. |
| `/predict` | `POST` | Retorna a predição de compatibilidade para uma vaga e um candidato. |
| `/evaluate_model` | `GET` | Avalia a performance do modelo atual e regista as métricas no MLflow. |
| `/trigger_retraining` | `POST` | Inicia o processo de retreino de um novo modelo. |
| `/health` | `GET` | Verifica a saúde da aplicação e dos seus componentes. |
| `/jobs` | `GET` | Lista as vagas disponíveis na amostra de dados. |
| `/candidates` | `GET` | Lista os candidatos disponíveis na amostra de dados. |
| `/matches` | `GET` | Retorna as combinações válidas de vaga-candidato da amostra. |

## Deploy no Render

Este projeto está configurado para deploy contínuo no Render.

1.  **Execute o `create_sample.py`** localmente para gerar a pasta `data/` com os dados de amostra.
2.  **Suba o seu código**, incluindo a pasta `data/`, para um repositório no GitHub.
3.  No painel do Render, crie um novo serviço do tipo **Blueprint**.
4.  Selecione o seu repositório. O Render irá ler o ficheiro `render.yaml` e configurar tudo automaticamente.
5.  A cada `git push` para a sua branch principal, um novo deploy será acionado.
