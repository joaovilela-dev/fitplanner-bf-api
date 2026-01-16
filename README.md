# 🏋️ FitPlanner Body Fat API

API backend desenvolvida em **Python + FastAPI** para **estimativa educacional de percentual de gordura corporal (Body Fat)** a partir de imagens e dados do usuário, utilizando conceitos de **Visão Computacional** e **modelagem heurística inspirada em Machine Learning**.

Este serviço faz parte do ecossistema do projeto **FitPlanner**, sendo consumido pelo aplicativo mobile desenvolvido em **Flutter**.

---

## 📌 Sobre o Projeto

A **FitPlanner Body Fat API** recebe informações básicas do usuário (sexo, idade, altura, peso) e uma imagem corporal, processando esses dados para gerar uma **estimativa aproximada de Body Fat**.

> ⚠️ **Aviso importante:**  
> Esta API tem finalidade **educacional e informativa**, não substituindo avaliação profissional médica ou física.

---

## 🎯 Problema que Resolve

- ❌ Falta de ferramentas acessíveis para estimar percentual de gordura  
- ❌ Dependência de equipamentos caros (bioimpedância, adipômetro)  
- ❌ Dificuldade de acompanhamento inicial para usuários iniciantes  

### ✅ Solução

- 📸 Estimativa de Body Fat baseada em imagem  
- 📊 Cálculo combinado com dados antropométricos  
- ⚡ API rápida, simples e escalável  
- 🔗 Integração direta com aplicativo Flutter  

---

## 🛠 Tecnologias Utilizadas

### Backend
- Python 3.10+
- FastAPI
- Uvicorn

### Visão Computacional / Modelagem
- OpenCV
- NumPy
- Scikit-learn (modelo heurístico inicial)

### Outros
- Pydantic (validação de dados)
- Pillow (processamento de imagens)

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.10 ou superior
- pip
- Ambiente virtual (recomendado)

### Passo a passo

```bash
# Clone o repositório
git clone https://github.com/joaovilela-dev/fitplanner-bf-api.git
cd fitplanner-bf-api

# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Execute a API
uvicorn main:app --reload
A API ficará disponível em:

http://127.0.0.1:8000

Documentação Swagger: http://127.0.0.1:8000/docs

📡 Principais Endpoints
🔹 Estimar Body Fat
POST /estimate-bodyfat

Parâmetros (JSON ou FormData):

json
Copiar código
{
  "sexo": "masculino",
  "idade": 22,
  "altura": 175,
  "peso": 78
}
Arquivo:

Imagem corporal (jpg/png)

Resposta:

json
Copiar código
{
  "body_fat_percentage": 15.8,
  "classification": "Fitness",
  "message": "Estimativa educacional baseada em imagem e dados corporais"
}
🧠 Arquitetura do Projeto
css
Copiar código
fitplanner-bf-api/
├── app/
│   ├── services/
│   ├── utils/
│   └── schemas.py
├── models/
├── scripts/
├── main.py
├── requirements.txt
└── README.md
🔐 Boas Práticas e Segurança
Não armazena imagens permanentemente

Validação rigorosa de dados com Pydantic

Separação clara entre lógica de negócio e rotas

Código modular e escalável

🔗 Integração com o Frontend
Este backend é consumido pelo aplicativo FitPlanner Frontend (Flutter):

👉 https://github.com/joaovilela-dev/fitplanner

📝 Roadmap
✅ Implementado
Estimativa básica de Body Fat

Processamento de imagem

API REST com FastAPI

🚧 Em desenvolvimento
Melhoria do modelo de estimativa

Normalização automática de imagens

Logs e métricas

📅 Planejado
Versionamento da API

Autenticação

Histórico de estimativas

Deploy em nuvem

📄 Licença
Este projeto está licenciado sob a MIT License.
Veja o arquivo LICENSE para mais detalhes.

👨‍💻 Autor
João Victor Vilela
GitHub: https://github.com/joaovilela-dev

⭐ Se este projeto te ajudou, considere dar uma estrela!