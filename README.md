# 💊 Agente Farmacêutico com IA (RAG)

Um agente de Inteligência Artificial especializado em responder perguntas sobre **bulas de medicamentos**, utilizando a arquitetura **RAG (Retrieval-Augmented Generation)**.

O sistema permite consultar informações como:
- Indicações
- Posologia
- Contraindicações
- Efeitos colaterais
- Precauções
- Superdose

Tudo com base **exclusivamente nas bulas oficiais**, evitando alucinações.

---

## 🚀 Demonstração

Interface simples construída com **Streamlit**, permitindo consultas como:

> "Nimesulida ajuda para dor de garganta?"  
> "Quais são os efeitos colaterais do paracetamol?"  
> "Como devo tomar ibuprofeno?"

---

## 🧠 Arquitetura do Projeto

Este projeto utiliza o padrão **RAG (Retrieval-Augmented Generation)**:

Pergunta do usuário  
⬇️  
**Retriever (FAISS + Embeddings)**  
⬇️  
**Contexto relevante (chunks das bulas)**  
⬇️  
**LLM (Groq - LLaMA 3)**  
⬇️  
**Resposta baseada apenas no contexto**


---

## 🏗️ Pipeline do Sistema

### 1. 📄 Carregamento de documentos
- PDFs de bulas são carregados usando `PyPDFLoader`
- Metadados são adicionados (nome do medicamento)


---

### 2. ✂️ Chunking inteligente
- Texto dividido em blocos com sobreposição
- Tokenização baseada em Hugging Face


---

### 3. 🏷️ Enriquecimento semântico
Cada chunk é classificado em categorias como:
- `indicacao`
- `posologia`
- `contraindicacao`
- `efeitos_adversos`
- `precaucoes`

Isso melhora significativamente a qualidade do RAG.


---

### 4. 🧠 Embeddings + Vector Store
- Modelo: `intfloat/multilingual-e5-small`
- Armazenamento vetorial com **FAISS**


---

### 5. 🔎 Retriever
- Busca por similaridade semântica
- Recupera os chunks mais relevantes


---

### 6. 🧩 Construção do contexto
- Junta os chunks relevantes
- Extrai categorias associadas


---

### 7. 🤖 LLM (Groq + LLaMA 3)
- Modelo: `llama-3.3-70b-versatile`
- Respostas com:
  - zero alucinação
  - baseadas apenas no contexto


---

### 8. 🔗 Chain RAG
- Combina:
  - pergunta
  - contexto recuperado
- Gera resposta final


---

### 9. 💻 Interface (Streamlit)
- Interface web simples e rápida
- Input de perguntas
- Exibição de respostas


---

## 🧪 Tecnologias Utilizadas

- **Python**
- **LangChain**
- **FAISS**
- **Hugging Face Transformers**
- **Groq (LLaMA 3)**
- **Streamlit**

---

## ▶️ Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd <nome-do-projeto>
code .
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate  

# Windows
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Executando o Streamlit

```bash
streamlit run main.py
```


### 5. Executando o Docker

```bash
docker build -t casas-california .
docker run -p 8501:8501 casas-california
```

---
## 📷 Preview da Aplicação
