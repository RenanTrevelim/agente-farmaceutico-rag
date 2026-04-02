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

📌 Implementação: :contentReference[oaicite:0]{index=0}

---

### 2. ✂️ Chunking inteligente
- Texto dividido em blocos com sobreposição
- Tokenização baseada em Hugging Face

📌 Implementação: :contentReference[oaicite:1]{index=1}

---

### 3. 🏷️ Enriquecimento semântico
Cada chunk é classificado em categorias como:
- `indicacao`
- `posologia`
- `contraindicacao`
- `efeitos_adversos`
- `precaucoes`

Isso melhora significativamente a qualidade do RAG.

📌 Implementação: :contentReference[oaicite:2]{index=2}

---

### 4. 🧠 Embeddings + Vector Store
- Modelo: `intfloat/multilingual-e5-small`
- Armazenamento vetorial com **FAISS**

📌 Implementação: :contentReference[oaicite:3]{index=3}

---

### 5. 🔎 Retriever
- Busca por similaridade semântica
- Recupera os chunks mais relevantes

📌 Implementação: :contentReference[oaicite:4]{index=4}

---

### 6. 🧩 Construção do contexto
- Junta os chunks relevantes
- Extrai categorias associadas

📌 Implementação: :contentReference[oaicite:5]{index=5}

---

### 7. 🤖 LLM (Groq + LLaMA 3)
- Modelo: `llama-3.3-70b-versatile`
- Respostas com:
  - zero alucinação
  - baseadas apenas no contexto

📌 Implementação: :contentReference[oaicite:6]{index=6}

---

### 8. 🔗 Chain RAG
- Combina:
  - pergunta
  - contexto recuperado
- Gera resposta final

📌 Implementação: :contentReference[oaicite:7]{index=7}

---

### 9. 💻 Interface (Streamlit)
- Interface web simples e rápida
- Input de perguntas
- Exibição de respostas

📌 Implementação: :contentReference[oaicite:8]{index=8}

---

## 🧪 Tecnologias Utilizadas

- **Python**
- **LangChain**
- **FAISS**
- **Hugging Face Transformers**
- **Groq (LLaMA 3)**
- **Streamlit**

---
