# 1. Importações

# 📄 Loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader

# ✂️ Divisão do texto em blocos
from langchain_text_splitters import CharacterTextSplitter

# Tokenizador dos nossos dados
from transformers import AutoTokenizer

# 🧠 Embeddings com Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings

# 🗄️ Banco vetorial
from langchain_community.vectorstores import FAISS

# 🤖 LLM com Groq
from langchain_groq import ChatGroq

# Imports da Chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os
import streamlit as st

os.environ['GROQ_API'] = "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"

########################################
# 2. Modelo LLM
########################################

llm = ChatGroq(model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key= sua_api)


########################################
# 3. Fazendo a Leitura do Documentos em PDF
########################################

def carregar_documentos():
    caminhos = [
        "data/IbuprofenoGotasMedquimica.pdf",
        "data/Paracetamol.pdf",
        "data/NimesulidaPratiDonaduzzi.pdf",
        "data/bula-Dipirona-Monohidratada-1g-Prati.pdf"
    ]

    documentos = []

    # Lendo os documentos dentro do "caminhos"
    for caminho in caminhos:
        docs = PyPDFLoader(caminho).load()

        # fazendo o metadados - colocando qual documento se refere
        for doc in docs:
            doc.metadata["documento"] = os.path.basename(caminho)

        documentos.extend(docs)

    return documentos

########################################
# 4. Fazendo o Chunk + Tokenizador
########################################

def gerar_chunk(documentos):

    tokenizador = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizador,
        chunk_size=1000,
        chunk_overlap=100
    )

    chunk = splitter.split_documents(documentos)

    return chunk

########################################
# 5. Fazendo o Enriquecimento dos metadados com Chunk
########################################

def enriquecer_chunk(chunks):
    for chunk in chunks:

        # Normalizando o texto para facilitar as verificações
        texto = chunk.page_content.lower()

        # Indicações terapêuticas
        if "para que este medicamento é indicado" in texto or "indicação" in texto:
            chunk.metadata["categoria"] = "indicacao"

        # funcionamento do medicamento
        elif "como este medicamento funciona" in texto or "funcionamento" in texto:
            chunk.metadata["categoria"] = "funcionamento"

        # Contraindicações do uso
        elif "quando não devo usar este medicamento" in texto or "contraindicação" in texto:
            chunk.metadata["categoria"] = "contraindicacao"

        # O que saber antes de Usar
        elif "o que devo saber antes de usar este medicamento" in texto or "saber antes" in texto:
            chunk.metadata["categoria"] = "saber_antes"

        # Como usar o medicamento
        elif "como devo usar este medicamento" in texto or "uso" in texto:
            chunk.metadata["categoria"] = "uso_medicamento"

        # Esquecer de tomar o medicamento
        elif "o que fazer quando esquecer de tomar o medicamento" in texto or "esquecer" in texto:
            chunk.metadata["categoria"] = "esquecer_medicamento"

        # O que o medicamento pode causar
        elif "o que pode causar este medicamento" in texto or "causa" in texto:
            chunk.metadata["categoria"] = "causa_medicamento"

        # Tomar altas doses do medicamento
        elif "o que fazer se tomar uma quantidade maior de medicamento" in texto or "alta" in texto:
            chunk.metadata["categoria"] = "alta_dose"

        else:
            chunk.metadata["categoria"] = "outros"

    return chunks

########################################
# 6. Fazendo o Vector Store + Embeddings
########################################

def cria_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store

########################################
# 7. Fazendo o Retriever
########################################

def cria_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever

########################################
# 8. Fazendo a Chain
########################################

def criar_chain(retriever, llm):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", "Responda de forma detalhada e apenas com base no contexto fornecido.\n Contexto: \n{contexto}"
        ),
        (
            "human", "Pergunta: \n{pergunta}"
        )
    ])

    def recuperar_contexto(pergunta):
        docs = retriever.invoke(pergunta)
        return "\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "contexto": RunnablePassthrough(recuperar_contexto),
            "pergunta": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

########################################
# 9. Interface Streamlit
########################################

st.set_page_config(
    page_title="Agente Farmacêutico com RAG",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Agente Farmacêutico")
st.caption("Consulte informações das bulas de forma simples e rápida.")

st.info("Medicamentos disponíveis: Dipirona, Ibuprofeno, Nimesulida e Paracetamol.")

pergunta = st.text_input(
    "Digite sua pergunta",
    placeholder="Ex.: Quais são os efeitos colaterais do paracetamol?"
)


if pergunta:
    with st.spinner("Consultando as Bulas dos Remédios..."):
        documentos = carregar_documentos()
        chunk = gerar_chunk(documentos)
        chunk = enriquecer_chunk(chunk)
        vectorstore = cria_vectorstore(chunk)
        retriever = cria_retriever(vectorstore)
        rag_chain = criar_chain(retriever, llm)

        resposta = rag_chain.invoke(pergunta)

    st.subheader("Resposta")
    st.write(resposta)