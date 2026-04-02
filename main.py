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

os.environ['GROQ_API'] = "API_GROQ"

########################################
# 2. Modelo LLM
########################################

llm = ChatGroq(model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key= "API_GROQ")


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

def encontrar_categoria(texto, regras):
    for nome_categoria, palavras_chave in regras.items():
        for palavra_chave in palavras_chave:
            if palavra_chave in texto:
                return nome_categoria
    return "outros"


def enriquecer_chunk(chunks):
    regras = {
    "indicacao": [
        "para que este medicamento é indicado",
        "para que este medicamento serve",
        "indicações terapêuticas",
        "indicação do medicamento",
        "indicações de uso"
    ],

    "funcionamento": [
        "como este medicamento funciona",
        "como o medicamento funciona",
        "ação esperada do medicamento",
        "mecanismo de ação"
    ],

    "contraindicacao": [
        "quando não devo usar este medicamento",
        "quando não usar este medicamento",
        "contraindicações",
        "contraindicação",
        "quem não deve usar",
        "não deve ser utilizado"
    ],

    "precaucoes": [
        "o que devo saber antes de usar este medicamento",
        "advertências e precauções",
        "advertências",
        "precauções",
        "cuidados de uso",
        "antes de usar"
    ],

    "posologia": [
        "como devo usar este medicamento",
        "como usar este medicamento",
        "posologia",
        "modo de usar",
        "modo de administração",
        "administração",
        "dose recomendada"
    ],

    "dose_esquecida": [
        "o que fazer quando eu me esquecer de usar este medicamento",
        "o que fazer se eu esquecer de usar este medicamento",
        "esquecimento de dose",
        "dose esquecida",
        "esquecer de tomar"
    ],

    "efeitos_adversos": [
        "quais os males que este medicamento pode me causar",
        "reações adversas",
        "reação adversa",
        "efeitos colaterais",
        "efeitos adversos",
        "eventos adversos"
    ],

    "superdose": [
        "o que fazer se alguém usar uma quantidade maior do que a indicada",
        "o que fazer se usar uma quantidade maior do que a indicada",
        "superdose",
        "sobredosagem",
        "uso em excesso",
        "dose excessiva"
    ],

    "composicao": [
        "composição",
        "composição do medicamento",
        "composição qualitativa",
        "composição quantitativa"
    ],

    "armazenamento": [
        "onde, como e por quanto tempo posso guardar este medicamento",
        "cuidados de conservação",
        "armazenamento",
        "conservação do medicamento"
    ]
}

    for chunk in chunks:
        texto = chunk.page_content.lower()
        categoria = encontrar_categoria(texto, regras)
        chunk.metadata["categoria"] = categoria

    return chunks

########################################
# 6. Fazendo o Vector Store + Embeddings
########################################
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

def cria_vectorstore(chunks):

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
        search_kwargs={"k": 4}
    )

    return retriever

########################################
# 8. Recuperando o Contexto
########################################

def recuperar_contexto(pergunta, retriever):
    docs = retriever.invoke(pergunta)

    contextos = []
    categorias = []

    for doc in docs:
        contextos.append(doc.page_content)
        if "categoria" in doc.metadata:
            categorias.append(doc.metadata["categoria"])

    return {
        "contexto": "\n\n".join(contextos),
        "categorias": list(set(categorias))
    }

########################################
# 9. Fazendo a Chain
########################################

def criar_chain(retriever, llm):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Você é um assistente de perguntas e respostas baseado em contexto recuperado.

            Regras:
            1. Responda usando somente as informações do CONTEXTO.
            2. Não invente informações, exemplos, números ou definições que não estejam no CONTEXTO.
            3. Se o CONTEXTO não trouxer informação suficiente para responder com segurança, diga claramente:
            "Não encontrei informação suficiente no contexto para responder."
            4. Se houver trechos conflitantes no CONTEXTO, aponte o conflito e não assuma nada sem evidência.
            5. Seja claro, objetivo e fiel ao texto recuperado.
            6. Sempre que possível, cite o trecho ou documento de onde tirou a resposta.
            7. Organize a resposta em tópicos quando isso melhorar a clareza.

            CONTEXTO:
            {contexto}
            """
        ),
        (
            "human", "Pergunta: \n{pergunta}"
        )
    ])
    
    rag_chain = (
        {
            "contexto": RunnableLambda(lambda pergunta: recuperar_contexto(pergunta, retriever)["contexto"]),
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
    layout="centered"
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
        recupera_contexto = recuperar_contexto(pergunta, retriever)

    st.subheader("Resposta")
    st.write(resposta)