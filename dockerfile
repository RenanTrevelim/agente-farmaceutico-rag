# Usa uma imagem base leve com Python 3.11
FROM python:3.14-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências para dentro do container
COPY requirements.txt .

# Instala as bibliotecas necessárias do projeto
RUN pip install -r requirements.txt

# Copia todo o restante do projeto para o container
COPY . .

# Comando que será executado ao iniciar o container
# Inicia o app Streamlit acessível externamente na porta 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]

# docker build -t agente-farmaceutico-rag .
# docker run -p 8501:8501 agente-farmaceutico-rag