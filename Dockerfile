# --- Estágio 1: Imagem Base ---
# Continuamos com uma imagem Python slim, que é leve e segura.
FROM python:3.10-slim

# --- Instalação de Dependências de Sistema ---
# ATUALIZAÇÃO CRÍTICA: opencv-python-headless precisa de algumas bibliotecas de sistema
# que não vêm na imagem 'slim'. Sem isso, o build pode falhar.
# O '&& rm -rf /var/lib/apt/lists/*' mantém a imagem final pequena.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Configuração do Ambiente ---
WORKDIR /app

# Define um valor padrão para a variável PORT.
# O Google Cloud Run irá sobrescrever isso com o valor que ele quiser (geralmente 8080).
ENV PORT 8080

# --- Instalação de Dependências Python (Otimizado) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# --- Cópia do Código da Aplicação ---
COPY . .

# --- Exposição da Porta ---
# Expõe a porta definida pela variável de ambiente.
EXPOSE $PORT

# --- Comando de Execução Dinâmico ---
# ATUALIZAÇÃO CRÍTICA: Em vez de '--port 8000', usamos '--port $PORT'.
# O shell dentro do contêiner substituirá $PORT pelo valor da variável de ambiente (ex: 8080)
# que o Google Cloud Run fornecerá quando iniciar o contêiner.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

