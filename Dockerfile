# --- Estágio 1: Imagem Base ---
# Usamos uma imagem Python "slim", que é menor e mais segura para produção.
# A versão específica (3.10) garante que o ambiente seja sempre o mesmo.
FROM python:3.10-slim

# --- Configuração do Ambiente ---
# Define o diretório de trabalho dentro do contêiner.
# Todos os comandos a seguir serão executados a partir daqui.
WORKDIR /app

# --- Instalação de Dependências (Otimizado) ---
# Copia APENAS o arquivo de requisitos primeiro. O Docker armazena essa camada em cache.
# Se você não mudar o requirements.txt, o Docker não reinstalará tudo a cada build.
COPY requirements.txt .

# Instala as dependências.
# --no-cache-dir economiza espaço na imagem final.
# --upgrade pip garante que estamos usando a versão mais recente do pip.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# --- Cópia do Código da Aplicação ---
# Agora, copia o resto do seu código (main.py, etc.) para o diretório /app.
COPY . .

# --- Exposição da Porta ---
# Informa ao Docker que a aplicação dentro do contêiner estará escutando na porta 8000.
# O Google Cloud Run usará essa informação para direcionar o tráfego.
EXPOSE 8000

# --- Comando de Execução ---
# Este é o comando que inicia sua API quando o contêiner é executado.
# uvicorn: O servidor ASGI que executa o FastAPI.
# main:app: "Execute o objeto chamado 'app' que está no arquivo 'main.py'".
# --host 0.0.0.0: ESSENCIAL. Permite que o servidor seja acessado de fora do contêiner.
# --port 8000: A porta que o Uvicorn usará, correspondendo ao EXPOSE.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
