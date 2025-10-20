import numpy as np
import matplotlib.pyplot as plt
import io
import os
import base64
import shutil
import time

# --- Importações do FastAPI ---
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse

# --- Importações científicas ---
try:
    # Apenas as libs essenciais (se a segmentação for feita por média, KMeans não é necessário)
    import cv2
    from PIL import Image
    from pillow_heif import register_heif_opener
    
    # Configura o Pillow para suportar HEIC
    register_heif_opener()
except ImportError as e:
    print(f"ERRO: Falta uma biblioteca essencial. Execute 'pip install -r requirements.txt'. Detalhe: {e}")
    exit()

# =======================================================
# --- CONFIGURAÇÕES GLOBAIS ---
# =======================================================
MIN_WAVELENGTH = 400  # nm (Azul/Violeta)
MAX_WAVELENGTH = 700  # nm (Vermelho)
BLOCK_SIZE_SEGMENTATION = 12
MAX_IMAGE_WIDTH = 800 # OTIMIZAÇÃO: Largura máxima para redimensionamento
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


# =======================================================
# --- FUNÇÕES DE PROCESSAMENTO E IA OTIMIZADAS ---
# =======================================================

def convert_heic_to_jpg_if_needed(image_path):
    """Verifica se o arquivo é HEIC/HEIF e o converte para JPG."""
    if image_path.lower().endswith(('.heic', '.heif')):
        try:
            img_pil = Image.open(image_path)
            new_path = image_path.rsplit('.', 1)[0] + '.jpg'
            img_pil.save(new_path, format="jpeg")
            return new_path
        except Exception as e:
            print(f"Aviso: Erro ao converter HEIC/HEIF para JPG: {e}")
    return image_path


def load_image(image_path):
    """Carrega, converte BGR->RGB e REDIMENSIONA a imagem (OTIMIZAÇÃO DE VELOCIDADE)."""
    
    converted_path = convert_heic_to_jpg_if_needed(image_path)
    img_bgr = cv2.imread(converted_path)

    if img_bgr is None:
        if converted_path != image_path:
             img_bgr = cv2.imread(image_path)
             if img_bgr is None:
                  raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")
        else:
            raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # --- OTIMIZAÇÃO CRUCIAL: Redução de Resolução (Máx 800px) ---
    h, w, _ = img_rgb.shape
    
    if w > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / w
        new_h = int(h * ratio)
        img_rgb = cv2.resize(img_rgb, (MAX_IMAGE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    
    return img_rgb


def create_kmeans_segmentation(img_rgb, block_size):
    """Segmentação RÁPIDA por Blocos usando a Média da Cor (OTIMIZADO)."""
    h, w, _ = img_rgb.shape
    segmented_img = np.zeros_like(img_rgb)

    if block_size <= 0: return img_rgb

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block_rgb = img_rgb[y:y_end, x:x_end]

            if block_rgb.size > 0:
                # CÁLCULO DE MÉDIA SIMPLES (MUITO MAIS RÁPIDO)
                average_color = np.mean(block_rgb, axis=(0, 1)).astype(np.uint8)
                segmented_img[y:y_end, x:x_end] = average_color

    return segmented_img


def analyze_spectrum_profile(img_rgb):
    """Extrai o perfil de intensidade (Canal V do HSV)."""
    h, w, _ = img_rgb.shape
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v_channel = img_hsv[:, :, 2]

    center_y = h // 2 
    band_height = min(30, h // 4) 
    
    start_y = max(0, center_y - band_height // 2)
    end_y = min(h, center_y + band_height // 2)

    spectral_band = v_channel[start_y : end_y, :]

    if spectral_band.size == 0:
        return np.array([]) 

    intensity_profile = np.mean(spectral_band, axis=0)
    intensity_profile = intensity_profile / 255.0 * 100
    return intensity_profile


def create_plot_buffer(img_rgb, segmented_img, profile_sample, pico_lambda, pico_absorbancia):
    """Gera a imagem de resultados e a armazena em um buffer de memória."""

    width = img_rgb.shape[1]
    pixels = np.arange(0, width)
    slope_lambda = (MAX_WAVELENGTH - MIN_WAVELENGTH) / width
    lambdas_eixo_x = slope_lambda * pixels + MIN_WAVELENGTH

    profile_blank = np.full_like(profile_sample, 95.0)
    T_bruta = profile_sample / profile_blank
    T_bruta_safe = np.clip(T_bruta, 1e-6, 1.0)
    A_bruta = -np.log10(T_bruta_safe)

    # --- PLOTAGEM ---
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("1A. Imagem Original do Espectro")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(segmented_img)
    plt.title("1B. IA Decodificando Cores (Segmentação Rápida)")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(lambdas_eixo_x, profile_blank, label='I₀ (Referência)', color='gray', linestyle='--')
    plt.plot(lambdas_eixo_x, profile_sample, label='I (Amostra)', color='blue')
    plt.title('2A. Perfil de Intensidade Espectral (I e I₀)', fontsize=14)
    plt.xlabel('Comprimento de Onda (λ em nm)')
    plt.ylabel('Intensidade de Luz (% Brilho)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 105)

    plt.subplot(2, 2, 4)
    plt.plot(lambdas_eixo_x, A_bruta, label='Absorbância Bruta (-log(T))', color='red')

    max_A_index = np.argmax(A_bruta)
    pico_lambda_plot = slope_lambda * pixels[max_A_index] + MIN_WAVELENGTH
    pico_absorbancia_plot = A_bruta[max_A_index]
    
    plt.scatter(pico_lambda_plot, pico_absorbancia_plot, color='black', s=100)
    plt.annotate(f"Pico: {pico_lambda_plot:.1f} nm", (pico_lambda_plot, pico_absorbancia_plot),
                 textcoords="offset points", xytext=(-20, 20), ha='left',
                 fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

    plt.title('2B. Espectro de Absorbância Bruta', fontsize=14)
    plt.xlabel('Comprimento de Onda (λ em nm)')
    plt.ylabel('Absorbância (A)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def perform_analysis(img_rgb):
    """Executa todas as etapas de análise do seu algoritmo."""
    
    segmented_image = create_kmeans_segmentation(img_rgb, block_size=BLOCK_SIZE_SEGMENTATION)
    profile_sample = analyze_spectrum_profile(img_rgb)
    
    if profile_sample.size == 0:
        raise ValueError("Erro ao extrair perfil de intensidade (tamanho zero).")

    width = img_rgb.shape[1]
    slope_lambda = (MAX_WAVELENGTH - MIN_WAVELENGTH) / width
    
    profile_blank = np.full_like(profile_sample, 95.0) 
    T_bruta = profile_sample / profile_blank
    T_bruta_safe = np.clip(T_bruta, 1e-6, 1.0)
    A_bruta = -np.log10(T_bruta_safe)
    
    max_A_index = np.argmax(A_bruta)
    pixels_base = np.arange(0, width)
    
    pico_lambda = slope_lambda * pixels_base[max_A_index] + MIN_WAVELENGTH
    pico_absorbancia = A_bruta[max_A_index]

    plot_buffer = create_plot_buffer(img_rgb, segmented_image, profile_sample, pico_lambda, pico_absorbancia)
    
    return {
        'pico_lambda': float(pico_lambda),
        'pico_absorbancia': float(pico_absorbancia),
        'plot_buffer': plot_buffer
    }


# =======================================================
# --- CONFIGURAÇÃO DA API FASTAPI ---
# =======================================================

app = FastAPI(
    title="IFOTOM - Espectroscopia de Imagem API",
    description="API de Visão Computacional e IA para análise de espectros de absorbância em imagens."
)

@app.get("/")
async def root():
    """Rota raiz para verificar se a API está online."""
    return {
        "message": "API IFOTOM está Online. Acesse /docs para testar o endpoint de análise.",
        "endpoint_de_analise": "/analyze"
    }

@app.post("/analyze")
async def analyze_image_endpoint(file: UploadFile = File(
    None, 
    description="Arquivo de imagem (JPG, PNG, HEIC) do espectro."
)):
    """
    Recebe um arquivo de imagem e retorna o pico de absorbância e o gráfico de análise.
    """
    
    if file is None:
         raise HTTPException(status_code=400, detail="Nenhum arquivo 'file' enviado.")

    unique_filename = f"{time.time()}_{file.filename}"
    temp_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        # 1. Salva o arquivo enviado
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Executa a Análise (inclui Redimensionamento e IA)
        img_rgb = load_image(temp_path)
        result = perform_analysis(img_rgb)

        # 3. Prepara a Resposta JSON
        response_data = {
            "status": "success",
            "pico_lambda_nm": result['pico_lambda'],
            "pico_absorbancia": result['pico_absorbancia'],
            # Converte a imagem (plot_buffer) para Base64
            "plot_base64": base64.b64encode(result['plot_buffer'].read()).decode('utf-8')
        }
        
        return JSONResponse(content=response_data)

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Arquivo não encontrado no disco (verifique o formato).")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro de processamento nos dados: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno da API: {e}. Confirme se as dependências estão instaladas.")
    finally:
        # 4. Limpeza: Garante que os arquivos temporários são removidos
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Limpa o JPG convertido (se foi HEIC)
            jpg_temp_path = temp_path.rsplit('.', 1)[0] + '.jpg'
            if os.path.exists(jpg_temp_path):
                os.remove(jpg_temp_path)
        except OSError as cleanup_error:
            print(f"Aviso de limpeza: Não foi possível remover arquivo temporário: {cleanup_error}")