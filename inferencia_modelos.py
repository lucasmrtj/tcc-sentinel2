import os
import glob
import cv2
import rasterio
import xarray as xr
import numpy as np
import torch
from pathlib import Path

# IMPORTAÇÃO DA U-NET: Busca a sua arquitetura diretamente do seu ficheiro local
from arq_unet import UNet

# =========================================================================
# FUNÇÕES AUXILIARES (CARREGAMENTO, LIMPEZA E ESCRITA)
# =========================================================================
def carregar_imagem_e_mascara(caminho_img, caminho_masc, calc_ndvi=True):
    # 1. Carrega a Imagem Multi-canal
    with xr.open_dataset(caminho_img) as ds:
        img = ds.to_array().values.astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(img) > 10.0: img = img / 10000.0
            
    # Adiciona o canal NDVI
    if calc_ndvi and img.shape[0] >= 4:
        red, nir = img[2, :, :], img[3, :, :]
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.concatenate((img, np.expand_dims(ndvi, axis=0)), axis=0)

    # 2. Carrega a Máscara Real e recolhe o perfil de georreferenciação
    with rasterio.open(caminho_masc) as src_mask:
        perfil_tif = src_mask.profile 
        masc_real = src_mask.read(1).astype(np.float32)
        masc_real = np.nan_to_num(masc_real, nan=0.0, posinf=0.0, neginf=0.0)
        masc_real = np.where(masc_real > 0, 1.0, 0.0)

    tensor_img = torch.from_numpy(img).unsqueeze(0)
    return tensor_img, masc_real, perfil_tif, img

def aplicar_pos_processamento_cv2(mascara_binaria, tamanho_kernel=3):
    # Filtro Morfológico (Fechamento e Abertura) para limpar "sal e pimenta"
    masc_numpy = mascara_binaria.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tamanho_kernel, tamanho_kernel))
    masc_limpa = cv2.morphologyEx(masc_numpy, cv2.MORPH_CLOSE, kernel)
    masc_limpa = cv2.morphologyEx(masc_limpa, cv2.MORPH_OPEN, kernel)
    return masc_limpa

def salvar_mascara_tif(matriz_mascara, caminho_saida, perfil_original):
    # Clona os metadados do Sentinel-2 original para manter as coordenadas de mapa
    perfil_salvar = perfil_original.copy()
    perfil_salvar.update(dtype=rasterio.float32, count=1, compress='lzw')
    
    with rasterio.open(caminho_saida, 'w', **perfil_salvar) as dest:
        dest.write(matriz_mascara.astype(np.float32), 1)

def compor_rgb_falso_cor(matriz_img_bruta):
    # Cria a imagem "Visível" PNG usando as bandas do Sentinel
    try:
        rgb = np.stack([matriz_img_bruta[3], matriz_img_bruta[2], matriz_img_bruta[1]], axis=-1)
        rgb_norm = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
        return rgb_norm.astype(np.uint8)
    except:
        return np.zeros((256, 256, 3), dtype=np.uint8)

# =========================================================================
# LOOP PRINCIPAL: TESTAR TODOS OS MODELOS E EXPORTAR ARQUIVOS
# =========================================================================
def executar_inferencia_em_lote():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> A iniciar inferência no dispositivo: {device}")

    # CONFIGURAÇÃO DE DIRETÓRIOS
    pasta_modelos = "." # Busca todos os .pt na raiz do repositório
    pasta_dataset_img = "dataset/sentinel2/images"
    pasta_dataset_masc = "dataset/sentinel2/masks"
    pasta_exportacao_raiz = "resultados_inferencia"

    # Encontrar Modelos Salvos
    modelos_pt = glob.glob(os.path.join(pasta_modelos, "*.pt"))
    if not modelos_pt:
        print("-> AVISO: Nenhum ficheiro '.pt' foi encontrado. Cancele e treine o modelo primeiro.")
        return

    # Buscar uma amostra de teste (por exemplo, as 10 primeiras imagens para avaliar)
    imagens_teste = sorted(glob.glob(os.path.join(pasta_dataset_img, '**', '*.nc'), recursive=True))[:10]

    for caminho_modelo in modelos_pt:
        nome_modelo = Path(caminho_modelo).stem
        print(f"\n=============================================")
        print(f"A Carregar Pesos: {nome_modelo}")
        print(f"=============================================")

        # Cria a hierarquia de pastas (resultados -> NOME_DO_MODELO)
        pasta_saida_modelo = os.path.join(pasta_exportacao_raiz, nome_modelo)
        os.makedirs(pasta_saida_modelo, exist_ok=True)

        # Inicia a U-Net e injeta os pesos do modelo atual
        modelo = UNet(in_channels=42, out_channels=1).to(device)
        modelo.load_state_dict(torch.load(caminho_modelo, map_location=device))
        modelo.eval()

        for caminho_img in imagens_teste:
            nome_arquivo = Path(caminho_img).stem
            caminho_relativo = os.path.relpath(caminho_img, pasta_dataset_img)
            caminho_masc = os.path.join(pasta_dataset_masc, caminho_relativo.replace("_S2_", "_S2label_").replace(".nc", ".tif"))

            if not os.path.exists(caminho_masc):
                continue

            print(f" -> A extrair predições para: {nome_arquivo}")
            tensor_in, masc_real, perfil_tif, img_bruta = carregar_imagem_e_mascara(caminho_img, caminho_masc)
            tensor_in = tensor_in.to(device)

            # PREDIÇÃO DA REDE NEURAL
            with torch.no_grad():
                predicao = modelo(tensor_in)
                pred_binaria_bruta = (torch.sigmoid(predicao) > 0.5).int().squeeze().cpu().numpy()

            # PÓS-PROCESSAMENTO DO RUÍDO
            pred_limpa = aplicar_pos_processamento_cv2(pred_binaria_bruta, tamanho_kernel=3)

            # --- CRIAÇÃO DOS FICHEIROS FÍSICOS ---
            # Cria a pasta final específica para esta imagem de satélite
            pasta_arquivos_especificos = os.path.join(pasta_saida_modelo, nome_arquivo)
            os.makedirs(pasta_arquivos_especificos, exist_ok=True)

            # 1. Salvar .TIF (Máscara Preditiva Georreferenciada para o QGIS)
            tif_saida = os.path.join(pasta_arquivos_especificos, f"{nome_arquivo}_PREDICAO_FINAL.tif")
            salvar_mascara_tif(pred_limpa, tif_saida, perfil_tif)

            # 2. Salvar .PNG da Imagem Falsa Cor do Satélite
            img_rgb_png = compor_rgb_falso_cor(img_bruta)
            cv2.imwrite(os.path.join(pasta_arquivos_especificos, "1_Imagem_Satelite.png"), cv2.cvtColor(img_rgb_png, cv2.COLOR_RGB2BGR))

            # 3. Salvar .PNG da Máscara Real (Ground Truth Binário)
            masc_real_png = (masc_real * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(pasta_arquivos_especificos, "2_Mascara_Referencia.png"), masc_real_png)

            # 4. Salvar .PNG da Predição Limpa
            pred_limpa_png = (pred_limpa * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(pasta_arquivos_especificos, "3_Predicao_UNET.png"), pred_limpa_png)

    print("\n[SUCESSO] Processo de inferência concluído! Todos os ficheiros TIF e PNG estão na pasta 'resultados_inferencia'.")

if __name__ == "__main__":
    executar_inferencia_em_lote()