import os
import glob
import rasterio
import numpy as np
import torch
import random
import xarray as xr



# 1. Função para encontrar os pares de imagem e máscara
def encontrar_pares(imagens_dir, mascs_dir):
    padrao_busca = os.path.join(imagens_dir, '**', '*.nc')
    arquivos_imagens = sorted(glob.glob(padrao_busca, recursive=True))
    
    pares = []
    for img_path in arquivos_imagens:
        caminho_relativo = os.path.relpath(img_path, imagens_dir)
        caminho_relativo_masc = caminho_relativo.replace("_S2_", "_S2label_").replace(".nc", ".tif")
        mask_path = os.path.join(mascs_dir, caminho_relativo_masc)
        
        if os.path.exists(mask_path):
            pares.append((img_path, mask_path))
        else:
            print(f"Máscara não encontrada para: {os.path.basename(img_path)}")
            
    return pares



# 2. Função de processamento (sem alterações, calcula o NDVI na hora)
def proc_img_e_masc(img_path, mask_path, calc_ndvi=True, threshold_binario=5.0):
    with xr.open_dataset(img_path) as ds:
        img = ds.to_array().values.astype(np.float32)
        
        if np.max(img) > 10.0:
            img = img / 10000.0
            
    if calc_ndvi and img.shape[0] >= 4:
        red = img[2, :, :]
        nir = img[3, :, :]
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.expand_dims(ndvi, axis=0) 
        img = np.concatenate((img, ndvi), axis=0)

    with rasterio.open(mask_path) as src_mask:
        masc = src_mask.read(1).astype(np.float32)

    if threshold_binario is not None:
        masc = np.where(masc > threshold_binario, 0.0, 1.0)
        
    masc = np.expand_dims(masc, axis=0)

    return torch.from_numpy(img), torch.from_numpy(masc)



# 3. NOVO: Gerador funcional em tempo real
def gerador_em_tempo_real(pasta_imagens, pasta_mascs, batch_size=8, embaralhar=True, calc_ndvi=True, threshold_binario=5.0):
    pares = encontrar_pares(pasta_imagens, pasta_mascs)
    
    if embaralhar:
        random.shuffle(pares)
        
    lote_imagens = []
    lote_mascaras = []
    
    for img_path, mask_path in pares:
        try:
            # Em vez de carregar do disco, processa a imagem diretamente
            img_tensor, mask_tensor = proc_img_e_masc(
                img_path, mask_path, calc_ndvi, threshold_binario
            )
            
            lote_imagens.append(img_tensor)
            lote_mascaras.append(mask_tensor)
            
            # Quando atinge o tamanho do lote, envia para a GPU
            if len(lote_imagens) == batch_size:
                yield torch.stack(lote_imagens), torch.stack(lote_mascaras)
                lote_imagens = []
                lote_mascaras = []
                
        except Exception as e:
            # Se um ficheiro estiver corrompido, imprime o erro e avança para o próximo
            print(f"Erro ao processar {img_path}: {e}")
            continue
            
    # Devolve o último lote se sobrar algum (menor que o batch_size)
    if len(lote_imagens) > 0:
        yield torch.stack(lote_imagens), torch.stack(lote_mascaras)

pasta_imagens = r"dataset/sentinel2/images"
pasta_mascs = r"dataset/sentinel2/masks"

# Instancia o gerador
meu_gerador = gerador_em_tempo_real(
    pasta_imagens, 
    pasta_mascs, 
    batch_size=8, 
    embaralhar=True
)

print("Testando o gerador puramente funcional...")
for batch_img, batch_mask in meu_gerador:
    print("Batch Imagens:", batch_img.shape)
    print("Batch Máscaras:", batch_mask.shape)
    # Coloque aqui o código de treino (ex: output = modelo(batch_img))
    # Parar no primeiro batch para testar
