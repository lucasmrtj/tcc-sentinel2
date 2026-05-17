#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import glob
import rasterio
import numpy as np
import torch
import random
import xarray as xr


# In[22]:


def encontrar_pares(imagens_dir, mascs_dir):
    padrao_busca = os.path.join(imagens_dir, '**', '*.nc')
    arquivos_imagens = sorted(glob.glob(padrao_busca, recursive=True))

    pares = []

    for img_path in arquivos_imagens:
        caminho_relativo = os.path.relpath(img_path, imagens_dir)

        caminho_relativo_masc = caminho_relativo.replace("_S2_", "_S2label_")

        caminho_relativo_masc = caminho_relativo_masc.replace(".nc", ".tif")

        mask_path = os.path.join(mascs_dir, caminho_relativo_masc)

        if os.path.exists(mask_path):
            pares.append((img_path, mask_path))
        else:
            print(f"Máscara não encontrada para: {os.path.basename(img_path)}")
            print(f"   -> Procurou em: {mask_path}\n")

    return pares


# In[23]:


def proc_img_e_masc(img_path, mask_path, calc_ndvi=True, threshold_binario=5.0):

    with xr.open_dataset(img_path) as ds:
 # A ordem geralmente fica alfabética: 0=B02(Azul), 1=B03(Verde), 2=B04(Vermelho), 3=B08(NIR)
        img = ds.to_array().values.astype(np.float32)

# reflectância x 10000. 
        if np.max(img) > 10.0:
            img = img / 10000.0
# NDVI 
# o índice 2 Vermelho e o índice 3 é o NIR 
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


# In[24]:


def pre_proc_e_salvar(pares_arquivos, pasta_saida_padrao, calc_ndvi=True, threshold_binario=5.0):
    os.makedirs(pasta_saida_padrao, exist_ok=True)

    print(f"Iniciando o pré-processamento de {len(pares_arquivos)} amostras...")

    for i, (img_path, mask_path) in enumerate(pares_arquivos):
        try:
            img_tensor, mask_tensor = proc_img_e_masc(
                img_path, mask_path, calc_ndvi, threshold_binario
            )

            nome_base = os.path.basename(img_path).replace(".nc", "")

            nome_arquivo_salvar = f"{nome_base}_pronto.pt"
            caminho_salvar = os.path.join(pasta_saida_padrao, nome_arquivo_salvar)

            torch.save({
                'imagem': img_tensor,
                'mascara': mask_tensor
            }, caminho_salvar)

            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{len(pares_arquivos)}] Salvo: {nome_arquivo_salvar}")

        except Exception as e:
            print(f"Erro{img_path}: {e}")



# In[25]:


pasta_imagens = r"dataset/sentinel2/images"
pasta_mascs = r"dataset/sentinel2/masks"
pasta_pre_processada = r"dataset/pre_processado"

meus_dados = encontrar_pares(pasta_imagens, pasta_mascs)

if len(meus_dados) > 0:
        pre_proc_e_salvar(meus_dados, pasta_pre_processada)

