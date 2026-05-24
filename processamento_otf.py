import os
import glob
import rasterio
import numpy as np
import torch
import torch.nn as nn
import random
import xarray as xr
import json
from arq_unet import UNet
from avalicao import calcular_metricas_pixel, calcular_metricas_objeto_e_borda, calcular_fidelidade_geometrica


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

# 2. Função de processamento (sem alterações, calcula o NDVI na hora)
def proc_img_e_masc(img_path, mask_path, calc_ndvi=True, threshold_binario=5.0):
    with xr.open_dataset(img_path) as ds:
        img = ds.to_array().values.astype(np.float32)
        
        # --- CORREÇÃO CRÍTICA: Elimina NaNs e Infs vindos do arquivo NetCDF ---
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        
        if np.max(img) > 10.0:
            img = img / 10000.0
            
    if calc_ndvi and img.shape[0] >= 4:
        red = img[2, :, :]
        nir = img[3, :, :]
        # O + 1e-8 evita divisão por zero, mas o nan_to_num garante estabilidade completa
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
        ndvi = np.expand_dims(ndvi, axis=0) 
        img = np.concatenate((img, ndvi), axis=0)
    with rasterio.open(mask_path) as src_mask:
        masc = src_mask.read(1).astype(np.float32)
        masc = np.nan_to_num(masc, nan=0.0, posinf=0.0, neginf=0.0)
    masc = np.where(masc > 0, 1.0, 0.0)
        
    masc = np.expand_dims(masc, axis=0)

    return torch.from_numpy(img), torch.from_numpy(masc)

# Modificamos o gerador para aceitar a lista 'pares' diretamente
def gerador_em_tempo_real(pares, batch_size=8, embaralhar=True, calc_ndvi=True, threshold_binario=5.0):
    # Criamos uma cópia para não estragar a lista original ao baralhar entre épocas
    lista_trabalho = list(pares)
    
    if embaralhar:
        random.shuffle(lista_trabalho)
        
    lote_imagens = []
    lote_mascaras = []
    
    for img_path, mask_path in lista_trabalho:
        try:
            # Processa em tempo real (on-the-fly) economizando espaço em disco
            img_tensor, mask_tensor = proc_img_e_masc(
                img_path, mask_path, calc_ndvi, threshold_binario
            )
            
            lote_imagens.append(img_tensor)
            lote_mascaras.append(mask_tensor)
            
            if len(lote_imagens) == batch_size:
                yield torch.stack(lote_imagens), torch.stack(lote_mascaras)
                lote_imagens = []
                lote_mascaras = []
                
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
            continue
            
    if len(lote_imagens) > 0:
        yield torch.stack(lote_imagens), torch.stack(lote_mascaras)

pasta_imagens = r"dataset/sentinel2/images"
pasta_mascs = r"dataset/sentinel2/masks"

# 1. Procura todos os pares disponíveis no disco
todos_os_pares = encontrar_pares(pasta_imagens, pasta_mascs)

# 2. Define uma semente (seed) para que a divisão seja sempre igual se reexecutar o código
random.seed(42)
random.shuffle(todos_os_pares)

# 3. Calcula a proporção (80% para treino, 20% para teste)
porcentagem_treino = 0.8
ponto_de_divisao = int(len(todos_os_pares) * porcentagem_treino)

pares_treino = todos_os_pares[:ponto_de_divisao]
pares_teste = todos_os_pares[ponto_de_divisao:]

print(f"Total de imagens encontradas: {len(todos_os_pares)}")
print(f"Quantidade para Treino (80%): {len(pares_treino)}")
print(f"Quantidade para Teste  (20%): {len(pares_teste)}")
# Instancia o gerador
meu_gerador = gerador_em_tempo_real(
    pares_treino, 
    batch_size=8, 
    embaralhar=True
)

print("Testando o gerador puramente funcional...")
#for batch_img, batch_mask in meu_gerador:
#    print("Batch Imagens:", batch_img.shape)
#    print("Batch Máscaras:", batch_mask.shape)
    # Coloque aqui o código de treino (ex: output = modelo(batch_img))
    # Parar no primeiro batch para testar

# Define o dispositivo de hardware (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinamento rodará no dispositivo: {device}")

# Instancia o modelo e joga para a memória do dispositivo
modelo = UNet(in_channels=42, out_channels=1).to(device)

# Função de Perda (Binary Cross Entropy com Logits)
pesos_positivos = torch.tensor([5.0]).to(device)
funcao_perda = nn.BCEWithLogitsLoss(pos_weight=pesos_positivos)

NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Otimizador Adam (taxa de aprendizado padrão de 1e-4 para segmentação é um bom início)
otimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

# NOVO: Reduz o LR pela metade se a loss de teste não melhorar por 3 épocas seguidas
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode='min', factor=0.5, patience=3, verbose=True)
# Dicionário para guardar as métricas que vão para os gráficos do TCC
historico = {
    "loss_treino": [], 
    "loss_teste": [], 
    "f1_pixel": [], 
    "iou_pixel": [], 
    "mcc_pixel": [],
    "nsr_objeto": [],                # <--- Nova métrica salva
    "precisao_borda_20m": [],        # <--- Nova métrica salva
    "revocacao_borda_20m": [],       # <--- Nova métrica salva
    "distancia_centroide_metros": [], # <--- Nova métrica salva
    "distorcao_forma_npi": []        # <--- Nova métrica salva
}
melhor_loss_teste = float("inf")

print(f"Treino: {len(pares_treino)} | Teste: {len(pares_teste)} | Rodando em: {device}\n")

for epoch in range(NUM_EPOCHS):
    # === PASSO 1: TREINO ===
    modelo.train()
    perda_treino_total = 0
    lotes_treino = 0
    
    gerador_treino = gerador_em_tempo_real(pares_treino, batch_size=BATCH_SIZE, embaralhar=True)
    for batch_img, batch_mask in gerador_treino:
        batch_img, batch_mask = batch_img.to(device), batch_mask.to(device)
        
        otimizador.zero_grad()
        predicoes = modelo(batch_img)
        perda = funcao_perda(predicoes, batch_mask)
        perda.backward()
        otimizador.step()
        
        perda_treino_total += perda.item()
        lotes_treino += 1
        
    perda_media_treino = perda_treino_total / lotes_treino if lotes_treino > 0 else 0

    # === PASSO 2: TESTE/VALIDAÇÃO ===
    modelo.eval() # Desativa o BatchNorm para não interferir nos testes
    perda_teste_total = 0
    lotes_teste = 0
    
    # torch.no_grad garante que o PyTorch não gaste memória calculando gradientes aqui
    with torch.no_grad():
        gerador_teste = gerador_em_tempo_real(pares_teste, batch_size=BATCH_SIZE, embaralhar=False)
        
        metricas_acumuladas = {
            "f1_pixel": [], "iou_pixel": [], "mcc_pixel": [],
            "nsr_objeto": [], "precisao_borda_20m": [], "revocacao_borda_20m": [],
            "distancia_centroide_metros": [], "distorcao_forma_npi": []
        }
        
        for batch_img, batch_mask in gerador_teste:
            batch_img, batch_mask = batch_img.to(device), batch_mask.to(device)
            
            predicoes = modelo(batch_img)
            
            pred_binaria = (torch.sigmoid(predicoes) > 0.5).int()
            mask_binaria = batch_mask.int()
            
            # 1. Calcula métricas por pixel (verifique se as chaves batem com a sua função)
            m_pixel = calcular_metricas_pixel(pred_binaria, mask_binaria)
            metricas_acumuladas["f1_pixel"].append(m_pixel["f1_pixel"])
            metricas_acumuladas["iou_pixel"].append(m_pixel["iou_pixel"])
            metricas_acumuladas["mcc_pixel"].append(m_pixel["mcc_pixel"])
            
            # 2. Calcula métricas por objeto/borda de 20m
            m_obj = calcular_metricas_objeto_e_borda(pred_binaria, mask_binaria)
            metricas_acumuladas["nsr_objeto"].append(m_obj["nsr_objeto"])
            metricas_acumuladas["precisao_borda_20m"].append(m_obj["precisao_borda_20m"])
            metricas_acumuladas["revocacao_borda_20m"].append(m_obj["revocacao_borda_20m"])
            
            # 3. Calcula fidelidade geométrica (executada por amostra representativa)
            m_geom = calcular_fidelidade_geometrica(pred_binaria[0], mask_binaria[0])
            if m_geom["distancia_centroide_metros"] != -1.0: # ignora se não houver talhão detectado
                metricas_acumuladas["distancia_centroide_metros"].append(m_geom["distancia_centroide_metros"])
                metricas_acumuladas["distorcao_forma_npi"].append(m_geom["distorcao_forma_npi"])

            perda = funcao_perda(predicoes, batch_mask)
            perda_teste_total += perda.item()
            lotes_teste += 1
            
        perda_media_teste = perda_teste_total / lotes_teste if lotes_teste > 0 else 0
        scheduler.step(perda_media_teste)
        
        # Guardando todos os dados calculados no histórico de épocas
        historico["loss_treino"].append(perda_media_treino)
        historico["loss_teste"].append(perda_media_teste)
        historico["f1_pixel"].append(np.mean(metricas_acumuladas['f1_pixel']))
        historico["iou_pixel"].append(np.mean(metricas_acumuladas['iou_pixel']))
        historico["mcc_pixel"].append(np.mean(metricas_acumuladas['mcc_pixel']))
        historico["nsr_objeto"].append(np.mean(metricas_acumuladas['nsr_objeto']))
        historico["precisao_borda_20m"].append(np.mean(metricas_acumuladas['precisao_borda_20m']))
        historico["revocacao_borda_20m"].append(np.mean(metricas_acumuladas['revocacao_borda_20m']))
        
        # Garante o append mesmo se nenhuma geometria válida foi detectada no lote
        dist_centroide = np.mean(metricas_acumuladas['distancia_centroide_metros']) if metricas_acumuladas['distancia_centroide_metros'] else 0.0
        dist_forma = np.mean(metricas_acumuladas['distorcao_forma_npi']) if metricas_acumuladas['distorcao_forma_npi'] else 0.0
        historico["distancia_centroide_metros"].append(dist_centroide)
        historico["distorcao_forma_npi"].append(dist_forma)
        
        # --- NOVO: SALVA O HISTÓRICO EM ARQUIVO EM CADA ÉPOCA ---
        with open("historico_treino_tcc_dropout_scheduler.json", "w") as f:
            json.dump(historico, f, indent=4)
        # Tira a média aritmética de tudo para exibir no print da época
        print(f"F1-Score (Pixel): {historico['f1_pixel'][-1]:.4f}")
        print(f"MCC (Pixel): {historico['mcc_pixel'][-1]:.4f}")
        print(f"Razão de Segmentos (NSR): {np.mean(metricas_acumuladas['nsr_objeto']):.4f}")
        print(f"Precisão de Borda (Tolerância 20m): {np.mean(metricas_acumuladas['precisao_borda_20m']):.4f}")
        print(f"Deslocamento do Centroide: {np.mean(metricas_acumuladas['distancia_centroide_metros']):.2f} metros")   
    
    # Print comparativo das duas perdas ao final de cada época
    print(f"Época [{epoch+1}/{NUM_EPOCHS}] -> Perda Treino: {perda_media_treino:.4f} | Perda Teste: {perda_media_teste:.4f}")
    
    # --- AJUSTE: Checkpoint inteligente para salvar o melhor modelo ---
    if perda_media_teste < melhor_loss_teste:
        melhor_loss_teste = perda_media_teste
        torch.save(modelo.state_dict(), "melhor_model_tcc_dropout_scheduler.pt")
        print("   [SALVO] Nova melhor perda em teste encontrada. Pesos atualizados!\n")
    else:
        print("\n")
