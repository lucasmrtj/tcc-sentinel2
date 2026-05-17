import torch
import numpy as np
import torch.nn.functional as F
from rasterio.features import shapes
from shapely.geometry import shape, Polygon


def calcular_metricas_pixel(pred_bin, true_bin):
    """
    Calcula as métricas tradicionais ponderadas por pixel:
    Precisão, Revocação, F1-Score, IoU e Coeficiente de Matthews (MCC).
    """
    tp = torch.sum((pred_bin == 1) & (true_bin == 1)).item()
    fp = torch.sum((pred_bin == 1) & (true_bin == 0)).item()
    fn = torch.sum((pred_bin == 0) & (true_bin == 1)).item()
    tn = torch.sum((pred_bin == 0) & (true_bin == 0)).item()
    
    precisao = tp / (tp + fp + 1e-8)
    revocacao = tp / (tp + fn + 1e-8)
    f1 = 2 * (precisao * revocacao) / (precisao + revocacao + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Coeficiente de Correlação de Matthews (MCC)
    num_mcc = (tp * tn) - (fp * fn)
    den_mcc = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num_mcc / (den_mcc + 1e-8)
    
    return {
        "precisao_pixel": precisao,
        "revocacao_pixel": revocacao,
        "f1_pixel": f1,
        "iou_pixel": iou,
        "mcc_pixel": mcc
    }

def calcular_metricas_objeto_e_borda(pred_bin, true_bin):
    """
    Mede Erro de Segmentação Potencial (PSE), Razão de Segmentos (NSR)
    e a Precisão/Revocação de borda com tolerância de 20 metros (2 pixels).
    """
    # 1. Erro de Segmentação Potencial (Proporção da referência fora da predição)
    area_ref = torch.sum(true_bin == 1).item()
    area_omitida = torch.sum((true_bin == 1) & (pred_bin == 0)).item()
    pse = area_omitida / (area_ref + 1e-8)
    
    # 2. Razão do Número de Segmentos (NSR) via contagem de polígonos do rasterio
    pred_np = pred_bin.squeeze().cpu().numpy().astype(np.int32)
    true_np = true_bin.squeeze().cpu().numpy().astype(np.int32)
    
    num_seg_pred = len(list(shapes(pred_np, mask=(pred_np == 1))))
    num_seg_true = len(list(shapes(true_np, mask=(true_np == 1))))
    nsr = num_seg_pred / (num_seg_true + 1e-8)
    
    # 3. Extração de Bordas (Fronteira = Máscara - Erosão)
    def extrair_borda(mask):
        padding = 1
        eroded = 1 - F.max_pool2d(1 - mask.float(), kernel_size=3, stride=1, padding=padding)
        return (mask - eroded) > 0

    borda_pred = extrair_borda(pred_bin)
    borda_true = extrair_borda(true_bin)
    
    # Dilação de 2 pixels (Tolerância de 20 metros para o Sentinel-2)
    zona_tol_pred = F.max_pool2d(borda_pred.float(), kernel_size=5, stride=1, padding=2) > 0
    zona_tol_true = F.max_pool2d(borda_true.float(), kernel_size=5, stride=1, padding=2) > 0
    
    # Cálculo das precisões com tolerância geográfica
    tp_borda_rev = torch.sum(borda_true & zona_tol_pred).item()
    revocacao_borda_20m = tp_borda_rev / (torch.sum(borda_true).item() + 1e-8)
    
    tp_borda_prec = torch.sum(borda_pred & zona_tol_true).item()
    precisao_borda_20m = tp_borda_prec / (torch.sum(borda_pred).item() + 1e-8)
    
    return {
        "pse_objeto": pse, # <--- CORRIGIDO: Agora retorna o número decimal do erro calculado!
        "nsr_objeto": nsr,
        "precisao_borda_20m": precisao_borda_20m,
        "revocacao_borda_20m": revocacao_borda_20m
    }

def calcular_fidelidade_geometrica(pred_bin, true_bin):
    """
    Mede a acurácia geométrica tridimensional: Área, Posição (Centroide) e Forma (NPI)
    """
    pred_np = pred_bin.squeeze().cpu().numpy().astype(np.int32)
    true_np = true_bin.squeeze().cpu().numpy().astype(np.int32)
    try: 
        polys_pred = [shape(s) for s, v in shapes(pred_np, mask=(pred_np == 1)) if shape(s).is_valid]
        polys_true = [shape(s) for s, v in shapes(true_np, mask=(true_np == 1)) if shape(s).is_valid]
        
        if len(polys_pred) == 0 or len(polys_true) == 0:
            return {"erro_area_proporcional": 1.0, "distancia_centroide_metros": -1.0, "npi_distorcao": 1.0}
        
        # Agrupa múltiplos fragmentos em um único objeto consolidado se necessário
        geom_pred = polys_pred[0]
        for p in polys_pred[1:]: geom_pred = geom_pred.union(p)
        geom_true = polys_true[0]
        for p in polys_true[1:]: geom_true = geom_true.union(p)
        
        # A. Acurácia de Área
        area_pred = geom_pred.area
        area_true = geom_true.area
        erro_area = abs(area_pred - area_true) / area_true
        
        # B. Acurácia de Posição (Distância euclidiana entre centroides convertida para metros)
        dist_centroide_pixels = geom_pred.centroid.distance(geom_true.centroid)
        dist_centroide_metros = dist_centroide_pixels * 10.0 # 1 pixel = 10m
        
        # C. Acurácia de Forma: Índice de Perímetro Normalizado (NPI)
        # NPI = (2 * sqrt(pi * Area)) / Perimetro
        npi_pred = (2 * np.sqrt(np.pi * area_pred)) / (geom_pred.length + 1e-8)
        npi_true = (2 * np.sqrt(np.pi * area_true)) / (geom_true.length + 1e-8)
        distorcao_forma = abs(npi_pred - npi_true)
        
        return {
            "erro_area_proporcional": erro_area,
            "distancia_centroide_metros": dist_centroide_metros,
            "npi_modelo": npi_pred,
            "npi_referencia": npi_true,
            "distorcao_forma_npi": distorcao_forma
        }
    except:
        return {"err_area": 1.0, "dist_cent": -1.0, "dist_forma": 1.0}
