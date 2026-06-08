import json
import glob
import os
import matplotlib.pyplot as plt

# 1. Encontrar todos os ficheiros JSON na pasta atual
ficheiros_json = glob.glob("*.json")

if not ficheiros_json:
    print("Nenhum ficheiro JSON encontrado nesta pasta.")
    exit()

# Lista com todas as chaves do JSON e os respetivos títulos para os gráficos
# Formato: (Chave_no_JSON, Título_do_Gráfico, Escala_Y)
metricas_info = [
    ("loss_treino", "Loss de Treino", "linear"),
    ("loss_teste", "Loss de Teste", "linear"),
    ("f1_pixel", "F1-Score (Pixel)", "linear"),
    ("iou_pixel", "IoU (Pixel)", "linear"),
    ("mcc_pixel", "MCC (Pixel)", "linear"),
    ("nsr_objeto", "Razão de Segmentos (NSR)", "log"), # Usando log devido aos valores em bilhões
    ("precisao_borda_20m", "Precisão de Borda (20m)", "linear"),
    ("revocacao_borda_20m", "Revocação de Borda (20m)", "linear"),
    ("distancia_centroide_metros", "Deslocamento do Centroide (Metros)", "linear"),
    ("distorcao_forma_npi", "Distorção de Forma (NPI)", "linear")
]

# 2. Criar a figura com uma grelha 5x2
# O figsize é maior (15 de largura por 20 de altura) para acomodar 10 gráficos confortavelmente
fig, axs = plt.subplots(5, 2, figsize=(15, 20))
fig.suptitle('Análise Completa de Todas as Métricas (Múltiplos Treinos)', fontsize=18, fontweight='bold', y=0.98)

# Transformar a matriz 5x2 de gráficos numa lista simples (1D) para facilitar o loop
axs = axs.flatten()

# 3. Fazer o loop por cada ficheiro encontrado
for ficheiro in ficheiros_json:
    with open(ficheiro, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    # Extrair o nome do ficheiro (ex: 'modelo_1')
    nome_legenda = os.path.splitext(os.path.basename(ficheiro))[0]
    
    # Obter o número de épocas usando o tamanho da primeira lista
    epocas = range(1, len(dados['loss_treino']) + 1)
    
    # Plotar cada métrica no seu respetivo subplot
    for i, (chave, titulo, escala) in enumerate(metricas_info):
        # Evitar erros caso alguma métrica não exista num ficheiro antigo
        if chave in dados:
            axs[i].plot(epocas, dados[chave], marker='o', markersize=4, label=nome_legenda)

# 4. Configurar os títulos, eixos e legendas de cada um dos 10 gráficos
for i, (chave, titulo, escala) in enumerate(metricas_info):
    axs[i].set_title(titulo, fontsize=12)
    axs[i].set_xlabel('Época')
    axs[i].set_ylabel('Valor')
    axs[i].set_yscale(escala) # Aplica escala 'log' para o NSR e 'linear' para os outros
    axs[i].grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar legenda (colocar apenas se houver ficheiros)
    if ficheiros_json:
        axs[i].legend(fontsize='small')

# 5. Ajustar o espaçamento entre os gráficos para os textos não se sobreporem
plt.tight_layout(rect=[0, 0, 1, 0.97]) # Deixa espaço no topo para o título principal

# Guardar a imagem em alta resolução
nome_imagem = 'todas_as_metricas_comparacao.png'
plt.savefig(nome_imagem, dpi=300)
print(f"Gráfico gerado com sucesso! Foram processados {len(ficheiros_json)} ficheiros(s).")
print(f"Imagem guardada como: {nome_imagem}")

# Exibir a janela com os gráficos
plt.show()