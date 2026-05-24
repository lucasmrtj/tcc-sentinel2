import torch
import torch.nn as nn



class BlocoDuploConv(nn.Module):
    """(Convolução -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




class UNet(nn.Module):
    def __init__(self, in_channels=42, out_channels=1):
        super().__init__()

        # --- ENCODER (Caminho de Contração) ---
        self.inc = BlocoDuploConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), BlocoDuploConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), BlocoDuploConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), BlocoDuploConv(256, 512))
        
        # --- BOTTLENECK (Gargalo) ---
        # Na classe UNet, altere apenas o self.down4:
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), 
            BlocoDuploConv(512, 1024),
            nn.Dropout2d(0.5) # <--- ADICIONE AQUI
        )

        # --- DECODER (Caminho de Expansão + Skip Connections) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = BlocoDuploConv(1024, 512) # 512 (up) + 512 (skip x4) = 1024

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = BlocoDuploConv(512, 256) # 256 (up) + 256 (skip x3) = 512

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = BlocoDuploConv(256, 128) # 128 (up) + 128 (skip x2) = 256

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = BlocoDuploConv(128, 64)   # 64 (up) + 64 (skip x1) = 126

        # --- CAMADA DE SAÍDA (Mapeamento Binário) ---
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Se o tensor vier em 5D [B, T, C, H, W], combinamos T e C em uma única dimensão de canais
        if len(x.shape) == 5:
            b, t, c, h, w = x.shape
            x = x.view(b, t * c, h, w)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder com conexões residuais (Skip Connections)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)
        
        logit_saida = self.outc(x)
        return logit_saida

# Instanciando o modelo (configurado para receber 7 meses * 6 canais = 42 canais)
modelo = UNet(in_channels=42, out_channels=1)

# Criando um lote fictício simulando exatamente o comportamento do seu gerador atual
batch_imagens_teste = torch.randn(8, 7, 6, 256, 256) 

# Passando o lote pela rede
print("Passando os dados pela U-Net...")
resultado = modelo(batch_imagens_teste)

print("\n--- Validação de Formatos ---")
print("Input vindo do gerador: ", batch_imagens_teste.shape)
print("Output gerado pela rede:", resultado.shape) 
print("Formato esperado da Máscara:", "torch.Size([8, 1, 256, 256])")
