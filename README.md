# Sistema de Detecção de Fala e Movimento

Este projeto implementa um sistema de análise de vídeo que detecta pessoas falando e caminhando usando técnicas de visão computacional e análise de áudio.

## 🎯 Funcionalidades

### Detecção de Fala
- **Análise Visual**: Detecção de movimento da boca usando múltiplas técnicas
- **Análise de Áudio**: Extração e análise de características de áudio (RMS, ZCR, Spectral features)
- **Correlação Áudio-Visual**: Combinação de sinais visuais e auditivos para maior precisão
- **Sistema de Cores**:
  - 🟢 **Verde**: Alta confiança (áudio + visual)
  - 🟡 **Amarelo**: Confiança média (áudio OU visual)
  - ⚪ **Cinza**: Face detectada mas não falando

### Detecção de Movimento
- **Rastreamento de Pessoas**: Usando YOLOv8 para detecção e rastreamento
- **Análise de Movimento**: Cálculo de velocidade e direção
- **Classificação**: Identificação de pessoas caminhando vs. paradas

## 📁 Estrutura do Projeto

```
├── detecçao_de_mov.py               # Script principal (detecção de movimento)
├── deteccao_de_fala.py.py           # Script de detecção de fala (versão final)
├── people-talking.mp4               # Vídeo de entrada (pessoas falando)
├── people-walking.mp4               # Vídeo de entrada (pessoas caminhando)
├── yolov8n.pt                       # Modelo YOLOv8 para detecção de pessoas
├── requirements.txt                 # Dependências do projeto
├── LICENSE                          # Licença MIT
└── README.md                        # Este arquivo
```

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install opencv-python ultralytics numpy librosa soundfile scipy
```

### Detecção de Movimento (Pessoas Caminhando)

```bash
python "detecçao_de_mov.py"
```

Este script processa o vídeo `people-walking.mp4` e gera `people-walking-final.mp4` com:
- Detecção de pessoas usando YOLOv8
- Rastreamento de movimento
- Cálculo de velocidade e direção
- Classificação de movimento

### Detecção de Fala (Pessoas Falando)

```bash
python speech_detection_final.py
```

Este script processa o vídeo `people-talking.mp4` e gera `people-talking-final-sensitive.mp4` com:
- Detecção de faces usando Haar Cascade
- Análise de movimento da boca
- Extração e análise de áudio
- Correlação áudio-visual

## 🔧 Técnicas Implementadas

### Detecção Visual de Fala
1. **Densidade de Bordas**: Análise de bordas na região da boca
2. **Área de Contornos**: Cálculo de área de contornos detectados
3. **Variação de Intensidade**: Coeficiente de variação de pixels
4. **Análise de Gradientes**: Detecção de movimento usando Sobel
5. **Variação de Textura**: Análise de variância de pixels

### Análise de Áudio
1. **RMS Energy**: Energia do sinal de áudio
2. **Zero Crossing Rate (ZCR)**: Taxa de cruzamento por zero
3. **Spectral Centroid**: Centro de massa do espectro
4. **Spectral Entropy**: Entropia espectral
5. **Spectral Bandwidth**: Largura do espectro

### Detecção de Movimento
1. **YOLOv8**: Detecção de pessoas em tempo real
2. **Rastreamento**: Associação de detecções entre frames
3. **Cálculo de Velocidade**: Análise de movimento temporal
4. **Classificação**: Identificação de padrões de movimento

## ⚙️ Configurações

### Thresholds de Detecção de Fala
```python
# Configurações ajustáveis no notebook
audio_thresholds = {
    'min_rms': 0.005,
    'min_zcr': 0.03,
    'max_zcr': 0.4,
    'min_spectral_centroid': 800,
    'min_spectral_entropy': 1.5,
    'min_spectral_bandwidth': 300
}

visual_thresholds = {
    'edge_density': 0.008,
    'contour_area': 0.003,
    'intensity_cv': 0.08,
    'gradient_mean': 8,
    'texture_variance': 80
}
```

## 📊 Resultados

### Vídeos de Saída
- `people-walking-final.mp4`: Vídeo com detecção de movimento
- `people-talking-final-sensitive.mp4`: Vídeo com detecção de fala

### Métricas de Performance
- **Detecção de Fala**: 5 técnicas visuais + análise de áudio
- **Sensibilidade**: Thresholds baixos para capturar movimento sutil
- **Precisão**: Correlação áudio-visual para reduzir falsos positivos

## 🛠️ Desenvolvimento

### Estrutura Modular
- Funções separadas para cada técnica
- Configurações centralizadas
- Fácil extensão e modificação
- Código limpo e bem documentado

## 📝 Notas Técnicas

### Dependências de Áudio (Opcionais)
- **librosa**: Análise de características de áudio (se disponível)
- **soundfile**: Leitura de arquivos de áudio (se disponível)
- **scipy**: Processamento de sinais (se disponível)

### Dependências de Visão
- **opencv-python**: Processamento de imagem e vídeo
- **ultralytics**: YOLOv8 para detecção de objetos
- **numpy**: Computação numérica

### Limitações
- Detecção visual depende da qualidade da iluminação
- Performance varia com resolução do vídeo
- Análise de áudio é opcional (funciona apenas com detecção visual)

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Contato

Para dúvidas ou sugestões, abra uma issue no GitHub.
