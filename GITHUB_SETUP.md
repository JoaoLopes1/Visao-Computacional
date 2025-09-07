# Instruções para Configuração no GitHub

## 📋 Checklist de Arquivos

✅ **Arquivos Essenciais Mantidos:**
- `Prova 02-B.py` - Script principal de detecção de movimento
- `speech_detection_final.py` - Script de detecção de fala (versão final)
- `README.md` - Documentação completa do projeto
- `requirements.txt` - Dependências do projeto
- `LICENSE` - Licença MIT
- `.gitignore` - Configuração do Git

✅ **Vídeos de Entrada:**
- `people-talking.mp4` - Vídeo de pessoas falando
- `people-walking.mp4` - Vídeo de pessoas caminhando
- `yolov8n.pt` - Modelo YOLOv8

✅ **Vídeos de Saída (Exemplos):**
- `people-talking-final-sensitive.mp4` - Resultado da detecção de fala
- `people-walking-final.mp4` - Resultado da detecção de movimento

## 🚀 Passos para Upload no GitHub

### 1. Criar Repositório no GitHub
1. Acesse [github.com](https://github.com)
2. Clique em "New repository"
3. Nome sugerido: `speech-movement-detection`
4. Descrição: "Sistema de detecção de fala e movimento usando visão computacional e análise de áudio"
5. Marque como "Public" ou "Private" conforme preferir
6. **NÃO** inicialize com README (já temos um)

### 2. Configurar Git Local (se disponível)
```bash
# Se o Git estiver instalado
git init
git add .
git commit -m "Initial commit: Sistema de detecção de fala e movimento"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/speech-movement-detection.git
git push -u origin main
```

### 3. Upload Manual (Alternativa)
Se o Git não estiver disponível:
1. Baixe todos os arquivos do projeto
2. No GitHub, use "Upload files" ou "Add file"
3. Faça upload de todos os arquivos listados acima

## 📁 Estrutura Final do Repositório

```
speech-movement-detection/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── GITHUB_SETUP.md
├── Prova 02-B.py
├── speech_detection_final.py
├── people-talking.mp4
├── people-walking.mp4
├── yolov8n.pt
├── people-talking-final-sensitive.mp4
└── people-walking-final.mp4
```

## 🔧 Configurações Recomendadas

### GitHub Pages (Opcional)
Para documentação online:
1. Vá em Settings > Pages
2. Source: Deploy from a branch
3. Branch: main
4. Folder: / (root)

### Issues e Pull Requests
- Habilite Issues para feedback
- Configure templates se necessário
- Defina labels para organização

### Proteções de Branch
- Habilite branch protection para main
- Requira pull request reviews
- Habilite status checks

## 📝 Notas Importantes

### Tamanho dos Arquivos
- **Vídeos**: Podem ser grandes (5-7MB cada)
- **Modelo YOLO**: ~6.5MB
- **Total estimado**: ~25MB

### Alternativas para Vídeos Grandes
Se os vídeos forem muito grandes para o GitHub:
1. Use Git LFS (Large File Storage)
2. Hoste os vídeos em outro serviço (Google Drive, Dropbox)
3. Inclua apenas vídeos de exemplo menores

### Dependências
O `requirements.txt` inclui todas as dependências necessárias:
```bash
pip install -r requirements.txt
```

## ✅ Verificação Final

Antes de fazer upload, verifique:
- [ ] Todos os arquivos de teste foram removidos
- [ ] README.md está completo e atualizado
- [ ] .gitignore está configurado corretamente
- [ ] requirements.txt inclui todas as dependências
- [ ] LICENSE está presente
- [ ] Vídeos de exemplo estão incluídos
- [ ] Código está limpo e comentado

## 🎯 Próximos Passos

Após o upload:
1. Teste a instalação seguindo o README
2. Adicione badges de status (se aplicável)
3. Configure GitHub Actions para CI/CD (opcional)
4. Adicione exemplos de uso
5. Solicite feedback da comunidade
