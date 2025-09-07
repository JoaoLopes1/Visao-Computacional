# Detecção de Fala Visual - Versão Final
import cv2
import numpy as np
from collections import deque
import os
import warnings
import logging

# Configurar warnings e logs
warnings.filterwarnings('ignore')

print("✅ Bibliotecas importadas com sucesso!")

# Verificar se o vídeo people-talking.mp4 existe
video_path = 'people-talking.mp4'

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        
        print(f"✅ {video_path} encontrado:")
        print(f"   - Resolução: {width}x{height}")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Duração: {duration:.2f} segundos")
        print(f"   - Total de frames: {frame_count}")
        cap.release()
    else:
        print(f"❌ Erro ao abrir {video_path}")
else:
    print(f"❌ Arquivo {video_path} não encontrado")

# Funções de detecção de faces
def detect_faces_haar(frame):
    """Detecta faces usando Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    return [(x, y, w, h) for (x, y, w, h) in faces]

# Funções de detecção de fala visual
def detect_faces_and_mouths(frame):
    """Detecção SIMPLES mas EFICAZ de movimento da boca"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_haar(frame)
    face_info = []
    
    for (x, y, w, h) in faces:
        # Região da face
        face_roi = gray[y:y+h, x:x+w]
        
        # Detectar boca na região inferior da face (60-90% da altura)
        mouth_region_y = int(h * 0.6)
        mouth_region_h = int(h * 0.3)
        mouth_roi = face_roi[mouth_region_y:mouth_region_y+mouth_region_h, :]
        
        if mouth_roi.size == 0:
            continue
        
        # 1. Detectar bordas com thresholds BAIXOS
        mouth_edges = cv2.Canny(mouth_roi, 10, 30)  # Sensível
        
        # 2. Calcular densidade de bordas
        edge_density = np.sum(mouth_edges > 0) / (mouth_roi.shape[0] * mouth_roi.shape[1])
        
        # 3. Detectar contornos
        contours, _ = cv2.findContours(mouth_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        normalized_contour_area = total_contour_area / (mouth_roi.shape[0] * mouth_roi.shape[1])
        
        # 4. Análise de variação de intensidade
        intensity_variance = np.var(mouth_roi.astype(np.float32))
        intensity_mean = np.mean(mouth_roi.astype(np.float32))
        intensity_cv = intensity_variance / (intensity_mean + 1e-10)
        
        # 5. Análise de gradientes
        grad_x = cv2.Sobel(mouth_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mouth_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        # 6. Análise de textura
        texture_variance = np.var(mouth_roi)
        
        # CRITÉRIOS SENSÍVEIS - Detectar movimento na boca
        is_talking = (
            edge_density > 0.008 or  # Baixo threshold
            normalized_contour_area > 0.003 or  # Baixo threshold
            intensity_cv > 0.08 or  # Baixo threshold
            gradient_mean > 8 or  # Baixo threshold
            texture_variance > 80  # Baixo threshold
        )
        
        # Calcular confiança baseada em quantos critérios foram atendidos
        criteria_met = sum([
            edge_density > 0.008,
            normalized_contour_area > 0.003,
            intensity_cv > 0.08,
            gradient_mean > 8,
            texture_variance > 80
        ])
        
        confidence = criteria_met / 5.0  # 5 critérios no total
        
        face_info.append({
            'bbox': (x, y, w, h),
            'mouth_region': (x, y + mouth_region_y, w, mouth_region_h),
            'edge_density': edge_density,
            'contour_area': normalized_contour_area,
            'intensity_cv': intensity_cv,
            'gradient_mean': gradient_mean,
            'texture_variance': texture_variance,
            'confidence': confidence,
            'is_talking': is_talking
        })
    
    return face_info

def analyze_talking_pattern(face_history):
    """Análise temporal SIMPLES"""
    if len(face_history) < 2:
        return False
    
    # Calcular variância de características
    confidences = [face['confidence'] for face in face_history]
    edge_densities = [face['edge_density'] for face in face_history]
    
    confidence_variance = np.var(confidences)
    edge_variance = np.var(edge_densities)
    
    # CRITÉRIOS SENSÍVEIS
    is_talking = (
        confidence_variance > 0.0005 or  # Baixo threshold
        edge_variance > 0.0005 or  # Baixo threshold
        np.mean(confidences) > 0.15  # Se a confiança média é positiva
    )
    
    return is_talking

# Função principal de processamento
def process_talking_video(input_path, output_path):
    """Processa vídeo com detecção visual sensível"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir {input_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    face_histories = {}
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🔄 Processando {input_path} com detecção visual sensível...")
    print(f"   Total de frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Processando frame {frame_count}/{total_frames}")
        
        # Detectar faces
        face_detections = detect_faces_and_mouths(frame)
        
        # Atualizar histórico de faces
        for face_info in face_detections:
            face_bbox = face_info['bbox']
            face_id = f"{face_bbox[0]}_{face_bbox[1]}"
            
            if face_id not in face_histories:
                face_histories[face_id] = deque(maxlen=10)
            
            face_histories[face_id].append(face_info)
        
        # Criar frame de saída
        output_frame = frame.copy()
        
        # Calcular tempo atual do frame
        current_time = frame_count / fps
        
        # Desenhar faces falando
        for face_info in face_detections:
            face_bbox = face_info['bbox']
            mouth_bbox = face_info['mouth_region']
            
            # Verificar se está falando (visual)
            face_id = f"{face_bbox[0]}_{face_bbox[1]}"
            if face_id in face_histories:
                visual_talking = analyze_talking_pattern(list(face_histories[face_id]))
            else:
                visual_talking = face_info['is_talking']
            
            # Usar confiança para determinar cor
            confidence = face_info['confidence']
            
            if visual_talking and confidence > 0.4:
                # Alta confiança - Verde
                color = (0, 255, 0)
                label = f"TALKING (High: {confidence:.2f})"
                thickness = 3
            elif visual_talking or confidence > 0.2:
                # Confiança média - Amarelo (SENSÍVEL)
                color = (0, 255, 255)
                label = f"TALKING (Med: {confidence:.2f})"
                thickness = 2
            else:
                # Face detectada mas não falando - Cinza
                color = (128, 128, 128)
                label = "Face"
                thickness = 2
            
            # Desenhar retângulos
            cv2.rectangle(output_frame, face_bbox, color, thickness)
            cv2.putText(output_frame, label, (face_bbox[0], face_bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Destacar região da boca se está falando
            if visual_talking or confidence > 0.2:
                cv2.rectangle(output_frame, mouth_bbox, color, 1)
        
        # Info do frame
        info_text = f"Sensitive Visual Detection - Time: {current_time:.1f}s"
        cv2.putText(output_frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(output_frame)
    
    cap.release()
    out.release()
    
    print(f"✅ Vídeo processado salvo em: {output_path}")
    return True

# Executar processamento
print("=" * 60)
print("DETECÇÃO DE FALA VISUAL - VERSÃO FINAL")
print("=" * 60)

# Processar vídeo de pessoas falando
print("\n🔄 Processando people-talking.mp4 (SENSÍVEL)...")
success_talking = process_talking_video(
    input_path='people-talking.mp4',
    output_path='people-talking-final-sensitive.mp4'
)

print("\n" + "=" * 60)
print("RESUMO")
print("=" * 60)

if success_talking:
    print("✅ people-talking-final-sensitive.mp4 - Processado com sucesso!")
    print("   • Detecção SENSÍVEL de fala")
    print("   • Verde: Alta confiança (confiança > 0.4)")
    print("   • Amarelo: Confiança média (confiança > 0.2)")
    print("   • Cinza: Face detectada mas não falando")
    print("   • 5 técnicas de detecção visual")
    print("   • Thresholds baixos para maior sensibilidade")
else:
    print("❌ Erro ao processar people-talking.mp4")

print("\n🎯 Melhorias implementadas:")
print("   • 5 técnicas de detecção visual")
print("   • Thresholds baixos para maior sensibilidade")
print("   • Confiança baseada em múltiplos critérios")
print("   • Detecção de movimento na boca")
print("   • Análise temporal simples mas eficaz")
