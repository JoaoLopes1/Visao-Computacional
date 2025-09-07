# Detecção de Pessoas Caminhando com Visão Computacional
import cv2
import numpy as np
from collections import deque
import os
import warnings
import logging

# Configurar warnings e logs
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Instalar e importar YOLO
try:
    from ultralytics import YOLO
    print("✅ YOLO importado com sucesso!")
except ImportError:
    print("Instalando ultralytics...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics"])
    from ultralytics import YOLO
    print("✅ YOLO instalado e importado com sucesso!")

print("✅ Todas as bibliotecas importadas com sucesso!")

# Verifica se o arquivo de vídeo existe e está acessível
def verificar_video():
    """Verifica se o arquivo de vídeo existe e está acessível"""
    video_file = 'people-walking.mp4'
    
    if os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps
            
            print(f"✅ {video_file}:")
            print(f"   - Resolução: {width}x{height}")
            print(f"   - FPS: {fps:.2f}")
            print(f"   - Duração: {duration:.2f} segundos")
            print(f"   - Total de frames: {frame_count}")
            cap.release()
        else:
            print(f"❌ Erro ao abrir {video_file}")
    else:
        print(f"❌ Arquivo {video_file} não encontrado")

# Executar verificação
verificar_video()

# Carregamento do modelo YOLO e funções de detecção
print("Carregando modelo YOLOv8...")
model = YOLO('yolov8n.pt')  # YOLOv8 nano - mais rápido
print("✅ Modelo YOLO carregado com sucesso!")

def detect_people_yolo(frame):
    """Detecta pessoas usando YOLOv8"""
    results = model(frame, verbose=False, save=False, show=False, conf=0.5)
    people_boxes = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls) == 0:  # Classe 'person' no COCO dataset
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    if conf > 0.5:
                        people_boxes.append((x, y, w, h, conf))
    
    return people_boxes

print("✅ Funções de detecção criadas!")

# Classe para rastreamento de pessoas
class PersonTracker:
    """Classe para rastrear pessoas entre frames e analisar movimento"""
    
    def __init__(self, max_disappeared=10):
        self.next_person_id = 0
        self.persons = {}
        self.max_disappeared = max_disappeared
        self.disappeared = {}
        
    def register(self, centroid, bbox):
        """Registra uma nova pessoa"""
        self.persons[self.next_person_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'movement_history': deque(maxlen=30),
            'is_walking': False
        }
        self.disappeared[self.next_person_id] = 0
        self.next_person_id += 1
        
    def deregister(self, person_id):
        """Remove uma pessoa do rastreamento"""
        del self.persons[person_id]
        del self.disappeared[person_id]
        
    def update(self, detections):
        """Atualiza o rastreamento com novas detecções"""
        if len(detections) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
            return
            
        # Calcular centroides das detecções
        input_centroids = []
        for (x, y, w, h, conf) in detections:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))
            
        # Se não há pessoas rastreadas, registrar todas as detecções
        if len(self.persons) == 0:
            for i, (centroid, detection) in enumerate(zip(input_centroids, detections)):
                self.register(centroid, detection)
        else:
            # Associar detecções existentes com pessoas rastreadas
            person_ids = list(self.persons.keys())
            person_centroids = [self.persons[pid]['centroid'] for pid in person_ids]
            
            # Calcular distâncias entre centroides existentes e novos
            D = np.linalg.norm(np.array(person_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
            
            # Encontrar associações
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if D[row, col] > 100:  # Threshold de distância máxima
                    continue
                    
                person_id = person_ids[row]
                old_centroid = self.persons[person_id]['centroid']
                new_centroid = input_centroids[col]
                
                # Atualizar posição e calcular movimento
                self.persons[person_id]['centroid'] = new_centroid
                self.persons[person_id]['bbox'] = detections[col]
                
                # Calcular velocidade
                dx = new_centroid[0] - old_centroid[0]
                dy = new_centroid[1] - old_centroid[1]
                speed = np.sqrt(dx**2 + dy**2)
                
                # Adicionar ao histórico de movimento
                self.persons[person_id]['movement_history'].append((dx, dy, speed))
                
                # Analisar padrão de movimento
                self._analyze_movement(person_id)
                
                used_row_indices.add(row)
                used_col_indices.add(col)
                
            # Registrar novas pessoas não associadas
            unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    person_id = person_ids[row]
                    self.disappeared[person_id] += 1
                    if self.disappeared[person_id] > self.max_disappeared:
                        self.deregister(person_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], detections[col])
                    
    def _analyze_movement(self, person_id):
        """Analisa o padrão de movimento para determinar se a pessoa está caminhando"""
        history = self.persons[person_id]['movement_history']
        
        if len(history) < 5:
            return
            
        # Calcular velocidade média
        speeds = [mov[2] for mov in history]
        avg_speed = np.mean(speeds)
        
        # Calcular variância da velocidade
        speed_variance = np.var(speeds)
        
        # Calcular direção do movimento
        dx_values = [mov[0] for mov in history]
        dy_values = [mov[1] for mov in history]
        
        # Verificar se há movimento consistente em uma direção
        dx_consistency = np.std(dx_values) < 20
        dy_consistency = np.std(dy_values) < 20
        
        # Critérios para caminhada
        is_walking = (2 < avg_speed < 15 and 
                     speed_variance < 50 and 
                     (dx_consistency or dy_consistency))
        
        self.persons[person_id]['is_walking'] = is_walking
        
    def get_persons(self):
        """Retorna informações sobre todas as pessoas rastreadas"""
        return self.persons

print("✅ Classe PersonTracker criada!")

# Função principal de processamento
def process_video_walking(input_path, output_path):
    """Processa vídeo com detecção de movimento de pessoas caminhando"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir {input_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = PersonTracker()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🔄 Processando {input_path}...")
    print(f"   Total de frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Processando frame {frame_count}/{total_frames}")
        
        # Detectar pessoas
        people_detections = detect_people_yolo(frame)
        tracker.update(people_detections)
        
        # Criar frame de saída
        output_frame = frame.copy()
        
        # Desenhar pessoas
        persons = tracker.get_persons()
        for person_id, person_data in persons.items():
            x, y, w, h, conf = person_data['bbox']
            
            if person_data['is_walking']:
                color = (0, 255, 0)  # Verde para caminhada
                label = f"Walking (ID: {person_id})"
            else:
                color = (255, 0, 0)  # Azul para parado
                label = f"Person (ID: {person_id})"
            
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info do frame
        info_text = "Processed: Walking Detection - YOLO Movement Analysis"
        cv2.putText(output_frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(output_frame)
    
    cap.release()
    out.release()
    
    print(f"✅ Vídeo processado salvo em: {output_path}")
    return True

print("✅ Função de processamento criada!")

# Execução do processamento
print("=" * 60)
print("DETECÇÃO DE MOVIMENTO - PESSOAS CAMINHANDO")
print("=" * 60)

# Processar vídeo de pessoas caminhando
print("\n🔄 Processando people-walking.mp4...")
success_walking = process_video_walking(
    input_path='people-walking.mp4',
    output_path='people-walking-final.mp4'
)

print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)

if success_walking:
    print("✅ people-walking-final.mp4 - Processado com sucesso!")
    print("   • Detecção de movimento de pessoas caminhando")
    print("   • Pessoas caminhando: Verde")
    print("   • Pessoas paradas: Azul")
else:
    print("❌ Erro ao processar people-walking.mp4")

print("\n🎯 Funcionalidades implementadas:")
print("   • Detecção de pessoas com YOLOv8")
print("   • Rastreamento de pessoas entre frames")
print("   • Análise de movimento para identificar caminhada")
print("   • Visualização com cores diferenciadas")
print("   • Código otimizado e focado apenas em movimento")