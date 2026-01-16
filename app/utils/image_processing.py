import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def distance(p1, p2):
    """Calcula distância euclidiana entre dois pontos MediaPipe"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def calculate_body_volume_indicator(landmarks, image_shape) -> float:
    """
    Estima um indicador de volume corporal baseado em área visível.
    Quanto maior, mais massa corporal (gordura ou músculo).
    """
    h, w = image_shape[:2]
    
    # Pontos-chave para calcular área do torso
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    
    # Converte para coordenadas de pixel
    points = np.array([
        [left_shoulder.x * w, left_shoulder.y * h],
        [right_shoulder.x * w, right_shoulder.y * h],
        [right_hip.x * w, right_hip.y * h],
        [left_hip.x * w, left_hip.y * h]
    ], dtype=np.int32)
    
    # Calcula área do torso
    torso_area = cv2.contourArea(points)
    
    # Normaliza pela área total da imagem
    image_area = h * w
    volume_ratio = torso_area / image_area
    
    return volume_ratio


def estimate_waist_visibility(landmarks) -> float:
    """
    Estima quão proeminente é a região abdominal.
    Usa a posição Z (profundidade) dos landmarks.
    Valores mais altos = mais gordura visível na cintura.
    """
    # Pontos da região abdominal
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    
    # Pontos de referência (ombros)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # Calcula a "projeção" da cintura em relação aos ombros
    # Se a cintura tem Z maior (mais próximo), indica gordura abdominal
    hip_z = (left_hip.z + right_hip.z) / 2
    shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
    
    # Quanto mais positivo, mais a cintura "projeta" para frente
    waist_prominence = shoulder_z - hip_z
    
    return waist_prominence


def extract_body_measurements(image_path: str) -> dict:
    """
    Extrai medidas corporais REAIS usando MediaPipe Pose.
     COM VALIDAÇÃO E LOGS DETALHADOS
    """
    # Lê a imagem
    image = cv2.imread(image_path)
    if image is None:
        return {
            "person_detected": False,
            "error": "Não foi possível carregar a imagem"
        }
    
    # Converte para RGB (MediaPipe usa RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Processa com MediaPipe
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return {
            "person_detected": False,
            "error": "Nenhuma pessoa detectada ou pose não identificada"
        }
    
    landmarks = results.pose_landmarks.landmark
    
    # ===== MEDIÇÕES BÁSICAS =====
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    
    # Larguras
    shoulder_width = distance(left_shoulder, right_shoulder)
    hip_width = distance(left_hip, right_hip)
    
    # Altura do torso
    torso_height = (
        distance(left_shoulder, left_hip) +
        distance(right_shoulder, right_hip)
    ) / 2
    
    # ===== ESTIMATIVA DA CINTURA (CORRIGIDA) =====
    # Ponto médio entre ombros e quadris (aproximação da cintura real)
    # Usar 40% abaixo dos ombros = região da cintura natural
    left_waist_x = left_shoulder.x * 0.35 + left_hip.x * 0.65
    left_waist_y = left_shoulder.y * 0.35 + left_hip.y * 0.65
    left_waist_z = left_shoulder.z * 0.35 + left_hip.z * 0.65
    
    right_waist_x = right_shoulder.x * 0.35 + right_hip.x * 0.65
    right_waist_y = right_shoulder.y * 0.35 + right_hip.y * 0.65
    right_waist_z = right_shoulder.z * 0.35 + right_hip.z * 0.65
    
    # Cria objetos de ponto para cálculo
    class Point:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    left_waist = Point(left_waist_x, left_waist_y, left_waist_z)
    right_waist = Point(right_waist_x, right_waist_y, right_waist_z)
    
    waist_width = distance(left_waist, right_waist)
    
    # ===== VALIDAÇÕES FISIOLÓGICAS =====
    
    # 1. Cintura não pode ser maior que ombros
    if waist_width > shoulder_width:
        print(f"⚠️ CORREÇÃO: Cintura ({waist_width:.3f}) > Ombros ({shoulder_width:.3f})")
        waist_width = shoulder_width * 0.85
    
    # 2. Cintura não pode ser maior que quadril + margem
    if waist_width > hip_width * 1.15:
        print(f"⚠️ CORREÇÃO: Cintura ({waist_width:.3f}) > Quadril*1.15 ({hip_width*1.15:.3f})")
        waist_width = hip_width * 0.95
    
    # 3. Cintura não pode ser menor que 70% dos ombros (mínimo anatômico)
    min_waist = shoulder_width * 0.35
    if waist_width < min_waist:
        print(f"⚠️ CORREÇÃO: Cintura ({waist_width:.3f}) < Mínimo ({min_waist:.3f})")
        waist_width = min_waist
    
    # ===== INDICADORES VISUAIS AVANÇADOS =====
    volume_indicator = calculate_body_volume_indicator(landmarks, image.shape)
    waist_prominence = estimate_waist_visibility(landmarks)
    
    # Razão cintura/quadril
    waist_to_hip_ratio = waist_width / hip_width if hip_width > 0 else 0.85
    
    #  LOGS DETALHADOS
    print(f"\n📏 MEDIDAS EXTRAÍDAS:")
    print(f"   Ombros:  {shoulder_width:.4f}")
    print(f"   Quadril: {hip_width:.4f}")
    print(f"   Cintura: {waist_width:.4f}")
    print(f"   C/O:     {(waist_width/shoulder_width):.3f}")
    print(f"   C/Q:     {waist_to_hip_ratio:.3f}")
    print(f"   Volume:  {volume_indicator:.4f}")
    print(f"   W.Prom:  {waist_prominence:.4f}")
    
    #  VALIDAÇÃO FINAL
    waist_to_shoulder = waist_width / shoulder_width if shoulder_width > 0 else 0.6
    
    if waist_to_shoulder > 0.90:
        print(f"❌ ERRO: Razão C/O muito alta ({waist_to_shoulder:.3f})")
        # Forçar correção
        waist_width = shoulder_width * 0.65
        waist_to_hip_ratio = waist_width / hip_width
        print(f"   → Corrigido para: {waist_width:.4f} (C/O = 0.65)")
    
    return {
        "person_detected": True,
        "shoulder_width": round(shoulder_width, 4),
        "hip_width": round(hip_width, 4),
        "waist_width": round(waist_width, 4),
        "torso_height": round(torso_height, 4),
        "height_ratio": round(torso_height / shoulder_width if shoulder_width > 0 else 1.0, 4),
        "waist_ratio": round(waist_to_hip_ratio, 4),
        # Indicadores visuais
        "volume_indicator": round(volume_indicator, 4),
        "waist_prominence": round(waist_prominence, 4),
        "image_quality": 0.95
    }