"""
Análise de textura da imagem para detectar definição muscular - V3 ULTRA CALIBRADO
 V3: Thresholds MUITO mais rigorosos para melhor separação entre físicos
"""

import cv2
import numpy as np
from typing import Optional


def analyze_muscle_definition(image_path: str, landmarks=None) -> dict:
    """
    Analisa definição muscular na imagem - V3 ULTRA CALIBRADO
    
     V3 Melhorias:
    - Thresholds ULTRA rigorosos
    - Penalização forte por gordura central
    - Melhor separação (atleta: 10-12%, comum: 22-25%)
    
    Returns:
        {
            "definition_score": float (0-1),
            "abs_visibility": float (0-1),
            "vascularity": float (0-1),
            "muscle_separation": float (0-1),
            "subcutaneous_fat": float (0-1),
            "central_fat": float (0-1),
            "confidence": float (0-1)
        }
    """
    
    try:
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            return _get_default_result()
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ===================================
        #  ANÁLISE DE TEXTURA GLOBAL
        # ===================================
        texture_score = _analyze_texture_complexity(gray)
        
        # ===================================
        #  DETECÇÃO DE BORDAS
        # ===================================
        edge_score = _detect_muscle_edges(gray)
        
        # ===================================
        #  ANÁLISE DE CONTRASTE
        # ===================================
        contrast_score = _analyze_local_contrast(gray)
        
        # ===================================
        #  ANÁLISE ABDOMINAL ULTRA RIGOROSA
        # ===================================
        abs_score = _analyze_abdominal_region_v3(image, gray, landmarks)
        
        # ===================================
        #  VASCULARIZAÇÃO ULTRA RIGOROSA
        # ===================================
        vascularity_score = _detect_vascularity_v3(gray)
        
        # ===================================
        #  GORDURA CENTRAL
        # ===================================
        central_fat_score = _detect_central_fat(gray)
        
        # ===================================
        #  SUAVIDADE DA PELE
        # ===================================
        smoothness = _analyze_skin_smoothness(gray)
        fat_score = 1.0 - smoothness
        
        # ===================================
        #  SCORE FINAL V3
        # ===================================
        definition_score = (
            texture_score * 0.12 +
            edge_score * 0.18 +
            contrast_score * 0.12 +
            abs_score * 0.35 +              # Aumentado
            vascularity_score * 0.18 +       # Aumentado
            (1.0 - central_fat_score) * 0.05
        )
        
        #  PENALIZAÇÃO FORTE por gordura central
        if central_fat_score > 0.65:
            definition_score *= 0.75  # -25%
        elif central_fat_score > 0.55:
            definition_score *= 0.85  # -15%
        
        # Confiança
        confidence = _calculate_confidence(image, gray)
        
        result = {
            "definition_score": round(float(definition_score), 3),
            "abs_visibility": round(float(abs_score), 3),
            "vascularity": round(float(vascularity_score), 3),
            "muscle_separation": round(float(edge_score), 3),
            "subcutaneous_fat": round(float(fat_score), 3),
            "central_fat": round(float(central_fat_score), 3),
            "texture_complexity": round(float(texture_score), 3),
            "local_contrast": round(float(contrast_score), 3),
            "confidence": round(float(confidence), 3)
        }
        
        #  LOGS COMPLETOS
        print(f"   Definition Score: {result['definition_score']:.3f}")
        print(f"   Abs Visibility:   {result['abs_visibility']:.3f}")
        print(f"   Vascularity:      {result['vascularity']:.3f}")
        print(f"   Central Fat:      {result['central_fat']:.3f}")  
        print(f"   Confidence:       {result['confidence']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"⚠️ Erro na análise de textura: {e}")
        return _get_default_result()


def _analyze_texture_complexity(gray: np.ndarray) -> float:
    """Analisa complexidade da textura"""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    normalized = min(variance / 500, 1.0)
    return normalized


def _detect_muscle_edges(gray: np.ndarray) -> float:
    """Detecta bordas"""
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    normalized = min(edge_density / 0.08, 1.0)
    return normalized


def _analyze_local_contrast(gray: np.ndarray) -> float:
    """Analisa contraste local"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    local_std = np.abs(gray.astype(float) - blurred.astype(float))
    contrast = local_std.mean()
    normalized = min(contrast / 20, 1.0)
    return normalized


def _analyze_abdominal_region_v3(image: np.ndarray, gray: np.ndarray, landmarks) -> float:
    """
     V3: Análise ULTRA RIGOROSA do abdômen
    
    Para ter abs_score > 0.80:
    - horizontal > 40 (era 35)
    - vertical > 35 (era 30)
    - variance > 2000 (era 1800)
    - std > 45 (era 40)
    """
    h, w = gray.shape
    
    # ROI: região abdominal
    y_start = int(h * 0.40)
    y_end = int(h * 0.70)
    x_start = int(w * 0.35)
    x_end = int(w * 0.65)
    
    roi = gray[y_start:y_end, x_start:x_end]
    
    if roi.size == 0:
        return 0.5
    
    # Análise de bordas
    edges_h = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    horizontal_features = np.abs(edges_h).mean()
    
    edges_v = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    vertical_features = np.abs(edges_v).mean()
    
    # Variância e desvio padrão
    roi_variance = roi.var()
    roi_std = roi.std()
    
    #  THRESHOLDS ULTRA RIGOROSOS
    abs_score = (
        min(horizontal_features / 40, 1.0) * 0.35 +  #
        min(vertical_features / 35, 1.0) * 0.25 +    
        min(roi_variance / 2000, 1.0) * 0.25 +       
        min(roi_std / 45, 1.0) * 0.15                
    )
    
    #  PENALIZAÇÃO MUITO FORTE
    if horizontal_features < 18:  # Era 15
        abs_score *= 0.3  # Era 0.4
    
    if vertical_features < 12:  # Era 10
        abs_score *= 0.5  # Era 0.6
    
    return abs_score


def _detect_vascularity_v3(gray: np.ndarray) -> float:
    """
     V3: Vascularização ULTRA RIGOROSA
    
    Para ter vasc > 0.80: vein_density > 0.06 (era 0.05)
    """
    # Equalização
    equalized = cv2.equalizeHist(gray)
    
    # Detecção de veias
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    tophat = cv2.morphologyEx(equalized, cv2.MORPH_TOPHAT, kernel)
    
    _, veins = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remover ruído
    kernel_clean = np.ones((2, 2), np.uint8)
    veins = cv2.morphologyEx(veins, cv2.MORPH_OPEN, kernel_clean)
    
    vein_density = np.count_nonzero(veins) / veins.size
    
    #  THRESHOLD ULTRA RIGOROSO
    normalized = min(vein_density / 0.06, 1.0)  #  0.06 (era 0.05)
    
    #  PENALIZAÇÃO FORTE
    if vein_density < 0.018:  # Era 0.015
        normalized *= 0.4  # Era 0.5
    
    return normalized


def _detect_central_fat(gray: np.ndarray) -> float:
    """
     V3: Detecta gordura abdominal central
    
    Returns:
        float: 0 = sem gordura, 1 = muita gordura
    """
    h, w = gray.shape
    
    # ROI: centro do abdômen
    y_start = int(h * 0.45)
    y_end = int(h * 0.65)
    x_start = int(w * 0.40)
    x_end = int(w * 0.60)
    
    roi = gray[y_start:y_end, x_start:x_end]
    
    if roi.size == 0:
        return 0.5
    
    # 1. Suavidade (gordura é lisa)
    blurred = cv2.GaussianBlur(roi, (9, 9), 0)
    difference = np.abs(roi.astype(float) - blurred.astype(float))
    smoothness = 1.0 - min(difference.mean() / 15, 1.0)
    
    # 2. Baixo contraste
    roi_std = roi.std()
    low_contrast = 1.0 - min(roi_std / 35, 1.0)
    
    # 3. Uniformidade
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    uniformity = 1.0 - min(entropy / 7.0, 1.0)
    
    # Score de gordura central
    fat_score = (
        smoothness * 0.40 +
        low_contrast * 0.35 +
        uniformity * 0.25
    )
    
    return fat_score


def _analyze_skin_smoothness(gray: np.ndarray) -> float:
    """Mede suavidade da pele"""
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    difference = np.abs(gray.astype(float) - blurred.astype(float))
    smoothness_score = 1.0 - min(difference.mean() / 10, 1.0)
    return smoothness_score


def _calculate_confidence(image: np.ndarray, gray: np.ndarray) -> float:
    """Calcula confiança"""
    h, w = gray.shape
    
    resolution_score = min((h * w) / (640 * 480), 1.0)
    
    mean_brightness = gray.mean()
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128
    
    contrast = gray.std()
    contrast_score = min(contrast / 50, 1.0)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    focus_score = min(laplacian_var / 300, 1.0)
    
    confidence = (
        resolution_score * 0.25 +
        brightness_score * 0.25 +
        contrast_score * 0.25 +
        focus_score * 0.25
    )
    
    return confidence


def _get_default_result() -> dict:
    """Resultado padrão"""
    return {
        "definition_score": 0.5,
        "abs_visibility": 0.5,
        "vascularity": 0.5,
        "muscle_separation": 0.5,
        "subcutaneous_fat": 0.5,
        "central_fat": 0.5,
        "texture_complexity": 0.5,
        "local_contrast": 0.5,
        "confidence": 0.3
    }


def estimate_bf_from_definition(definition_data: dict, sex: str, bmi: float) -> float:
    """
     V3: Estimativa ULTRA CALIBRADA
    
    Thresholds muito mais rigorosos para melhor separação:
    - Atleta: visual_score > 0.85 → BF 7-11%
    - Comum: visual_score 0.40-0.60 → BF 20-28%
    """
    
    definition_score = definition_data["definition_score"]
    abs_visibility = definition_data["abs_visibility"]
    vascularity = definition_data["vascularity"]
    central_fat = definition_data.get("central_fat", 0.5)
    
    # Score combinado
    visual_score = (
        definition_score * 0.40 +
        abs_visibility * 0.35 +
        vascularity * 0.25
    )
    
    #  PENALIZAÇÃO FORTE por gordura central
    if central_fat > 0.70:
        visual_score *= 0.70  # -30%
    elif central_fat > 0.60:
        visual_score *= 0.80  # -20%
    elif central_fat > 0.50:
        visual_score *= 0.90  # -10%
    
    #  Mapping ULTRA RIGOROSO
    if sex == "male":
        if visual_score > 0.85:  # 🆕 0.85 (era 0.80)
            bf_estimate = 7 + (1.0 - visual_score) * 27  # 7-11%
        elif visual_score > 0.70:  # 🆕 0.70 (era 0.65)
            bf_estimate = 11 + (0.85 - visual_score) * 33  # 11-16%
        elif visual_score > 0.50:  # 🆕 0.50 (era 0.45)
            bf_estimate = 16 + (0.70 - visual_score) * 40  # 16-24%
        else:
            bf_estimate = 24 + (0.50 - visual_score) * 50  # 24-49%
    else:
        if visual_score > 0.80:
            bf_estimate = 14 + (1.0 - visual_score) * 30
        elif visual_score > 0.60:
            bf_estimate = 20 + (0.80 - visual_score) * 30
        elif visual_score > 0.40:
            bf_estimate = 26 + (0.60 - visual_score) * 40
        else:
            bf_estimate = 34 + (0.40 - visual_score) * 50
    
    # Ajuste por BMI
    if bmi < 20 and bf_estimate > 16:
        bf_estimate = 9 + (bmi - 18) * 3
    elif bmi > 30 and bf_estimate < 20:
        bf_estimate = 20 + (bmi - 30) * 2.5
    
    #  PENALIZAÇÃO ADICIONAL por gordura central
    if central_fat > 0.75:
        bf_estimate = min(bf_estimate + 5, 45 if sex == "male" else 50)
    elif central_fat > 0.65:
        bf_estimate = min(bf_estimate + 3, 45 if sex == "male" else 50)
    
    return round(bf_estimate, 1)