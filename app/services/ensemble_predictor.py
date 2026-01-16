"""
Sistema ENSEMBLE de predição de Body Fat - V3 SAFE MODE
Arquitetura profissional com ML em quarentena.

🔒 MODO SAFE (Produção):
    - Regras (60%) + Textura (40%)
    - ML não influencia o resultado final
    - ML registrado apenas para análise

🧪 MODO EXPERIMENTAL (Pesquisa):
    - ML incluído no ensemble (apenas para comparação)
    - Disponível via flag use_experimental_ml=True
"""

import numpy as np
import os


# ===================================
# CONFIGURAÇÃO GLOBAL
# ===================================
USE_EXPERIMENTAL_ML = os.getenv("USE_EXPERIMENTAL_ML", "false").lower() == "true"


def ensemble_predict_body_fat(
    ml_prediction: float,
    rules_prediction: float,
    texture_data: dict,
    bmi: float,
    sex: str,
    measurements: dict,
    ratios: dict,
    use_experimental_ml: bool = USE_EXPERIMENTAL_ML
) -> dict:
    """
    Predição ENSEMBLE V3 - SAFE MODE
    
    🔒 MODO SAFE (padrão):
        - BF Final = Regras (60%) + Textura (40%)
        - ML registrado mas não influencia
        - Maior confiabilidade
    
    🧪 MODO EXPERIMENTAL:
        - BF Final inclui ML no ensemble
        - Apenas para análise e comparação
    
    Args:
        ml_prediction: Predição do modelo ML
        rules_prediction: Predição baseada em regras
        texture_data: Dados de análise de textura
        bmi: Índice de Massa Corporal
        sex: "male" ou "female"
        measurements: Medições corporais
        ratios: Razões corporais
        use_experimental_ml: Se True, inclui ML no cálculo final
    
    Returns:
        {
            "final_prediction": float,
            "safe_prediction": float (sempre sem ML),
            "experimental_prediction": float (com ML),
            "mode": "SAFE" ou "EXPERIMENTAL",
            ...
        }
    """
    
    adjustments = []
    
    # ===================================
    #  PREDIÇÃO BASEADA EM TEXTURA
    # ===================================
    from app.services.texture_analyzer import estimate_bf_from_definition
    texture_prediction = estimate_bf_from_definition(texture_data, sex, bmi)
    
    # ===================================
    #  ANÁLISE DE QUALIDADE DAS MEDIÇÕES
    # ===================================
    
    shoulder_width = measurements.get("shoulder_width", 0.4)
    hip_width = measurements.get("hip_width", 0.3)
    waist_to_shoulder = ratios.get("waist_to_shoulder", 0.6)
    volume_indicator = measurements.get("volume_indicator", 0.15)
    
    # Detectar medições suspeitas (fundo branco, detecção parcial)
    measurements_suspicious = (
        shoulder_width < 0.25 or
        hip_width < 0.15 or
        waist_to_shoulder > 0.80
    )
    
    if measurements_suspicious:
        adjustments.append(
            f"⚠️ Medidas antropométricas suspeitas (shoulder={shoulder_width:.3f}, hip={hip_width:.3f})"
        )
        print(f"   ⚠️ MEDIDAS SUSPEITAS (shoulder={shoulder_width:.3f}, hip={hip_width:.3f})")
    
    # ===================================
    #  CALCULAR CONFIANÇA DE CADA MÉTODO
    # ===================================
    
    # Textura: Confiança da análise de imagem
    texture_confidence = texture_data.get("confidence", 0.5)
    
    # Regras: Confiança baseada em consistência dos dados
    rules_confidence = 0.7
    
    # ML: Sempre registrado, mas confiança depende do modo
    ml_confidence = _calculate_ml_confidence(measurements, ratios, bmi)
    
    # ===================================
    #  AJUSTES DINÂMICOS DE CONFIANÇA
    # ===================================
    
    # Extrair métricas relevantes
    definition_score = texture_data.get("definition_score", 0.5)
    abs_visibility = texture_data.get("abs_visibility", 0.5)
    central_fat = texture_data.get("central_fat", 0.5)
    
    # --- CASO 1: ATLETA ---
    is_athlete = (
        definition_score > 0.65 and
        abs_visibility > 0.60 and
        central_fat < 0.45 and
        20 <= bmi <= 26
    )
    
    if is_athlete:
        adjustments.append("🏋️ Físico atlético - Priorizando análise visual")
        print("   🏋️ ATLETA DETECTADO")
        texture_confidence = min(texture_confidence * 1.4, 1.0)
        rules_confidence *= 0.9
    
    # --- CASO 2: SOBREPESO ---
    is_overweight = (
        bmi > 28 and
        waist_to_shoulder > 0.65
    )
    
    if is_overweight:
        adjustments.append("📈 Sobrepeso - Priorizando regras fisiológicas")
        print("   📈 SOBREPESO DETECTADO")
        rules_confidence = min(rules_confidence * 1.3, 1.0)
        texture_confidence *= 0.85
    
    # --- CASO 3: MEDIDAS SUSPEITAS ---
    if measurements_suspicious:
        adjustments.append("📸 Detecção parcial - Priorizando análise visual")
        print("   📸 PRIORIZANDO TEXTURA")
        texture_confidence = min(texture_confidence * 1.6, 1.0)
        rules_confidence *= 0.7
    
    # --- CASO 4: BAIXA QUALIDADE DE IMAGEM ---
    if texture_confidence < 0.4:
        adjustments.append("⚠️ Baixa qualidade de imagem - Priorizando regras")
        print("   ⚠️ BAIXA QUALIDADE")
        rules_confidence = min(rules_confidence * 1.2, 1.0)
        texture_confidence *= 0.8
    
    # ===================================
    #  MODO SAFE 
    # ===================================
    
    # Normalizar confianças (SEM ML)
    total_safe = rules_confidence + texture_confidence
    safe_rules_weight = rules_confidence / total_safe
    safe_texture_weight = texture_confidence / total_safe
    
    # Calcular BF SAFE (apenas Regras + Textura)
    safe_prediction = (
        rules_prediction * safe_rules_weight +
        texture_prediction * safe_texture_weight
    )
    
    print(f"\n🔒 MODO SAFE - Predições:")
    print(f"   Regras:  {rules_prediction:.1f}% (peso {safe_rules_weight:.2f})")
    print(f"   Textura: {texture_prediction:.1f}% (peso {safe_texture_weight:.2f})")
    print(f"   → SAFE:  {safe_prediction:.1f}%")
    
    # ===================================
    #  MODO EXPERIMENTAL 
    # ===================================
    
    experimental_prediction = None
    experimental_weights = None
    ml_divergence = abs(ml_prediction - rules_prediction)
    
    if use_experimental_ml:
        # Ajustar confiança do ML baseado em divergência
        if ml_divergence > 10:
            adjustments.append(f"⚠️ ML diverge {ml_divergence:.1f}% das regras")
            ml_confidence *= 0.3  # Reduzir drasticamente
        
        # Normalizar confianças (COM ML)
        total_exp = ml_confidence + rules_confidence + texture_confidence
        exp_ml_weight = ml_confidence / total_exp
        exp_rules_weight = rules_confidence / total_exp
        exp_texture_weight = texture_confidence / total_exp
        
        experimental_prediction = (
            ml_prediction * exp_ml_weight +
            rules_prediction * exp_rules_weight +
            texture_prediction * exp_texture_weight
        )
        
        experimental_weights = {
            "ml": round(exp_ml_weight, 3),
            "rules": round(exp_rules_weight, 3),
            "texture": round(exp_texture_weight, 3)
        }
        
        print(f"\n🧪 MODO EXPERIMENTAL - Predições:")
        print(f"   ML:      {ml_prediction:.1f}% (peso {exp_ml_weight:.2f})")
        print(f"   Regras:  {rules_prediction:.1f}% (peso {exp_rules_weight:.2f})")
        print(f"   Textura: {texture_prediction:.1f}% (peso {exp_texture_weight:.2f})")
        print(f"   → EXPERIMENTAL: {experimental_prediction:.1f}%")
    
    # ===================================
    #  VALIDAÇÃO FISIOLÓGICA FINAL
    # ===================================
    from app.services.bf_validator import validate_and_adjust_bf
    
    # Validar predição SAFE
    validation = validate_and_adjust_bf(
        bf_prediction=safe_prediction,
        bmi=bmi,
        sex=sex,
        measurements=measurements,
        ratios=ratios
    )
    
    final_prediction = validation["adjusted_bf"]
    
    if validation["was_adjusted"]:
        adjustments.append(
            f"🔧 Validação: {safe_prediction:.1f}% → {final_prediction:.1f}% "
            f"({validation['reason']})"
        )
        print(f"   🔧 VALIDAÇÃO: {safe_prediction:.1f}% → {final_prediction:.1f}%")
    
    print(f"   ✅ FINAL: {final_prediction:.1f}%")
    
    # ===================================
    #  CONFIANÇA FINAL
    # ===================================
    
    # Calcular divergência entre métodos principais
    predictions_safe = [rules_prediction, texture_prediction]
    pred_std_safe = np.std(predictions_safe)
    
    # Confiança baseada em concordância
    agreement = 1.0 - (pred_std_safe / 20)
    avg_confidence = (rules_confidence + texture_confidence) / 2
    
    final_confidence = (agreement * 0.6 + avg_confidence * 0.4)
    final_confidence = max(0.3, min(final_confidence, 1.0))
    
    #  ML como DETECTOR DE CONFIANÇA
    if ml_divergence > 15:
        final_confidence *= 0.85  # Reduzir confiança se ML diverge muito
        adjustments.append(f" Confiança reduzida devido à divergência do ML ({ml_divergence:.1f}%)")
    
    # ===================================
    #  RESULTADO COM MODO SAFE
    # ===================================
    
    mode = "EXPERIMENTAL" if use_experimental_ml else "SAFE"
    
    result = {
        # Predições finais
        "final_prediction": round(experimental_prediction if use_experimental_ml else final_prediction, 1),
        "safe_prediction": round(final_prediction, 1),  # Sempre disponível
        "experimental_prediction": round(experimental_prediction, 1) if experimental_prediction else None,
        
        # Predições individuais
        "ml_prediction": round(ml_prediction, 1),
        "rules_prediction": round(rules_prediction, 1),
        "texture_prediction": round(texture_prediction, 1),
        
        # Pesos SAFE
        "safe_weights": {
            "rules": round(safe_rules_weight, 3),
            "texture": round(safe_texture_weight, 3)
        },
        
        # Pesos EXPERIMENTAL (se aplicável)
        "experimental_weights": experimental_weights,
        
        # Metadados
        "mode": mode,
        "confidence": round(final_confidence, 2),
        "confidence_level": _get_confidence_level(final_confidence),
        "method_used": f"Ensemble V3 - {mode} MODE (Rules + Texture)",
        "adjustments": adjustments,
        "texture_analysis": texture_data,
        
        # ML como detector de confiança
        "ml_analysis": {
            "prediction": round(ml_prediction, 1),
            "divergence_from_rules": round(ml_divergence, 1),
            "confidence_impact": "reduced" if ml_divergence > 15 else "neutral",
            "status": "quarantine" if not use_experimental_ml else "active"
        },
        
        # Casos especiais
        "special_cases": {
            "is_athlete": is_athlete,
            "is_overweight": is_overweight,
            "measurements_suspicious": measurements_suspicious,
            "low_image_quality": texture_confidence < 0.4
        }
    }
    
    return result


def _calculate_ml_confidence(measurements: dict, ratios: dict, bmi: float) -> float:
    """
    Calcula confiança do modelo ML.
     V3: Confiança reduzida por padrão (ML em quarentena)
    """
    
    confidence = 0.5  #  Reduzido de 1.0 para 0.5
    
    # BMI típico: 18-35
    if bmi < 18 or bmi > 35:
        confidence *= 0.8
    
    # Waist ratio típico: 0.40-0.75
    waist_ratio = ratios.get("waist_to_shoulder", 0.6)
    if waist_ratio < 0.35 or waist_ratio > 0.80:
        confidence *= 0.85
    
    # Volume indicator típico: 0.12-0.30
    volume = measurements.get("volume_indicator", 0.15)
    if volume < 0.10 or volume > 0.35:
        confidence *= 0.9
    
    return confidence


def _get_confidence_level(confidence: float) -> str:
    """Retorna nível textual de confiança"""
    if confidence >= 0.85:
        return "Muito Alta"
    elif confidence >= 0.70:
        return "Alta"
    elif confidence >= 0.55:
        return "Média"
    elif confidence >= 0.40:
        return "Baixa"
    else:
        return "Muito Baixa"