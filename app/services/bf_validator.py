"""
Validador inteligente de Body Fat para detectar e corrigir predições absurdas.
Usa lógica fisiológica para ajustar estimativas do modelo ML.
"""


def validate_and_adjust_bf(
    bf_prediction: float,
    bmi: float,
    sex: str,
    measurements: dict,
    ratios: dict
) -> dict:
    """
    Valida predição de BF e ajusta se necessário.
    
    Lógica:
    1. BMI baixo + volume alto = atlético (BF deve ser BAIXO)
    2. BMI alto + cintura larga = sobrepeso (BF deve ser ALTO)
    3. Proporções atléticas + BMI normal = BF moderado
    
    Returns:
        {
            "adjusted_bf": float,
            "original_bf": float,
            "was_adjusted": bool,
            "reason": str
        }
    """
    
    original_bf = bf_prediction
    adjusted_bf = bf_prediction
    was_adjusted = False
    reason = ""
    
    # ===================================
    # REGRA 1: BMI muito baixo + predição alta = IMPOSSÍVEL
    # ===================================
    if bmi < 22 and bf_prediction > 20:
        # Pessoa magra não pode ter BF alto
        if sex == "male":
            adjusted_bf = 8 + (bmi - 18) * 2  # 8-16%
        else:
            adjusted_bf = 16 + (bmi - 18) * 2  # 16-24%
        
        was_adjusted = True
        reason = f"BMI muito baixo ({bmi:.1f}) incompatível com BF alto"
    
    # ===================================
    # REGRA 2: BMI normal + ALTA DEFINIÇÃO = Atlético
    # ===================================
    waist_to_shoulder = ratios.get("waist_to_shoulder", 0.6)
    volume_indicator = measurements.get("volume_indicator", 0.15)
    
    # Detectar físico atlético:
    # - Cintura estreita (< 0.50)
    # - Volume alto (músculos) (> 0.20)
    # - BMI normal (20-26)
    is_athletic = (
        waist_to_shoulder < 0.50 and
        volume_indicator > 0.20 and
        20 <= bmi <= 26
    )
    
    if is_athletic and bf_prediction > 18:
        # Físico atlético não pode ter BF > 18%
        if sex == "male":
            adjusted_bf = 8 + (waist_to_shoulder * 20)  # 8-18%
        else:
            adjusted_bf = 16 + (waist_to_shoulder * 20)  # 16-26%
        
        was_adjusted = True
        reason = "Físico atlético detectado (cintura estreita + volume alto)"
    
    # ===================================
    # REGRA 3: BMI alto + predição baixa = IMPOSSÍVEL
    # ===================================
    if bmi > 28 and bf_prediction < 20:
        # Pessoa com sobrepeso não pode ter BF muito baixo
        if sex == "male":
            adjusted_bf = 20 + (bmi - 28) * 1.5  # 20-35%
        else:
            adjusted_bf = 28 + (bmi - 28) * 1.5  # 28-43%
        
        was_adjusted = True
        reason = f"BMI alto ({bmi:.1f}) incompatível com BF baixo"
    
    # ===================================
    # REGRA 4: Cintura muito larga = Alto BF garantido
    # ===================================
    if waist_to_shoulder > 0.70 and bf_prediction < 22:
        # Cintura larga sempre indica gordura abdominal
        if sex == "male":
            adjusted_bf = 22 + (waist_to_shoulder - 0.70) * 40
        else:
            adjusted_bf = 30 + (waist_to_shoulder - 0.70) * 40
        
        was_adjusted = True
        reason = "Cintura muito larga indica alto BF"
    
    # ===================================
    # LIMITES FISIOLÓGICOS FINAIS
    # ===================================
    if sex == "male":
        adjusted_bf = max(5, min(adjusted_bf, 45))
    else:
        adjusted_bf = max(12, min(adjusted_bf, 50))
    
    return {
        "adjusted_bf": round(adjusted_bf, 1),
        "original_bf": round(original_bf, 1),
        "was_adjusted": was_adjusted,
        "adjustment": round(adjusted_bf - original_bf, 1),
        "reason": reason,
        "confidence": "high" if not was_adjusted else "adjusted"
    }


def get_bf_category(bf: float, sex: str) -> dict:
    """
    Categoriza o percentual de gordura corporal.
    
    Returns:
        {
            "category": str,
            "label": str,
            "description": str
        }
    """
    
    if sex == "male":
        if bf < 6:
            return {
                "category": "essential",
                "label": "Essencial (Extremo)",
                "description": "Nível mínimo para sobrevivência. Risco à saúde."
            }
        elif bf < 14:
            return {
                "category": "athletic",
                "label": "Atlético",
                "description": "Músculos bem definidos, vascularização visível."
            }
        elif bf < 18:
            return {
                "category": "fit",
                "label": "Fitness",
                "description": "Aparência saudável e atlética."
            }
        elif bf < 25:
            return {
                "category": "average",
                "label": "Médio",
                "description": "Faixa comum para homens adultos."
            }
        else:
            return {
                "category": "overweight",
                "label": "Sobrepeso",
                "description": "Excesso de gordura corporal visível."
            }
    else:  # female
        if bf < 14:
            return {
                "category": "essential",
                "label": "Essencial (Extremo)",
                "description": "Nível mínimo para sobrevivência. Risco à saúde."
            }
        elif bf < 21:
            return {
                "category": "athletic",
                "label": "Atlético",
                "description": "Músculos definidos, baixo percentual de gordura."
            }
        elif bf < 25:
            return {
                "category": "fit",
                "label": "Fitness",
                "description": "Aparência saudável e tonificada."
            }
        elif bf < 32:
            return {
                "category": "average",
                "label": "Médio",
                "description": "Faixa comum para mulheres adultas."
            }
        else:
            return {
                "category": "overweight",
                "label": "Sobrepeso",
                "description": "Excesso de gordura corporal visível."
            }


def get_detailed_analysis(
    bf: float,
    bmi: float,
    sex: str,
    age: int,
    ratios: dict
) -> dict:
    """
    Análise detalhada da composição corporal.
    """
    
    category = get_bf_category(bf, sex)
    
    # Estimativa de massa magra
    # (simplificado, ideal seria ter peso real)
    if sex == "male":
        avg_weight = bmi * 1.75 * 1.75  # Assume 1.75m de altura média
        fat_mass = avg_weight * (bf / 100)
        lean_mass = avg_weight - fat_mass
    else:
        avg_weight = bmi * 1.65 * 1.65  # Assume 1.65m de altura média
        fat_mass = avg_weight * (bf / 100)
        lean_mass = avg_weight - fat_mass
    
    # Análise de proporções
    waist_to_shoulder = ratios.get("waist_to_shoulder", 0.6)
    
    if waist_to_shoulder < 0.45:
        body_shape = "Triângulo invertido (V-shape)"
        shape_note = "Ombros largos, cintura estreita - forma atlética"
    elif waist_to_shoulder < 0.55:
        body_shape = "Retangular"
        shape_note = "Proporções equilibradas"
    elif waist_to_shoulder < 0.65:
        body_shape = "Oval"
        shape_note = "Cintura um pouco mais larga"
    else:
        body_shape = "Circular/Maçã"
        shape_note = "Acúmulo de gordura na região abdominal"
    
    return {
        "bf_percentage": bf,
        "category": category,
        "body_shape": body_shape,
        "shape_description": shape_note,
        "estimated_fat_mass_kg": round(fat_mass, 1),
        "estimated_lean_mass_kg": round(lean_mass, 1),
        "health_status": _get_health_status(bf, bmi, sex, age),
        "recommendations": _get_recommendations(bf, bmi, sex, category["category"])
    }


def _get_health_status(bf: float, bmi: float, sex: str, age: int) -> str:
    """Avalia status de saúde baseado em BF e BMI"""
    
    if sex == "male":
        healthy_bf_range = (10, 22)
    else:
        healthy_bf_range = (18, 30)
    
    healthy_bmi_range = (18.5, 24.9)
    
    bf_healthy = healthy_bf_range[0] <= bf <= healthy_bf_range[1]
    bmi_healthy = healthy_bmi_range[0] <= bmi <= healthy_bmi_range[1]
    
    if bf_healthy and bmi_healthy:
        return "✅ Saudável"
    elif bf < healthy_bf_range[0]:
        return "⚠️ Gordura corporal muito baixa"
    elif bf > healthy_bf_range[1]:
        return "⚠️ Gordura corporal elevada"
    elif bmi < healthy_bmi_range[0]:
        return "⚠️ Peso abaixo do ideal"
    elif bmi > healthy_bmi_range[1]:
        return "⚠️ Sobrepeso"
    else:
        return "ℹ️ Avaliação individual recomendada"


def _get_recommendations(bf: float, bmi: float, sex: str, category: str) -> list:
    """Gera recomendações personalizadas"""
    
    recs = []
    
    if category == "essential":
        recs.append("⚠️ Consulte um médico - BF muito baixo pode ser prejudicial")
        recs.append("Considere aumentar ingestão calórica gradualmente")
    
    elif category == "athletic":
        recs.append("✅ Mantenha treino de força e alimentação balanceada")
        recs.append("Monitore hidratação e recuperação muscular")
    
    elif category == "fit":
        recs.append("✅ Continue com hábitos saudáveis atuais")
        recs.append("Varie exercícios para manter progresso")
    
    elif category == "average":
        if bf > (18 if sex == "male" else 25):
            recs.append("Considere aumentar atividade física gradualmente")
            recs.append("Ajuste alimentação para déficit calórico leve")
    
    elif category == "overweight":
        recs.append("Recomendado: exercícios aeróbicos 3-5x/semana")
        recs.append("Considere acompanhamento nutricional profissional")
        recs.append("Foque em déficit calórico sustentável (300-500 kcal/dia)")
    
    return recs