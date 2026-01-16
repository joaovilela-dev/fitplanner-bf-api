def estimate_body_fat(
    body_type: str, 
    bmi: float, 
    sex: str, 
    ratios: dict = None,
    measurements: dict = None
) -> float:
    """
    Estima percentual de gordura corporal usando múltiplos indicadores.
    Combina: tipo corporal, IMC, ratios e indicadores visuais.
    """
    
    # ===== BASE INICIAL PELO TIPO CORPORAL =====
    if body_type == "lean":
        base = 10 if sex == "male" else 18
    elif body_type == "athletic":
        base = 12 if sex == "male" else 20
    elif body_type == "muscular":
        base = 15 if sex == "male" else 23
    elif body_type == "overweight":
        base = 25 if sex == "male" else 32
    else:
        base = 15 if sex == "male" else 25

    # ===== AJUSTE PROGRESSIVO PELO IMC =====
    if bmi < 18.5:
        bmi_adjustment = -3
    elif 18.5 <= bmi < 22:
        bmi_adjustment = -1
    elif 22 <= bmi < 25:
        bmi_adjustment = 1
    elif 25 <= bmi < 27.5:
        bmi_adjustment = 4
    elif 27.5 <= bmi < 30:
        bmi_adjustment = 7
    elif 30 <= bmi < 32.5:
        bmi_adjustment = 10
    else:  # BMI >= 32.5
        bmi_adjustment = 14

    # ===== AJUSTE PELA RAZÃO CINTURA/OMBRO =====
    waist_adjustment = 0
    if ratios:
        waist_to_shoulder = ratios.get("waist_to_shoulder", 0)
        
        if sex == "male":
            # Homens acumulam gordura na cintura
            if waist_to_shoulder > 0.70:
                waist_adjustment = 6
            elif waist_to_shoulder > 0.65:
                waist_adjustment = 4
            elif waist_to_shoulder > 0.58:
                waist_adjustment = 2
            elif waist_to_shoulder > 0.52:
                waist_adjustment = 1
        else:
            # Mulheres têm distribuição mais uniforme
            if waist_to_shoulder > 0.75:
                waist_adjustment = 5
            elif waist_to_shoulder > 0.70:
                waist_adjustment = 3
            elif waist_to_shoulder > 0.62:
                waist_adjustment = 2
            elif waist_to_shoulder > 0.55:
                waist_adjustment = 1

    # ===== AJUSTE POR INDICADORES VISUAIS AVANÇADOS =====
    visual_adjustment = 0
    if measurements:
        # Volume corporal (quanto maior, mais massa)
        volume_indicator = measurements.get("volume_indicator", 0)
        if volume_indicator > 0.25:
            visual_adjustment += 3
        elif volume_indicator > 0.20:
            visual_adjustment += 2
        elif volume_indicator > 0.15:
            visual_adjustment += 1
        
        # Proeminência da cintura (gordura abdominal)
        waist_prominence = measurements.get("waist_prominence", 0)
        if waist_prominence > 0.03:
            visual_adjustment += 4
        elif waist_prominence > 0.01:
            visual_adjustment += 2
        elif waist_prominence > 0:
            visual_adjustment += 1

    # ===== CÁLCULO FINAL =====
    import random
    noise = random.uniform(-0.5, 0.5)  # pequena aleatoriedade

    body_fat = base + bmi_adjustment + waist_adjustment + visual_adjustment + noise

    # ===== LIMITES REALISTAS =====
    if sex == "male":
        body_fat = max(5, min(body_fat, 45))
    else:
        body_fat = max(12, min(body_fat, 50))

    return round(body_fat, 1)