def classify_body_type(ratios: dict, sex: str, bmi: float = None) -> str:
    """
    Classifica o tipo corporal combinando ratios corporais e BMI.
    Retorna: 'lean', 'muscular', 'overweight'
    """

    waist_to_shoulder = ratios.get("waist_to_shoulder", 0)
    hip_to_shoulder = ratios.get("hip_to_shoulder", 0)
    torso_to_shoulder = ratios.get("torso_to_shoulder", 0)

    # Thresholds diferentes para homens e mulheres
    if sex == "female":
        if waist_to_shoulder < 0.45 and hip_to_shoulder < 0.9:
            body_type = "muscular"
        elif waist_to_shoulder > 0.6 or (bmi and bmi >= 28):
            body_type = "overweight"
        else:
            body_type = "lean"
    else:  # male
        if waist_to_shoulder < 0.4 and hip_to_shoulder < 0.85:
            body_type = "muscular"
        elif waist_to_shoulder > 0.6 or (bmi and bmi >= 28):
            body_type = "overweight"
        else:
            body_type = "lean"

    return body_type