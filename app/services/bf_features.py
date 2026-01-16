"""
Construtor de features para o modelo de Body Fat ML.
Transforma medições brutas + dados do usuário em vetor de features.
"""


def build_features(
    measurements: dict,
    age: int,
    weight_kg: float,
    height_cm: float,
    sex: str
) -> dict:
    """
    Constrói vetor de features para o modelo ML de Body Fat.
    
    IMPORTANTE: A ordem das features DEVE corresponder ao treinamento!
    
    Features esperadas (9 no total):
    1. age: Idade em anos
    2. bmi: Índice de Massa Corporal
    3. sex: 0=female, 1=male
    4. shoulder_width: Largura dos ombros (normalizada)
    5. hip_width: Largura do quadril (normalizada)
    6. height_ratio: Proporção altura/largura do torso
    7. waist_ratio: Razão cintura/ombro
    8. volume_indicator: Indicador de volume corporal
    9. waist_prominence: Proeminência abdominal
    
    Args:
        measurements: Dict com medições do MediaPipe
        age: Idade do usuário
        weight_kg: Peso em kg
        height_cm: Altura em cm
        sex: "male" ou "female"
    
    Returns:
        Dict com features prontas para o modelo ML
    """
    
    # ===================================
    #  CALCULAR BMI
    # ===================================
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # ===================================
    #  CODIFICAR SEXO
    # ===================================
    sex_normalized = sex.lower().strip()
    
    if sex_normalized in ['m', 'male', 'masculino', 'homem', 'man']:
        sex_encoded = 1
    elif sex_normalized in ['f', 'female', 'feminino', 'mulher', 'woman']:
        sex_encoded = 0
    else:
        # Fallback: assumir masculino se indefinido
        print(f"⚠️ Sexo '{sex}' não reconhecido. Usando 'male' como padrão.")
        sex_encoded = 1
    
    # ===================================
    #  EXTRAIR MEDIÇÕES
    # ===================================
    shoulder_width = measurements.get("shoulder_width", 0.45)
    hip_width = measurements.get("hip_width", 0.38)
    height_ratio = measurements.get("height_ratio", 0.90)
    waist_ratio = measurements.get("waist_ratio", 0.60)
    volume_indicator = measurements.get("volume_indicator", 0.15)
    waist_prominence = measurements.get("waist_prominence", 0.01)
    
    # ===================================
    #  VALIDAÇÕES DE SANIDADE
    # ===================================
    # Garantir que valores estão em ranges razoáveis
    
    # Age: 16-100 anos
    if not (16 <= age <= 100):
        print(f"⚠️ Idade fora do range: {age} (esperado: 16-100)")
        age = max(16, min(age, 100))
    
    # BMI: 12-50
    if not (12 <= bmi <= 50):
        print(f"⚠️ BMI fora do range: {bmi:.1f} (esperado: 12-50)")
        bmi = max(12, min(bmi, 50))
    
    # Shoulder width: 0.30-0.55
    if not (0.30 <= shoulder_width <= 0.55):
        print(f"⚠️ Shoulder width fora do range: {shoulder_width:.3f}")
        shoulder_width = max(0.30, min(shoulder_width, 0.55))
    
    # Hip width: 0.25-0.50
    if not (0.25 <= hip_width <= 0.50):
        print(f"⚠️ Hip width fora do range: {hip_width:.3f}")
        hip_width = max(0.25, min(hip_width, 0.50))
    
    # Waist ratio: 0.35-0.85
    if not (0.35 <= waist_ratio <= 0.85):
        print(f"⚠️ Waist ratio fora do range: {waist_ratio:.3f}")
        waist_ratio = max(0.35, min(waist_ratio, 0.85))
    
    # Volume indicator: 0.08-0.35
    if not (0.08 <= volume_indicator <= 0.35):
        print(f"⚠️ Volume indicator fora do range: {volume_indicator:.3f}")
        volume_indicator = max(0.08, min(volume_indicator, 0.35))
    
    # Waist prominence: -0.02 a 0.10
    if not (-0.02 <= waist_prominence <= 0.10):
        print(f"⚠️ Waist prominence fora do range: {waist_prominence:.3f}")
        waist_prominence = max(-0.02, min(waist_prominence, 0.10))
    
    # ===================================
    #  MONTAR VETOR DE FEATURES
    # ===================================
    features = {
        "age": age,
        "bmi": round(bmi, 2),
        "sex": sex_encoded,
        "shoulder_width": round(shoulder_width, 4),
        "hip_width": round(hip_width, 4),
        "height_ratio": round(height_ratio, 4),
        "waist_ratio": round(waist_ratio, 4),
        "volume_indicator": round(volume_indicator, 4),
        "waist_prominence": round(waist_prominence, 4)
    }
    

    print(f"\n📊 Features construídas:")
    print(f"   Age: {features['age']}")
    print(f"   BMI: {features['bmi']:.1f}")
    print(f"   Sex: {'Male' if features['sex'] == 1 else 'Female'}")
    print(f"   Shoulder: {features['shoulder_width']:.3f}")
    print(f"   Hip: {features['hip_width']:.3f}")
    print(f"   Waist Ratio: {features['waist_ratio']:.3f}")
    print(f"   Volume: {features['volume_indicator']:.3f}")
    print(f"   Prominence: {features['waist_prominence']:.3f}")
    
    return features


def validate_features(features: dict) -> bool:
    """
    Valida se todas as features necessárias estão presentes.
    
    Returns:
        True se válido, False caso contrário
    """
    required_features = [
        "age", "bmi", "sex",
        "shoulder_width", "hip_width", "height_ratio", "waist_ratio",
        "volume_indicator", "waist_prominence"
    ]
    
    missing = [f for f in required_features if f not in features]
    
    if missing:
        print(f"❌ Features faltando: {missing}")
        return False
    
    return True


def get_feature_importance_explanation() -> dict:
    """
    Retorna explicação sobre cada feature e sua importância.
    Útil para debugging e interpretação do modelo.
    """
    return {
        "age": {
            "description": "Idade do usuário em anos",
            "range": "16-100",
            "importance": "Alta - Metabolismo muda com idade",
            "effect": "↑ Idade → ↑ Body Fat (tendência)"
        },
        "bmi": {
            "description": "Índice de Massa Corporal (peso/altura²)",
            "range": "12-50",
            "importance": "Muito Alta - Principal indicador de massa",
            "effect": "↑ BMI → ↑ Body Fat (forte correlação)"
        },
        "sex": {
            "description": "Sexo biológico (0=F, 1=M)",
            "range": "0 ou 1",
            "importance": "Alta - Diferenças hormonais",
            "effect": "Mulheres tendem a ter BF ~8-10% maior"
        },
        "shoulder_width": {
            "description": "Largura dos ombros (normalizada)",
            "range": "0.30-0.55",
            "importance": "Média - Estrutura óssea",
            "effect": "Homens: ombros largos = atlético"
        },
        "hip_width": {
            "description": "Largura do quadril (normalizada)",
            "range": "0.25-0.50",
            "importance": "Média - Distribuição de gordura",
            "effect": "Mulheres: quadris largos = feminino típico"
        },
        "height_ratio": {
            "description": "Proporção torso/largura",
            "range": "0.75-1.10",
            "importance": "Baixa - Estrutura geral",
            "effect": "Pouca influência direta"
        },
        "waist_ratio": {
            "description": "Razão cintura/ombro",
            "range": "0.35-0.85",
            "importance": "Muito Alta - Principal indicador visual",
            "effect": "↑ Waist Ratio → ↑ Body Fat (gordura abdominal)"
        },
        "volume_indicator": {
            "description": "Indicador de massa corporal visível",
            "range": "0.08-0.35",
            "importance": "Alta - Massa total aparente",
            "effect": "↑ Volume → ↑ Body Fat (em não-atletas)"
        },
        "waist_prominence": {
            "description": "Projeção abdominal (profundidade Z)",
            "range": "-0.02 a 0.10",
            "importance": "Alta - Gordura visceral",
            "effect": "↑ Prominence → ↑ Body Fat (barriga saliente)"
        }
    }