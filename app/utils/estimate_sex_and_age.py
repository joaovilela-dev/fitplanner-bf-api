"""
Estimativa de sexo e idade com VALIDAÇÃO SUAVE (Nível 2).
Mantém compatibilidade com código existente, mas adiciona validação opcional.
"""
from deepface import DeepFace
import cv2
from typing import Optional


def estimate_sex_and_age(
    image_path: str,
    user_sex: Optional[str] = None,
    user_age: Optional[int] = None,
    validate: bool = False
) -> dict:
    """
    Estima sexo e idade usando DeepFace (análise facial).
    
    **Modo 1 (validate=False):** Apenas detecta (comportamento original)
    **Modo 2 (validate=True):** Detecta + valida contra dados do usuário
    
    Args:
        image_path: Caminho da imagem
        user_sex: Sexo informado pelo usuário (opcional)
        user_age: Idade informada pelo usuário (opcional)
        validate: Se True, valida dados do usuário contra a foto
    
    Retorna:
        {
            "sex": "male" | "female" | None,
            "age": int | None,
            "confidence": float (0 a 1),
            "validation": {  # Apenas se validate=True
                "performed": bool,
                "sex_match": bool,
                "age_match": bool,
                "warnings": list[str]
            }
        }
    """
    try:
        # ===================================
        #  Análise com DeepFace
        # ===================================
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender'],
            enforce_detection=False,  # Continua mesmo se face não for perfeita
            detector_backend='opencv'  # Mais rápido
        )
        
        # DeepFace pode retornar lista ou dict
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Extrai informações detectadas
        detected_age = int(analysis.get('age', 25))
        gender_dict = analysis.get('gender', {})
        
        # DeepFace retorna {"Man": 99.8, "Woman": 0.2}
        man_confidence = gender_dict.get('Man', 0)
        woman_confidence = gender_dict.get('Woman', 0)
        
        if man_confidence > woman_confidence:
            detected_sex = "male"
            sex_confidence = man_confidence / 100
        else:
            detected_sex = "female"
            sex_confidence = woman_confidence / 100
        
        # ===================================
        #  Resultado Básico (sempre retorna)
        # ===================================
        result = {
            "sex": detected_sex,
            "age": detected_age,
            "confidence": round(sex_confidence, 2)
        }
        
        # ===================================
        #  Validação (apenas se solicitado)
        # ===================================
        if validate and user_sex and user_age:
            validation = _validate_against_user_data(
                detected_sex=detected_sex,
                detected_age=detected_age,
                user_sex=user_sex,
                user_age=user_age,
                confidence=sex_confidence
            )
            result["validation"] = validation
        else:
            result["validation"] = {
                "performed": False,
                "sex_match": None,
                "age_match": None,
                "warnings": []
            }
        
        return result
        
    except Exception as e:
        # ===================================
        #  Fallback: Erro na detecção
        # ===================================
        return {
            "sex": None,
            "age": None,
            "confidence": 0.0,
            "validation": {
                "performed": False,
                "sex_match": None,
                "age_match": None,
                "warnings": [f"Não foi possível analisar face: {str(e)}"]
            },
            "error": f"Não foi possível analisar face: {str(e)}"
        }


def _validate_against_user_data(
    detected_sex: str,
    detected_age: int,
    user_sex: str,
    user_age: int,
    confidence: float
) -> dict:
    """
    Valida dados detectados contra informações do usuário.
    Retorna avisos se houver divergências significativas.
    
    **Nível 2 (Soft):** Apenas avisa, não bloqueia.
    """
    warnings = []
    
    # Normalizar sexo do usuário
    user_sex_normalized = _normalize_sex(user_sex)
    
    # ===================================
    # Validação de Sexo
    # ===================================
    sex_match = (detected_sex == user_sex_normalized)
    
    if not sex_match and confidence >= 0.6:
        sex_label = "masculino" if detected_sex == "male" else "feminino"
        warnings.append(
            f"⚠️ A foto parece ser de uma pessoa do sexo {sex_label}, "
            f"mas você informou {user_sex_normalized}. "
            f"Confiança: {confidence*100:.0f}%"
        )
    elif not sex_match and confidence < 0.6:
        warnings.append(
            " A detecção de sexo teve baixa confiança. "
            "Usando seus dados informados."
        )
    
    # ===================================
    # Validação de Idade
    # ===================================
    age_diff = abs(detected_age - user_age)
    age_match = age_diff <= 5  # Margem de ±5 anos
    
    if age_diff > 10:
        warnings.append(
            f"⚠️ A idade aparente na foto é {detected_age} anos, "
            f"mas você informou {user_age} anos. "
            f"Diferença: {age_diff} anos."
        )
    elif age_diff > 5:
        warnings.append(
            f" Pequena diferença de idade detectada "
            f"(foto: {detected_age}, informado: {user_age})."
        )
    
    # ===================================
    # Resultado da Validação
    # ===================================
    return {
        "performed": True,
        "sex_match": sex_match,
        "age_match": age_match,
        "detected_sex": detected_sex,
        "detected_age": detected_age,
        "confidence": confidence,
        "warnings": warnings
    }


def _normalize_sex(sex: str) -> str:
    """Normaliza valores de sexo para 'male' ou 'female'"""
    sex_lower = str(sex).lower()
    
    if sex_lower in ['m', 'male', 'masculino', 'homem', 'man']:
        return 'male'
    elif sex_lower in ['f', 'female', 'feminino', 'mulher', 'woman']:
        return 'female'
    else:
        return sex_lower  # Retorna original se não reconhecer