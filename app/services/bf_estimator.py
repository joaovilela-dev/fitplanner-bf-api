import os
import json
import uuid

from app.utils.image_processing import extract_body_measurements
from app.utils.estimate_sex_and_age import estimate_sex_and_age
from app.services.body_ratios import calculate_body_ratios
from app.services.body_classifier import classify_body_type
from app.services.bf_ml import BodyFatRegressor
from app.services.bf_features import build_features
from app.utils.image_validation import validate_image_content
from app.services.texture_analyzer import analyze_muscle_definition
from app.services.ensemble_predictor import ensemble_predict_body_fat

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def _save_log(result: dict):
    """Salva os resultados da estimativa para auditoria."""
    try:
        filename = os.path.join(LOG_DIR, f"{uuid.uuid4()}.json")
        
        # Tentar converter tipos NumPy
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()  # Converte para Python nativo
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        result_safe = convert_types(result)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_safe, f, ensure_ascii=False, indent=4)
    except Exception as e:
        # Se falhar ao salvar log, não quebrar a API
        print(f"⚠️ Erro ao salvar log: {e}")

def check_age_visual(predicted_age: int, minimum_age: int = 16) -> dict:
    """Bloqueia imagens que aparentem ser de menores de idade."""
    if predicted_age and predicted_age < minimum_age:
        return {
            "error": "A foto enviada não parece ser de um adolescente ou adulto.",
            "block_reason": "UNDERAGE",
            "person_detected": True,
            "predicted_age": predicted_age
        }
    return {"ok": True}


def estimate_body_composition(
    image_path: str,
    age: int,
    weight_kg: float,
    height_cm: float,
    sex: str,
    force: bool = False,
    use_ml: bool = True,
    validate_photo: bool = True,
    use_ensemble: bool = True
) -> dict:
    """
    Estima composição corporal com ENSEMBLE V3 SAFE MODE.
    
    🔒 MODO SAFE (padrão):
        - BF Final = Regras (60%) + Textura (40%)
        - ML registrado mas não influencia
        - Maior confiabilidade
    """

    measurements = {}
    alerts = []

    try:
        # ===============================
        #  Validação da imagem
        # ===============================
        validation = validate_image_content(image_path)
        if not validation["valid"]:
            return {
                "error": "Imagem inválida para análise",
                "reason": validation.get("error"),
                "person_detected": validation.get("person_count", 0) > 0,
                "alerts": [validation["message"]],
                "raw_measurements": {}
            }

        # ===============================
        #  Extração de medidas (REAL com MediaPipe)
        # ===============================
        measurements = extract_body_measurements(image_path)

        if not measurements.get("person_detected"):
            return {
                "error": "Nenhuma pessoa detectada na imagem.",
                "person_detected": False
            }

        # ===============================
        #  ANÁLISE DE TEXTURA/DEFINIÇÃO MUSCULAR
        # ===============================
        print("\n🔬 Analisando definição muscular...")
        texture_data = analyze_muscle_definition(image_path)
        
        print(f"   Definition Score: {texture_data['definition_score']:.3f}")
        print(f"   Abs Visibility:   {texture_data['abs_visibility']:.3f}")
        print(f"   Vascularity:      {texture_data['vascularity']:.3f}")
        print(f"   Confidence:       {texture_data['confidence']:.3f}")

        # ===============================
        #  Estimativa visual (validação)
        # ===============================
        visual_info = estimate_sex_and_age(
            image_path=image_path,
            user_sex=sex,
            user_age=age,
            validate=validate_photo
        )
        
        visual_sex = visual_info.get("sex")
        visual_age = visual_info.get("age")
        confidence = visual_info.get("confidence", 0)
        validation_result = visual_info.get("validation", {})

        # Bloqueio por idade
        if visual_age is not None:
            age_check = check_age_visual(visual_age)
            if "error" in age_check:
                return age_check

        # Divergência de sexo
        sexo_usado_no_calculo = sex
        
        if validate_photo and validation_result.get("performed"):
            validation_warnings = validation_result.get("warnings", [])
            alerts.extend(validation_warnings)
            
            if not validation_result.get("sex_match") and confidence >= 0.7:
                alerts.append(
                    "⚠️ IMPORTANTE: Os dados informados podem não corresponder "
                    "à pessoa na foto. Verifique se estão corretos."
                )

        # ===============================
        #  Cálculo de proporções corporais
        # ===============================
        ratios = calculate_body_ratios(measurements)

        # ===============================
        #  IMC
        # ===============================
        height_m = float(height_cm) / 100
        bmi = float(weight_kg) / (height_m ** 2)

        # ===============================
        #  Tipo corporal
        # ===============================
        body_type = classify_body_type(ratios, sexo_usado_no_calculo, bmi)

        # ===============================
        #  PREDIÇÃO ML
        # ===============================
        ml_prediction = None
        if use_ml:
            try:
                regressor = BodyFatRegressor()
                features = build_features(
                    measurements=measurements,
                    age=age,
                    weight_kg=weight_kg,
                    height_cm=height_cm,
                    sex=sexo_usado_no_calculo
                )
                ml_prediction = regressor.predict(features)
                print(f"\n🤖 Predição ML: {ml_prediction}%")
                
            except Exception as e:
                alerts.append(f"Modelo ML não disponível: {str(e)}")
                ml_prediction = None

        # ===============================
        #  PREDIÇÃO BASEADA EM REGRAS
        # ===============================
        from app.services.body_fat_logic import estimate_body_fat
        rules_prediction = estimate_body_fat(
            body_type, bmi, sexo_usado_no_calculo, ratios, measurements
        )
        print(f"📏 Predição Regras: {rules_prediction}%")

        # ===============================
        #  ENSEMBLE V3 SAFE MODE
        # ===============================
        if use_ensemble and ml_prediction is not None:
            ensemble_result = ensemble_predict_body_fat(
                ml_prediction=ml_prediction,
                rules_prediction=rules_prediction,
                texture_data=texture_data,
                bmi=bmi,
                sex=sexo_usado_no_calculo,
                measurements=measurements,
                ratios=ratios
            )
            
            body_fat = ensemble_result["safe_prediction"]
            estimation_method = f"Ensemble V3 - {ensemble_result.get('mode', 'SAFE')} MODE"
            
            alerts.extend(ensemble_result.get("adjustments", []))
            
            ensemble_info = {
                "mode": ensemble_result.get("mode", "SAFE"),
                "ml_prediction": ensemble_result["ml_prediction"],
                "rules_prediction": ensemble_result["rules_prediction"],
                "texture_prediction": ensemble_result["texture_prediction"],
                "safe_prediction": ensemble_result["safe_prediction"],
                "experimental_prediction": ensemble_result.get("experimental_prediction"),
                
                #  Pesos SAFE (Regras + Textura)
                "safe_weights": ensemble_result.get("safe_weights", {}),
                
                #  Pesos EXPERIMENTAL (se ativo)
                "experimental_weights": ensemble_result.get("experimental_weights"),
                
                # Metadados
                "confidence": ensemble_result["confidence"],
                "confidence_level": ensemble_result.get("confidence_level", "Média"),
                
                #  Análise do ML (em quarentena)
                "ml_analysis": ensemble_result.get("ml_analysis", {}),
                
                # Casos especiais detectados
                "special_cases": ensemble_result.get("special_cases", {})
            }
            
        else:
            # Fallback: usar apenas regras
            body_fat = rules_prediction
            estimation_method = "Rule-based"
            ensemble_info = None

        # ===============================
        #  Análise detalhada
        # ===============================
        from app.services.bf_validator import get_detailed_analysis
        
        detailed_analysis = get_detailed_analysis(
            bf=body_fat,
            bmi=bmi,
            sex=sexo_usado_no_calculo,
            age=age,
            ratios=ratios
        )

        # ===============================
        #  Resultado final
        # ===============================
        result = {
            "person_detected": True,
            "body_type": body_type,
            "body_fat_percentage": body_fat,
            "bmi": round(bmi, 1),
            "estimation_method": estimation_method,
            "sex_used_for_calculation": sexo_usado_no_calculo,
            "ratios": ratios,
            "measurements": {
                "shoulder_width": round(measurements["shoulder_width"], 4),
                "hip_width": round(measurements["hip_width"], 4),
                "waist_width": round(measurements["waist_width"], 4),
                "volume_indicator": round(measurements.get("volume_indicator", 0), 4),
                "waist_prominence": round(measurements.get("waist_prominence", 0), 4)
            },
            "alerts": alerts,
            "visual_info": {
                "detected_sex": visual_sex,
                "detected_age": visual_age,
                "confidence": confidence
            },
            "texture_analysis": texture_data,
            "ensemble_info": ensemble_info,
            "detailed_analysis": detailed_analysis,
            "disclaimer": (
                "Estimativa educacional baseada em análise visual avançada e machine learning. "
                "Não substitui avaliação clínica ou profissional."
            )
        }
        
        if validate_photo and not validation_result.get("sex_match") and confidence >= 0.7:
            result["validation_alert"] = {
                "severity": "warning",
                "message": "Possível divergência entre dados informados e pessoa na foto",
                "recommendation": "Verifique se a foto e os dados estão corretos antes de prosseguir"
            }

        _save_log(result)
        return result

    except Exception as e:
        import traceback
        print(f"\n❌ ERRO CRÍTICO:")
        print(traceback.format_exc())
        
        return {
            "error": "Erro ao processar medições",
            "detail": str(e),
            "raw_measurements": measurements
        }