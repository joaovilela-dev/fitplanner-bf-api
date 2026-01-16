import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.services.bf_ml import BodyFatRegressor
import sys


def generate_test_cases():
    """
    Gera casos de teste realistas para avaliar o modelo.
    Simula diferentes tipos corporais e perfis.
    """
    test_cases = []
    
    # Caso 1: Homem magro jovem
    test_cases.append({
        "name": "Homem magro jovem (atleta)",
        "age": 25,
        "bmi": 21.5,
        "sex": 1,
        "shoulder_width": 0.48,
        "hip_width": 0.35,
        "height_ratio": 0.92,
        "waist_ratio": 0.48,
        "volume_indicator": 0.12,
        "waist_prominence": -0.01,
        "expected_bf": 10.0  # 8-12%
    })
    
    # Caso 2: Homem com sobrepeso (como a foto que você enviou)
    test_cases.append({
        "name": "Homem com sobrepeso (BMI ~28)",
        "age": 35,
        "bmi": 28.0,
        "sex": 1,
        "shoulder_width": 0.45,
        "hip_width": 0.40,
        "height_ratio": 0.88,
        "waist_ratio": 0.68,
        "volume_indicator": 0.22,
        "waist_prominence": 0.04,
        "expected_bf": 26.0  # 24-28%
    })
    
    # Caso 3: Homem obeso
    test_cases.append({
        "name": "Homem obeso (BMI 33)",
        "age": 42,
        "bmi": 33.0,
        "sex": 1,
        "shoulder_width": 0.44,
        "hip_width": 0.45,
        "height_ratio": 0.85,
        "waist_ratio": 0.82,
        "volume_indicator": 0.28,
        "waist_prominence": 0.07,
        "expected_bf": 35.0  # 32-38%
    })
    
    # Caso 4: Mulher magra jovem
    test_cases.append({
        "name": "Mulher magra jovem (fitness)",
        "age": 27,
        "bmi": 20.5,
        "sex": 0,
        "shoulder_width": 0.38,
        "hip_width": 0.42,
        "height_ratio": 0.90,
        "waist_ratio": 0.52,
        "volume_indicator": 0.13,
        "waist_prominence": -0.01,
        "expected_bf": 20.0  # 18-22%
    })
    
    # Caso 5: Mulher com sobrepeso
    test_cases.append({
        "name": "Mulher com sobrepeso (BMI 29)",
        "age": 38,
        "bmi": 29.0,
        "sex": 0,
        "shoulder_width": 0.36,
        "hip_width": 0.48,
        "height_ratio": 0.87,
        "waist_ratio": 0.72,
        "volume_indicator": 0.24,
        "waist_prominence": 0.03,
        "expected_bf": 35.0  # 33-37%
    })
    
    # Caso 6: Homem musculoso
    test_cases.append({
        "name": "Homem musculoso (bodybuilder)",
        "age": 30,
        "bmi": 27.0,
        "sex": 1,
        "shoulder_width": 0.52,
        "hip_width": 0.38,
        "height_ratio": 0.95,
        "waist_ratio": 0.50,
        "volume_indicator": 0.18,
        "waist_prominence": -0.005,
        "expected_bf": 14.0  # 12-16%
    })
    
    return test_cases


def evaluate_model():
    """
    Avalia o modelo ML com casos de teste realistas.
    """
    print("=" * 60)
    print("🔬 AVALIAÇÃO DO MODELO DE BODY FAT")
    print("=" * 60)
    
    try:
        regressor = BodyFatRegressor()
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\n💡 Execute primeiro: python scripts/train_bf.py")
        sys.exit(1)
    
    test_cases = generate_test_cases()
    
    predictions = []
    expected = []
    errors = []
    
    print("\n📊 RESULTADOS POR CASO:\n")
    
    for case in test_cases:
        name = case.pop("name")
        expected_bf = case.pop("expected_bf")
        
        predicted_bf = regressor.predict(case)
        error = abs(predicted_bf - expected_bf)
        
        predictions.append(predicted_bf)
        expected.append(expected_bf)
        errors.append(error)
        
        # Emoji de status
        if error <= 2:
            status = "✅"
        elif error <= 4:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"{status} {name}")
        print(f"   Esperado: {expected_bf:.1f}% | Predito: {predicted_bf:.1f}% | Erro: {error:.1f}%")
        print()
    
    # Métricas gerais
    mae = mean_absolute_error(expected, predictions)
    rmse = np.sqrt(mean_squared_error(expected, predictions))
    r2 = r2_score(expected, predictions)
    
    print("=" * 60)
    print("📈 MÉTRICAS GERAIS:")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error):  {mae:.2f}%")
    print(f"RMSE (Root Mean Squared):   {rmse:.2f}%")
    print(f"R² Score:                   {r2:.3f}")
    print(f"Erro Médio:                 {np.mean(errors):.2f}%")
    print(f"Erro Máximo:                {np.max(errors):.2f}%")
    
    print("\n" + "=" * 60)
    
    if mae <= 3.0:
        print("✅ Modelo com performance EXCELENTE!")
    elif mae <= 5.0:
        print("⚠️ Modelo com performance BOA (aceitável)")
    else:
        print("❌ Modelo precisa de ajustes")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_model()