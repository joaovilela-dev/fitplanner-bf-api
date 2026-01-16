import os
import joblib
import numpy as np


class BodyFatRegressor:
    def __init__(self):
        model_path = os.path.join("models", "bodyfat_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ Modelo não encontrado em {model_path}. "
                "Execute: python scripts/train_bf.py"
            )
        
        self.model = joblib.load(model_path)
        print(f"✅ Modelo carregado: {model_path}")

    def predict(self, features: dict) -> float:
        """
        Prediz Body Fat % usando TODAS as features relevantes.
        
        Features esperadas (9 no total):
        - age: idade em anos
        - bmi: índice de massa corporal
        - sex: 0=female, 1=male
        - shoulder_width: largura dos ombros (normalizada)
        - hip_width: largura do quadril (normalizada)
        - height_ratio: proporção altura/largura
        - waist_ratio: cintura/quadril
        - volume_indicator: indicador de volume corporal
        - waist_prominence: proeminência abdominal
        
        Returns:
            float: Body Fat % (5-45 para homens, 12-50 para mulheres)
        """
        
        # ===================================
        # ORDEM CORRETA DAS FEATURES
        # ===================================
        # DEVE CORRESPONDER À ORDEM DO TREINAMENTO!
        X = np.array([[
            features["age"],
            features["bmi"],
            features["sex"],
            features["shoulder_width"],
            features["hip_width"],
            features["height_ratio"],
            features["waist_ratio"],
            features["volume_indicator"],
            features["waist_prominence"]
        ]], dtype=float)
        
        # ===================================
        # VALIDAÇÃO DE SANIDADE
        # ===================================
        # Verificar se valores estão em ranges razoáveis
        if features["age"] < 16 or features["age"] > 100:
            print(f"⚠️ Idade fora do range: {features['age']}")
        
        if features["bmi"] < 15 or features["bmi"] > 50:
            print(f"⚠️ BMI fora do range: {features['bmi']}")
        
        # ===================================
        # PREDIÇÃO
        # ===================================
        prediction = self.model.predict(X)[0]
        
        # ===================================
        # PÓS-PROCESSAMENTO
        # ===================================
        # Aplicar limites fisiológicos realistas
        sex_str = "male" if features["sex"] == 1 else "female"
        
        if sex_str == "male":
            # Homens: 5-45%
            prediction = max(5.0, min(prediction, 45.0))
        else:
            # Mulheres: 12-50%
            prediction = max(12.0, min(prediction, 50.0))
        
        # ===================================
        # DEBUG INFO
        # ===================================
        print(f"📊 Predição ML:")
        print(f"   Age: {features['age']} | BMI: {features['bmi']:.1f} | Sex: {sex_str}")
        print(f"   Waist Ratio: {features['waist_ratio']:.3f}")
        print(f"   Volume Indicator: {features['volume_indicator']:.3f}")
        print(f"   Waist Prominence: {features['waist_prominence']:.3f}")
        print(f"   → Body Fat: {prediction:.1f}%")
        
        return round(float(prediction), 1)