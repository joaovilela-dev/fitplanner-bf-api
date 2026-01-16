#!/usr/bin/env python3
"""
Script de treinamento CALIBRADO para Body Fat Prediction.
 V2: Dados sintéticos mais realistas que seguem as regras fisiológicas
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime


# ======================================================
#  GERAÇÃO DE DATASET SINTÉTICO CALIBRADO
# ======================================================

def generate_synthetic_dataset(n_samples=1500):
    """
    Gera dataset sintético CALIBRADO com as regras fisiológicas.
    
     V2 MELHORIAS:
    - Body Fat calculado usando as MESMAS REGRAS do body_fat_logic.py
    - Menos ruído para evitar ML aprender padrões errados
    - Mais amostras em regiões críticas (atletas, sobrepeso)
    """
    
    np.random.seed(42)
    data = []
    
    # ===================================
    # CATEGORIAS DE COMPOSIÇÃO CORPORAL
    # ===================================
    
    categories = [
        # (n_samples, bf_base, bmi_range, waist_range, volume_range, name, sex)
        
        # HOMENS
        (200, 8, (18, 22), (0.38, 0.45), (0.08, 0.14), "Atleta Elite - Masculino", 1),
        (250, 12, (22, 24), (0.45, 0.52), (0.10, 0.16), "Atlético - Masculino", 1),
        (300, 15, (23, 26), (0.50, 0.58), (0.12, 0.18), "Fitness - Masculino", 1),
        (200, 20, (25, 28), (0.56, 0.65), (0.15, 0.22), "Médio - Masculino", 1),
        (150, 28, (28, 33), (0.63, 0.75), (0.18, 0.28), "Sobrepeso - Masculino", 1),
        (50, 38, (33, 42), (0.72, 0.85), (0.22, 0.35), "Obesidade - Masculino", 1),
        
        # MULHERES
        (150, 16, (18, 21), (0.42, 0.48), (0.10, 0.16), "Atlética - Feminino", 0),
        (200, 20, (21, 24), (0.48, 0.56), (0.12, 0.19), "Fitness - Feminino", 0),
        (200, 27, (24, 28), (0.55, 0.65), (0.15, 0.23), "Média - Feminino", 0),
        (150, 35, (28, 35), (0.63, 0.75), (0.18, 0.30), "Sobrepeso - Feminino", 0),
    ]
    
    for n, bf_base, bmi_range, waist_range, vol_range, category, sex in categories:
        for _ in range(n):
            
            # ===================================
            #  GERAR PARÂMETROS BASE
            # ===================================
            
            # BMI com pequena variação
            bmi = np.random.uniform(*bmi_range)
            
            # Idade (afeta BF levemente)
            age = np.random.randint(18, 65)
            age_factor = (age - 25) * 0.1  # Pequeno aumento com idade
            
            # Waist ratio correlacionado com BF
            waist_ratio = np.random.uniform(*waist_range)
            
            # Volume correlacionado com BF
            volume_indicator = np.random.uniform(*vol_range)
            
            # Waist prominence correlacionado com BF
            if sex == 1:  # Masculino
                waist_prominence = 0.001 + (waist_ratio - 0.45) * 0.08
            else:  # Feminino
                waist_prominence = 0.001 + (waist_ratio - 0.50) * 0.06
            
            waist_prominence = np.clip(waist_prominence + np.random.normal(0, 0.003), -0.01, 0.08)
            
            # Estrutura corporal
            if sex == 1:
                shoulder_width = np.random.uniform(0.38, 0.48)
                hip_width = shoulder_width * np.random.uniform(0.80, 0.92)
            else:
                shoulder_width = np.random.uniform(0.32, 0.40)
                hip_width = shoulder_width * np.random.uniform(0.95, 1.10)
            
            height_ratio = np.random.uniform(0.85, 1.05)
            
            # ===================================
            #  CALCULAR BF USANDO AS REGRAS 
            # ===================================
            
            # Base pelo tipo corporal
            bf = bf_base
            
            # Ajuste por BMI (mesma lógica do body_fat_logic.py)
            if bmi < 18.5:
                bmi_adj = -3
            elif 18.5 <= bmi < 22:
                bmi_adj = -1
            elif 22 <= bmi < 25:
                bmi_adj = 1
            elif 25 <= bmi < 27.5:
                bmi_adj = 4
            elif 27.5 <= bmi < 30:
                bmi_adj = 7
            elif 30 <= bmi < 32.5:
                bmi_adj = 10
            else:
                bmi_adj = 14
            
            bf += bmi_adj
            
            # Ajuste por waist ratio
            if sex == 1:  # Masculino
                if waist_ratio > 0.70:
                    waist_adj = 6
                elif waist_ratio > 0.65:
                    waist_adj = 4
                elif waist_ratio > 0.58:
                    waist_adj = 2
                elif waist_ratio > 0.52:
                    waist_adj = 1
                else:
                    waist_adj = 0
            else:  # Feminino
                if waist_ratio > 0.75:
                    waist_adj = 5
                elif waist_ratio > 0.70:
                    waist_adj = 3
                elif waist_ratio > 0.62:
                    waist_adj = 2
                elif waist_ratio > 0.55:
                    waist_adj = 1
                else:
                    waist_adj = 0
            
            bf += waist_adj
            
            # Ajuste por volume
            if volume_indicator > 0.25:
                vol_adj = 3
            elif volume_indicator > 0.20:
                vol_adj = 2
            elif volume_indicator > 0.15:
                vol_adj = 1
            else:
                vol_adj = 0
            
            bf += vol_adj
            
            # Ajuste por waist prominence
            if waist_prominence > 0.03:
                prom_adj = 4
            elif waist_prominence > 0.01:
                prom_adj = 2
            elif waist_prominence > 0:
                prom_adj = 1
            else:
                prom_adj = 0
            
            bf += prom_adj
            
            # Ajuste por idade
            bf += age_factor
            
            noise = np.random.normal(0, 1.0)  # Reduzido de 2.0 para 1.0
            bf += noise
            
            # Limites fisiológicos
            if sex == 1:
                bf = np.clip(bf, 5, 45)
            else:
                bf = np.clip(bf, 12, 50)
            
            # ===================================
            #  ADICIONAR AO DATASET
            # ===================================
            
            data.append({
                'age': age,
                'bmi': round(bmi, 2),
                'sex': sex,
                'shoulder_width': round(shoulder_width, 4),
                'hip_width': round(hip_width, 4),
                'height_ratio': round(height_ratio, 4),
                'waist_ratio': round(waist_ratio, 4),
                'volume_indicator': round(volume_indicator, 4),
                'waist_prominence': round(waist_prominence, 4),
                'body_fat_percentage': round(bf, 1),
                'category': category
            })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Dataset gerado: {len(df)} amostras")
    print(f"\n📊 Distribuição por categoria:")
    print(df['category'].value_counts().sort_index())
    print(f"\n📈 Estatísticas de Body Fat:")
    print(df['body_fat_percentage'].describe())
    
    #  Validação: verificar se BF está coerente com BMI
    for sex_val in [0, 1]:
        sex_name = "Masculino" if sex_val == 1 else "Feminino"
        df_sex = df[df['sex'] == sex_val]
        
        low_bmi_high_bf = df_sex[(df_sex['bmi'] < 22) & (df_sex['body_fat_percentage'] > 25)]
        high_bmi_low_bf = df_sex[(df_sex['bmi'] > 30) & (df_sex['body_fat_percentage'] < 20)]
        
        print(f"\n🔍 Validação ({sex_name}):")
        print(f"   BMI baixo + BF alto: {len(low_bmi_high_bf)} amostras (deve ser ~0)")
        print(f"   BMI alto + BF baixo: {len(high_bmi_low_bf)} amostras (deve ser ~0)")
    
    return df


# ======================================================
# TREINAMENTO DO MODELO (IGUAL)
# ======================================================

def train_model(df):
    """Treina modelo Random Forest otimizado"""
    
    feature_columns = [
        'age', 'bmi', 'sex',
        'shoulder_width', 'hip_width', 'height_ratio', 'waist_ratio',
        'volume_indicator', 'waist_prominence'
    ]
    
    X = df[feature_columns]
    y = df['body_fat_percentage']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n🔄 Split: {len(X_train)} treino, {len(X_test)} teste")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid Search
    print("\n🔍 Otimizando hiperparâmetros...")
    
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='neg_mean_absolute_error',
        verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"\n✅ Melhores parâmetros:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Validação Cruzada
    print("\n📊 Validação Cruzada (5-fold):")
    cv_scores = cross_val_score(
        best_model, X_train_scaled, y_train,
        cv=5, scoring='neg_mean_absolute_error'
    )
    print(f"   MAE médio: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}%")
    
    # Avaliação no teste
    y_pred = best_model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Métricas no Teste:")
    print(f"   MAE:  {mae:.2f}%")
    print(f"   RMSE: {rmse:.2f}%")
    print(f"   R²:   {r2:.3f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']:20s} {row['importance']:.4f}")
    
    # 🆕 Análise de erros por faixa
    errors = np.abs(y_test - y_pred)
    df_test = pd.DataFrame({'true_bf': y_test, 'pred_bf': y_pred, 'error': errors})
    
    print(f"\n📊 Erros por Faixa de BF:")
    bf_ranges = [(5, 15), (15, 25), (25, 35), (35, 50)]
    for low, high in bf_ranges:
        mask = (df_test['true_bf'] >= low) & (df_test['true_bf'] < high)
        if mask.sum() > 0:
            range_mae = df_test[mask]['error'].mean()
            range_std = df_test[mask]['error'].std()
            print(f"   {low:2d}-{high:2d}%: MAE = {range_mae:.2f}% ± {range_std:.2f}%")
    
    return best_model, scaler, feature_importance


# ======================================================
# SALVAR MODELO
# ======================================================

def save_model_artifacts(model, scaler, feature_importance, df):
    """Salva modelo, scaler e metadados"""
    
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, "models/bodyfat_model.pkl")
    print(f"\n💾 Modelo salvo: models/bodyfat_model.pkl")
    
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"💾 Scaler salvo: models/scaler.pkl")
    
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'version': '2.0 - Calibrated',
        'n_samples': len(df),
        'features': [
            'age', 'bmi', 'sex', 'shoulder_width', 'hip_width',
            'height_ratio', 'waist_ratio', 'volume_indicator', 'waist_prominence'
        ],
        'model_type': 'RandomForestRegressor',
        'feature_importance': feature_importance.to_dict('records')
    }
    
    import json
    with open("models/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata salvo: models/metadata.json")
    
    df.to_csv("models/training_dataset.csv", index=False)
    print(f"💾 Dataset salvo: models/training_dataset.csv")


# ======================================================
# VISUALIZAÇÕES
# ======================================================

def plot_results(df, model, scaler):
    """Gera gráficos de análise"""
    
    feature_columns = [
        'age', 'bmi', 'sex', 'shoulder_width', 'hip_width',
        'height_ratio', 'waist_ratio', 'volume_indicator', 'waist_prominence'
    ]
    
    X = df[feature_columns]
    y = df['body_fat_percentage']
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predito vs Real
    axes[0, 0].scatter(y, y_pred, alpha=0.3, s=10)
    axes[0, 0].plot([5, 50], [5, 50], 'r--', lw=2)
    axes[0, 0].set_xlabel('Body Fat Real (%)')
    axes[0, 0].set_ylabel('Body Fat Predito (%)')
    axes[0, 0].set_title('Predições vs Valores Reais')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribuição de erros
    errors = y_pred - y
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Erro de Predição (%)')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title(f'Distribuição de Erros (MAE={np.abs(errors).mean():.2f}%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. BF por BMI
    male_mask = df['sex'] == 1
    axes[1, 0].scatter(df[male_mask]['bmi'], df[male_mask]['body_fat_percentage'],
                      alpha=0.3, s=10, label='Masculino', color='blue')
    axes[1, 0].scatter(df[~male_mask]['bmi'], df[~male_mask]['body_fat_percentage'],
                      alpha=0.3, s=10, label='Feminino', color='pink')
    axes[1, 0].set_xlabel('BMI')
    axes[1, 0].set_ylabel('Body Fat (%)')
    axes[1, 0].set_title('Body Fat vs BMI por Sexo')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Erro por faixa
    df_plot = pd.DataFrame({'bf': y, 'error': np.abs(errors)})
    df_plot['bf_range'] = pd.cut(df_plot['bf'], bins=[0, 15, 25, 35, 50],
                                  labels=['5-15%', '15-25%', '25-35%', '35-50%'])
    
    df_plot.boxplot(column='error', by='bf_range', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Faixa de Body Fat')
    axes[1, 1].set_ylabel('Erro Absoluto (%)')
    axes[1, 1].set_title('Distribuição de Erros por Faixa')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig("models/training_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráficos salvos: models/training_analysis.png")
    plt.show()


# ======================================================
# MAIN
# ======================================================

def main():
    print("=" * 70)
    print("🏋️ BODY FAT PREDICTOR - TREINAMENTO V2 CALIBRADO")
    print("=" * 70)
    
    print("\n1️⃣ Gerando dataset sintético CALIBRADO...")
    df = generate_synthetic_dataset(n_samples=1500)
    
    print("\n2️⃣ Treinando modelo...")
    model, scaler, feature_importance = train_model(df)
    
    print("\n3️⃣ Salvando modelo e artefatos...")
    save_model_artifacts(model, scaler, feature_importance, df)
    
    print("\n4️⃣ Gerando visualizações...")
    try:
        plot_results(df, model, scaler)
    except Exception as e:
        print(f"⚠️ Erro ao gerar gráficos: {e}")
    
    print("\n" + "=" * 70)
    print("✅ TREINAMENTO V2 CONCLUÍDO!")
    print("=" * 70)
    print("\n📁 Arquivos gerados:")
    print("   - models/bodyfat_model.pkl")
    print("   - models/scaler.pkl")
    print("   - models/metadata.json")
    print("   - models/training_dataset.csv")
    print("\n🚀 Próximo passo: Reiniciar a API")
    print("   uvicorn main:app --reload")


if __name__ == "__main__":
    main()