#!/usr/bin/env python3
"""
Script de verificação para garantir que V3 SAFE + Texture V2 foram instalados corretamente.
"""

import os
import sys

def check_file_exists(filepath):
    """Verifica se arquivo existe"""
    if os.path.exists(filepath):
        print(f"✅ Arquivo encontrado: {filepath}")
        return True
    else:
        print(f"❌ Arquivo NÃO encontrado: {filepath}")
        return False

def check_file_content(filepath, markers):
    """Verifica se arquivo contém marcadores específicos"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    for name, marker in markers.items():
        found = marker in content
        results[name] = found
        
        if found:
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name} - NÃO ENCONTRADO!")
    
    return all(results.values())

def main():
    print("=" * 70)
    print("🔍 VERIFICAÇÃO DE INSTALAÇÃO - V3 SAFE + TEXTURE V2")
    print("=" * 70)
    
    all_ok = True
    
    # ===================================
    # 1. VERIFICAR ENSEMBLE PREDICTOR V3
    # ===================================
    print("\n📋 1. Verificando ensemble_predictor.py...")
    
    ensemble_path = "app/services/ensemble_predictor.py"
    
    if not check_file_exists(ensemble_path):
        all_ok = False
    else:
        ensemble_markers = {
            "USE_EXPERIMENTAL_ML": "USE_EXPERIMENTAL_ML = os.getenv",
            "MODO SAFE": "🔒 MODO SAFE",
            "central_fat usado": 'central_fat = texture_data.get("central_fat"',
            "safe_prediction": "safe_prediction =",
            "safe_weights": "safe_weights",
        }
        
        if not check_file_content(ensemble_path, ensemble_markers):
            print("\n⚠️ ensemble_predictor.py parece estar INCORRETO ou INCOMPLETO!")
            all_ok = False
        else:
            print("\n✅ ensemble_predictor.py V3 SAFE instalado corretamente!")
    
    # ===================================
    # 2. VERIFICAR TEXTURE ANALYZER V2
    # ===================================
    print("\n📋 2. Verificando texture_analyzer.py...")
    
    texture_path = "app/services/texture_analyzer.py"
    
    if not check_file_exists(texture_path):
        all_ok = False
    else:
        texture_markers = {
            "_detect_central_fat (função)": "def _detect_central_fat(gray:",
            "_analyze_abdominal_region_v2": "def _analyze_abdominal_region_v2(",
            "_detect_vascularity_v2": "def _detect_vascularity_v2(",
            "central_fat no retorno": '"central_fat": round(float(central_fat_score)',
            "Thresholds rigorosos (35)": "horizontal_features / 35",
        }
        
        if not check_file_content(texture_path, texture_markers):
            print("\n⚠️ texture_analyzer.py parece estar INCORRETO ou INCOMPLETO!")
            all_ok = False
        else:
            print("\n✅ texture_analyzer.py V2 instalado corretamente!")
    
    # ===================================
    # 3. VERIFICAR bf_features.py
    # ===================================
    print("\n📋 3. Verificando bf_features.py...")
    
    features_path = "app/services/bf_features.py"
    
    if not check_file_exists(features_path):
        print("⚠️ bf_features.py não encontrado - pode causar erro!")
        print("   Use o código fornecido no artifact 'bf_features_complete'")
        all_ok = False
    else:
        features_markers = {
            "build_features": "def build_features(",
            "validate_features": "def validate_features(",
        }
        
        if check_file_content(features_path, features_markers):
            print("\n✅ bf_features.py instalado corretamente!")
        else:
            print("\n⚠️ bf_features.py existe mas pode estar incompleto")
    
    # ===================================
    # RESULTADO FINAL
    # ===================================
    print("\n" + "=" * 70)
    
    if all_ok:
        print("✅ INSTALAÇÃO COMPLETA E CORRETA!")
        print("=" * 70)
        print("\n🚀 Próximo passo:")
        print("   uvicorn main:app --reload")
        print("\n📊 Após reiniciar, você deve ver nos logs:")
        print("   - 'Central Fat: X.XXX'")
        print("   - '🔒 MODO SAFE - Predições:'")
        print("   - 'safe_prediction' no JSON de resposta")
        return 0
    else:
        print("❌ INSTALAÇÃO INCOMPLETA OU INCORRETA!")
        print("=" * 70)
        print("\n🔧 Ações necessárias:")
        print("   1. Verifique os arquivos marcados com ❌")
        print("   2. Substitua pelos códigos dos artifacts")
        print("   3. Execute este script novamente")
        print("\n📁 Artifacts necessários:")
        print("   - ensemble_predictor_v3_safe")
        print("   - texture_analyzer_v2")
        print("   - bf_features_complete")
        return 1

if __name__ == "__main__":
    sys.exit(main())