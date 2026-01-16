#!/usr/bin/env python3
"""
Patch automático para adicionar central_fat no ensemble_predictor.py
"""

import os
import sys
import re

def apply_patch():
    filepath = "app/services/ensemble_predictor.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Arquivo não encontrado: {filepath}")
        return False
    
    print(f"📝 Lendo {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se já está aplicado
    if 'central_fat = texture_data.get("central_fat"' in content or \
       'central_fat = definition_data.get("central_fat"' in content:
        print("✅ Patch já aplicado! Nada a fazer.")
        return True
    
    # Procurar por vários padrões possíveis
    patterns = [
        # Padrão 1: abs_visibility seguido de qualquer coisa
        (r'(abs_visibility = texture_data\.get\("abs_visibility",\s*[\d.]+\))',
         r'\1\n    central_fat = texture_data.get("central_fat", 0.5)  # 🆕 NOVO'),
        
        # Padrão 2: definition_score e abs_visibility juntos
        (r'(definition_score = texture_data\.get\("definition_score",\s*[\d.]+\)\s*\n\s*abs_visibility = texture_data\.get\("abs_visibility",\s*[\d.]+\))',
         r'\1\n    central_fat = texture_data.get("central_fat", 0.5)  # 🆕 NOVO'),
    ]
    
    patched = False
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            patched = True
            print(f"✅ Padrão encontrado e aplicado!")
            break
    
    if not patched:
        print("⚠️ Nenhum padrão reconhecido encontrado.")
        print("\n🔍 Procurando manualmente por 'abs_visibility'...")
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'abs_visibility = texture_data.get' in line or \
               'abs_visibility = definition_data.get' in line:
                print(f"   Encontrado na linha {i+1}: {line.strip()}")
                
                # Adicionar linha após
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + 'central_fat = texture_data.get("central_fat", 0.5)  # 🆕 NOVO'
                lines.insert(i + 1, new_line)
                content = '\n'.join(lines)
                patched = True
                print(f"✅ Linha adicionada!")
                break
    
    if not patched:
        print("\n❌ Não foi possível aplicar o patch automaticamente")
        print("\n📋 Adicione manualmente esta linha no ensemble_predictor.py:")
        print('    central_fat = texture_data.get("central_fat", 0.5)')
        print("\n📍 Logo após a linha que contém:")
        print('    abs_visibility = texture_data.get("abs_visibility", ...)')
        return False
    
    # Fazer backup
    backup_path = filepath + ".before_patch"
    with open(backup_path, 'w', encoding='utf-8') as f:
        with open(filepath, 'r', encoding='utf-8') as original:
            f.write(original.read())
    print(f"💾 Backup do original salvo: {backup_path}")
    
    # Salvar arquivo corrigido
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Patch aplicado com sucesso!")
    return True

def main():
    print("=" * 70)
    print("🔧 PATCH: Adicionar central_fat ao ensemble_predictor.py")
    print("=" * 70)
    
    if apply_patch():
        print("\n✅ Correção concluída!")
        print("\n🚀 Próximo passo:")
        print("   1. Execute: python verify_v3_installation.py")
        print("   2. Se tudo estiver ✅, reinicie a API")
        return 0
    else:
        print("\n❌ Não foi possível aplicar o patch automaticamente")
        print("\n🔧 Correção manual:")
        print("   1. Abra: app/services/ensemble_predictor.py")
        print("   2. Procure por: abs_visibility = texture_data.get")
        print("   3. Adicione logo após:")
        print('      central_fat = texture_data.get("central_fat", 0.5)')
        return 1

if __name__ == "__main__":
    sys.exit(main())