#!/usr/bin/env python3
"""
Script para verificar se todas as dependências estão instaladas.
Execute: python check_dependencies.py
"""

import sys
import subprocess

# Lista de pacotes necessários
REQUIRED_PACKAGES = {
    'fastapi': '0.104.1',
    'uvicorn': '0.24.0',
    'python-multipart': '0.0.6',
    'aiofiles': '23.2.1',
    'scikit-learn': '1.3.2',
    'joblib': '1.3.2',
    'numpy': '1.24.3',
    'opencv-python': '4.8.1.78',
    'mediapipe': '0.10.8',
    'ultralytics': '8.0.196',
    'Pillow': '10.1.0',
    'deepface': '0.0.79',
    'tf-keras': '2.15.0',
    'pydantic': '2.5.0',
    'python-dotenv': '1.0.0'
}

def check_python_version():
    """Verifica versão do Python"""
    print("🐍 Verificando versão do Python...")
    version = sys.version_info
    print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        print("   ✅ Python 3.9+ instalado\n")
        return True
    else:
        print("   ❌ Python 3.9+ necessário\n")
        return False


def get_installed_packages():
    """Obtém lista de pacotes instalados"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        installed = {}
        for line in result.stdout.split('\n')[2:]:  # Pula cabeçalho
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    installed[parts[0].lower()] = parts[1]
        
        return installed
    except Exception as e:
        print(f"❌ Erro ao verificar pacotes: {e}")
        return {}


def check_package(package_name, required_version, installed_packages):
    """Verifica se um pacote específico está instalado"""
    package_key = package_name.lower()
    
    if package_key in installed_packages:
        installed_version = installed_packages[package_key]
        
        # Verifica versão (comparação simplificada)
        if installed_version == required_version:
            print(f"   ✅ {package_name:25s} {installed_version:15s} (correto)")
            return 'ok'
        else:
            print(f"   ⚠️  {package_name:25s} {installed_version:15s} (requer {required_version})")
            return 'outdated'
    else:
        print(f"   ❌ {package_name:25s} {'NÃO INSTALADO':15s}")
        return 'missing'


def main():
    print("=" * 70)
    print("🔍 VERIFICADOR DE DEPENDÊNCIAS - Body Fat System")
    print("=" * 70)
    print()
    
    # Verifica Python
    if not check_python_version():
        print("💡 Instale Python 3.9+ em: https://www.python.org/downloads/")
        return
    
    # Verifica pip
    print("📦 Verificando pip...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ✅ {result.stdout.strip()}\n")
    except Exception:
        print("   ❌ pip não encontrado\n")
        return
    
    # Obtém pacotes instalados
    print("📋 Verificando pacotes instalados...")
    installed = get_installed_packages()
    print()
    
    # Verifica cada pacote
    print("📊 STATUS DOS PACOTES:\n")
    
    results = {'ok': 0, 'outdated': 0, 'missing': 0}
    
    for package, version in REQUIRED_PACKAGES.items():
        status = check_package(package, version, installed)
        results[status] += 1
    
    # Resumo
    print("\n" + "=" * 70)
    print("📈 RESUMO:")
    print("=" * 70)
    print(f"✅ Instalados e corretos:  {results['ok']}")
    print(f"⚠️  Versão desatualizada:   {results['outdated']}")
    print(f"❌ Não instalados:         {results['missing']}")
    print()
    
    # Recomendações
    if results['missing'] > 0 or results['outdated'] > 0:
        print("💡 AÇÕES RECOMENDADAS:\n")
        
        if results['missing'] > 0:
            print("1️⃣  Instalar pacotes faltantes:")
            print("   pip install -r requirements.txt\n")
        
        if results['outdated'] > 0:
            print("2️⃣  Atualizar pacotes desatualizados:")
            print("   pip install --upgrade -r requirements.txt\n")
        
        print("3️⃣  OU instalar tudo do zero:")
        print("   pip install --force-reinstall -r requirements.txt\n")
    else:
        print("✅ TUDO PRONTO! Todas as dependências estão instaladas.\n")
        print("🚀 Próximos passos:")
        print("   1. python scripts/train_bf.py")
        print("   2. python scripts/evaluate_bf.py")
        print("   3. uvicorn main:app --reload\n")
    
    print("=" * 70)


if __name__ == "__main__":
    main()