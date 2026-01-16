"""
Configurações centralizadas do FitPlanner Body Fat API.
Versão 2.1.0 - Com validação suave (Nível 2)
"""

class ValidationConfig:
    """
    Configurações para validação de fotos com DeepFace.
    
    NÍVEL 2 (Soft Validation):
    - Detecta divergências entre dados e foto
    - AVISA o usuário sobre inconsistências
    - NÃO bloqueia o processamento
    - Sempre usa dados informados pelo usuário
    """
    
    # Habilitar validação de fotos
    ENABLE_PHOTO_VALIDATION = True
    
    # Modo estrito (False = Nível 2, True = Nível 3)
    STRICT_MODE = False  # Nível 2: apenas avisa, não bloqueia
    
    # Confiança mínima para considerar a detecção válida (0-1)
    # Detecções abaixo deste valor são ignoradas
    MIN_CONFIDENCE = 0.6  # 60%
    
    # Margem de erro aceitável para idade (anos)
    MAX_AGE_DIFF = 5  # ±5 anos é considerado ok
    
    # Backend de detecção facial do DeepFace
    # Opções: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib'
    DETECTOR_BACKEND = 'opencv'  # opencv é o mais rápido
    
    # Ações de análise do DeepFace
    ACTIONS = ['age', 'gender']  # O que detectar
    
    # Forçar detecção de rosto
    # False = continua processamento mesmo sem detectar rosto
    ENFORCE_DETECTION = False
    
    # Idade mínima permitida (proteção de menores)
    MINIMUM_AGE = 16


class APIConfig:
    """Configurações gerais da API FastAPI"""
    
    # Informações da API
    TITLE = "FitPlanner Body Composition API"
    DESCRIPTION = "Estimativa educacional de composição corporal com validação de fotos"
    VERSION = "2.1.0"
    
    # Diretórios
    UPLOAD_DIR = "temp_images"
    RESULTS_DIR = "results"
    LOG_DIR = "logs"
    
    # Limites de arquivo
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Formatos de imagem aceitos
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
    
    # CORS
    ALLOW_ORIGINS = ["*"]  # Em produção, especifique domínios específicos
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]
    
    # Rate limiting (requisições por minuto por IP)
    RATE_LIMIT_PER_MINUTE = 10


class MLConfig:
    """Configurações do modelo de Machine Learning"""
    
    # Caminhos dos modelos
    MODEL_DIR = "models"
    BODY_FAT_MODEL_PATH = "models/bodyfat_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    
    # Features esperadas pelo modelo
    # (Deve corresponder às features usadas no treinamento)
    FEATURES = [
        'shoulder_width',
        'hip_width',
        'height_ratio',
        'waist_ratio',
        'volume_indicator',
        'waist_prominence',
        'age',
        'bmi',
        'sex'  # 0=female, 1=male
    ]
    
    # Fallback para regras se modelo não disponível
    USE_RULES_AS_FALLBACK = True


class ImageValidationConfig:
    """Configurações para validação de conteúdo de imagens (YOLO)"""
    
    # Modelo YOLO
    YOLO_MODEL = "yolov8n.pt"  # Modelo leve
    
    # Confiança mínima para detecção de pessoa
    PERSON_DETECTION_CONFIDENCE = 0.4
    
    # Permitir múltiplas pessoas na imagem
    ALLOW_MULTIPLE_PEOPLE = False
    
    # Classe do YOLO para pessoa
    PERSON_CLASS_NAME = "person"


class PhysiologicalLimits:
    """Limites fisiológicos para validação de dados"""
    
    # Idade
    MIN_AGE = 16
    MAX_AGE = 100
    
    # Peso (kg)
    MIN_WEIGHT = 30
    MAX_WEIGHT = 300
    
    # Altura (cm)
    MIN_HEIGHT = 100
    MAX_HEIGHT = 250
    
    # Body Fat Percentage (%)
    MIN_BODY_FAT_MALE = 5
    MAX_BODY_FAT_MALE = 45
    MIN_BODY_FAT_FEMALE = 12
    MAX_BODY_FAT_FEMALE = 50
    
    # IMC
    MIN_BMI = 12
    MAX_BMI = 50


class Messages:
    """Mensagens do sistema em PT-BR"""
    
    # Validação
    VALIDATION_SEX_MISMATCH = (
        "⚠️ A foto parece ser de uma pessoa do sexo {detected_sex}, "
        "mas você informou {informed_sex}. Confiança: {confidence}%"
    )
    
    VALIDATION_AGE_MISMATCH = (
        "⚠️ A idade aparente na foto é {detected_age} anos, "
        "mas você informou {informed_age} anos. Diferença: {diff} anos."
    )
    
    VALIDATION_LOW_CONFIDENCE = (
        "ℹ️ A detecção automática teve baixa confiança. "
        "Usando seus dados informados."
    )
    
    VALIDATION_NO_FACE = (
        "ℹ️ Não foi possível detectar o rosto na foto. "
        "Usando seus dados informados."
    )
    
    VALIDATION_CRITICAL_WARNING = (
        "⚠️ IMPORTANTE: Os dados informados podem não corresponder "
        "à pessoa na foto. Verifique se estão corretos."
    )
    
    # Erros
    ERROR_NO_PERSON = "Nenhuma pessoa detectada na imagem."
    ERROR_MULTIPLE_PEOPLE = "Múltiplas pessoas detectadas. Use foto com apenas uma pessoa."
    ERROR_UNDERAGE = "A foto enviada não parece ser de um adolescente ou adulto."
    ERROR_INVALID_IMAGE = "Imagem inválida ou corrompida."
    ERROR_INVALID_FORMAT = "Formato de arquivo não suportado. Use JPG, PNG ou WEBP."
    
    # Avisos
    WARNING_MODEL_UNAVAILABLE = "Modelo ML não disponível. Usando estimativa baseada em regras."
    WARNING_DEEPFACE_UNAVAILABLE = "Validação de foto desabilitada (DeepFace não disponível)."
    
    # Disclaimers
    DISCLAIMER_EDUCATIONAL = (
        "Estimativa educacional baseada em análise visual e machine learning. "
        "Não substitui avaliação clínica ou profissional."
    )
    
    DISCLAIMER_VALIDATION = (
        "A validação de foto é uma medida de segurança adicional. "
        "Sempre use seus dados reais para obter a melhor estimativa."
    )


class FeatureFlags:
    """Feature flags para habilitar/desabilitar funcionalidades"""
    
    # Validação de fotos
    ENABLE_PHOTO_VALIDATION = True
    
    # Modelo ML
    ENABLE_ML_MODEL = True
    
    # Validação de conteúdo (YOLO)
    ENABLE_CONTENT_VALIDATION = True
    
    # Bloqueio de menores
    ENABLE_AGE_RESTRICTION = True
    
    # Logging de resultados
    ENABLE_RESULT_LOGGING = True
    
    # Limpeza de arquivos temporários
    ENABLE_TEMP_FILE_CLEANUP = True


# ======================================================
# Funções Auxiliares
# ======================================================

def get_sex_label_ptbr(sex: str) -> str:
    """Retorna label em PT-BR para sexo"""
    return "masculino" if sex.lower() == "male" else "feminino"


def format_validation_message(
    message_template: str,
    **kwargs
) -> str:
    """Formata mensagem de validação com parâmetros"""
    try:
        return message_template.format(**kwargs)
    except KeyError:
        return message_template


def is_validation_enabled() -> bool:
    """Verifica se validação está habilitada"""
    return (
        FeatureFlags.ENABLE_PHOTO_VALIDATION and
        ValidationConfig.ENABLE_PHOTO_VALIDATION
    )


def get_api_info() -> dict:
    """Retorna informações da API para endpoint root"""
    return {
        "service": APIConfig.TITLE,
        "version": APIConfig.VERSION,
        "features": {
            "photo_validation": "enabled (Level 2 - Soft)" if is_validation_enabled() else "disabled",
            "ml_model": "enabled" if FeatureFlags.ENABLE_ML_MODEL else "disabled",
            "content_validation": "enabled" if FeatureFlags.ENABLE_CONTENT_VALIDATION else "disabled"
        },
        "limits": {
            "max_file_size_mb": APIConfig.MAX_FILE_SIZE_MB,
            "allowed_formats": list(APIConfig.ALLOWED_EXTENSIONS),
            "age_range": f"{PhysiologicalLimits.MIN_AGE}-{PhysiologicalLimits.MAX_AGE}",
            "weight_range": f"{PhysiologicalLimits.MIN_WEIGHT}-{PhysiologicalLimits.MAX_WEIGHT}kg",
            "height_range": f"{PhysiologicalLimits.MIN_HEIGHT}-{PhysiologicalLimits.MAX_HEIGHT}cm"
        }
    }


# ======================================================
# Configuração Padrão para Desenvolvimento
# ======================================================

class DevelopmentConfig:
    """Configurações para ambiente de desenvolvimento"""
    DEBUG = True
    VERBOSE_LOGGING = True
    DISABLE_RATE_LIMIT = True
    KEEP_TEMP_FILES = True  # Não deletar arquivos temporários


class ProductionConfig:
    """Configurações para ambiente de produção"""
    DEBUG = False
    VERBOSE_LOGGING = False
    DISABLE_RATE_LIMIT = False
    KEEP_TEMP_FILES = False
    
    # CORS restrito
    ALLOW_ORIGINS = [
        "https://fitplanner.app",
        "https://www.fitplanner.app"
    ]


# ======================================================
# Selecionar configuração baseada em variável de ambiente
# ======================================================

import os

ENV = os.getenv("ENVIRONMENT", "development").lower()

if ENV == "production":
    Config = ProductionConfig
else:
    Config = DevelopmentConfig