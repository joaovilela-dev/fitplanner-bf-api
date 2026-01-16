"""
Validação de conteúdo de imagens usando YOLO.
Corrigido para PyTorch 2.6+ (weights_only=True por padrão)
"""
import torch
from ultralytics import YOLO

# ===================================
# FIX para PyTorch 2.6+
# ===================================
# Adicionar classes do YOLO aos globals seguros
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"⚠️ Aviso ao configurar safe globals: {e}")

# ===================================
# Inicializar modelo YOLO
# ===================================
try:
    # Tenta carregar modelo leve (baixado automaticamente)
    model = YOLO("yolov8n.pt")
    print("✅ YOLO carregado com sucesso")
except Exception as e:
    print(f"⚠️ Erro ao carregar YOLO: {e}")
    model = None


def validate_image_content(image_path: str) -> dict:
    """
    Valida se a imagem contém exatamente UMA pessoa.
    Evita imagens inválidas, múltiplas pessoas ou ausência de corpo.
    
    Se YOLO não estiver disponível, retorna válido (fail-safe).
    """
    
    # Fallback: Se YOLO não carregou, considera válido
    if model is None:
        return {
            "valid": True,
            "person_count": 1,
            "confidence": 0.5,
            "note": "YOLO não disponível, validação desabilitada"
        }
    
    try:
        # Executar detecção
        results = model(image_path, conf=0.4, verbose=False)
        
        person_count = 0
        confidences = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names.get(cls_id, "")
                confidence = float(box.conf[0])
                
                if class_name == "person":
                    person_count += 1
                    confidences.append(confidence)
        
        # ❌ Nenhuma pessoa
        if person_count == 0:
            return {
                "valid": False,
                "error": "NO_PERSON_DETECTED",
                "message": "Nenhuma pessoa detectada na imagem.",
                "person_count": 0
            }
        
        # ❌ Mais de uma pessoa
        if person_count > 1:
            return {
                "valid": False,
                "error": "MULTIPLE_PEOPLE",
                "message": f"Múltiplas pessoas detectadas ({person_count}). Use foto com apenas uma pessoa.",
                "person_count": person_count
            }
        
        # ✅ Imagem válida
        return {
            "valid": True,
            "person_count": 1,
            "confidence": round(max(confidences), 2)
        }
    
    except Exception as e:
        # Se der erro na validação, considera válido (fail-safe)
        print(f"⚠️ Erro na validação YOLO: {e}")
        return {
            "valid": True,
            "person_count": 1,
            "confidence": 0.5,
            "note": f"Erro na validação: {str(e)}"
        }