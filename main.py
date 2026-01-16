from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles

from app.services.bf_estimator import estimate_body_composition
from app.utils.result_logger import log_estimate

# ======================================================
# 🔹 App
# ======================================================
app = FastAPI(
    title="FitPlanner Body Composition API",
    description="Estimativa educacional de composição corporal com validação de fotos (Nível 2)",
    version="2.1.0"
)

# ======================================================
# 🔹 CORS (Flutter / Web)
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 🔹 Diretório temporário
# ======================================================
UPLOAD_DIR = "temp_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================================================
# 🔹 Função utilitária: salvar imagem temporária
# ======================================================
async def save_temp_image(upload: UploadFile) -> str:
    """Salva imagem temporária e retorna o caminho"""
    ext = upload.filename.split(".")[-1] if "." in upload.filename else "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(path, "wb") as f:
        content = await upload.read()
        await f.write(content)

    return path

# ======================================================
# 🔹 Endpoints de health check
# ======================================================
@app.get("/")
async def root():
    """Endpoint de verificação da API"""
    return {
        "status": "online",
        "service": "FitPlanner Body Composition API",
        "version": "2.1.0",
        "features": {
            "photo_validation": "enabled (Level 2 - Soft)",
            "ml_model": "enabled",
            "deepface": "enabled"
        },
        "endpoints": {
            "docs": "/docs",
            "estimate": "/estimate-body-composition",
            "estimate_no_validation": "/estimate-body-composition-no-validation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint com info de validação"""
    try:
        from deepface import DeepFace
        deepface_status = "available"
    except ImportError:
        deepface_status = "not_available"
    
    return {
        "status": "healthy",
        "service": "body-fat-api",
        "validation": {
            "deepface": deepface_status,
            "level": "2 (soft - warns but doesn't block)"
        }
    }

# ======================================================
# 🔹 Endpoint principal COM validação (Nível 2)
# ======================================================
@app.post("/estimate-body-composition")
async def estimate_body(
    image: UploadFile = File(...),
    age: int = Form(...),
    weight_kg: float = Form(...),
    height_cm: float = Form(...),
    sex: str = Form(...),
    validate_photo: bool = Form(True)
):
    """
    Estima a composição corporal baseada em uma foto.
    """
    
    # Validações básicas
    sex = sex.lower()
    
    if sex not in ["male", "female"]:
        raise HTTPException(
            status_code=400,
            detail="Sexo deve ser 'male' ou 'female'."
        )
    
    if not 16 <= age <= 100:
        raise HTTPException(
            status_code=400,
            detail="Idade deve estar entre 16 e 100 anos."
        )
    
    if not 30 <= weight_kg <= 300:
        raise HTTPException(
            status_code=400,
            detail="Peso deve estar entre 30 e 300 kg."
        )
    
    if not 100 <= height_cm <= 250:
        raise HTTPException(
            status_code=400,
            detail="Altura deve estar entre 100 e 250 cm."
        )
    
    # 🔧 VALIDAÇÃO MELHORADA DE IMAGEM
    if not image.filename:
        raise HTTPException(
            status_code=400,
            detail="Nome do arquivo não fornecido."
        )
    
    # Aceitar tanto content_type quanto extensão do arquivo
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    valid_content_types = {'image/jpeg', 'image/png', 'image/webp', 'image/jpg'}
    
    file_ext = os.path.splitext(image.filename)[1].lower()
    content_type = image.content_type or ''
    
    # Validar por extensão OU content_type
    is_valid_extension = file_ext in valid_extensions
    is_valid_content_type = content_type.lower() in valid_content_types
    
    if not (is_valid_extension or is_valid_content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Formato de arquivo não suportado. Use JPG, PNG ou WEBP. "
                   f"(Recebido: ext={file_ext}, type={content_type})"
        )

    image_path = None

    try:
        # ===============================
        #  Salvar imagem temporária
        # ===============================
        image_path = await save_temp_image(image)
        
        print(f"📸 Imagem salva: {image_path}")
        print(f"📄 Arquivo: {image.filename} (type: {image.content_type})")
        print(f"👤 Dados: {age}y, {weight_kg}kg, {height_cm}cm, {sex}")
        print(f"🔍 Validação: {'Habilitada' if validate_photo else 'Desabilitada'}")

        # ===============================
        #  Estimar composição corporal COM validação
        # ===============================
        estimation = estimate_body_composition(
            image_path=image_path,
            age=age,
            weight_kg=weight_kg,
            height_cm=height_cm,
            sex=sex,
            use_ml=True,
            validate_photo=validate_photo
        )
        
        print(f"✅ Estimativa concluída")

        # ===============================
        #  Se houver erro da IA
        # ===============================
        if "error" in estimation:
            print(f"⚠️ Erro na estimativa: {estimation.get('error')}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": estimation.get("error"),
                    "reason": estimation.get("reason"),
                    "alerts": estimation.get("alerts", [])
                }
            )

        # ===============================
        #  Resposta final
        # ===============================
        response = {
            "body_fat_percentage": round(estimation["body_fat_percentage"], 1),
            "bmi": round(estimation.get("bmi", 0), 1),
            "body_type": estimation.get("body_type"),
            "estimation_method": estimation.get("estimation_method", "ML Model"),
            "ratios": estimation.get("ratios"),
            "measurements": estimation.get("measurements"),
            "person_detected": estimation.get("person_detected", False),
            "alerts": estimation.get("alerts", []),
            "visual_info": estimation.get("visual_info", {}),
            "validation": estimation.get("validation", {}),
            "validation_alert": estimation.get("validation_alert"),
            "disclaimer": estimation.get(
                "disclaimer", 
                "Estimativa educacional. Não substitui avaliação profissional."
            )
        }
        
        print(f"✅ Body Fat: {response['body_fat_percentage']}%")
        
        if response.get("alerts"):
            print(f"⚠️ Avisos: {len(response['alerts'])}")
            for alert in response["alerts"]:
                print(f"   - {alert}")

        # ===============================
        #  Log
        # ===============================
        log_estimate({
            "age": age,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "sex": sex,
            "body_fat_percentage": response["body_fat_percentage"],
            "bmi": response["bmi"],
            "body_type": response.get("body_type"),
            "ratios": response.get("ratios"),
            "alerts": response.get("alerts"),
            "visual_info": response.get("visual_info"),
            "validation": response.get("validation"),
            "person_detected": response["person_detected"]
        })

        return response

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Erro interno: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar imagem: {str(e)}"
        )

    finally:
        # ===============================
        #  Limpeza do arquivo temporário
        # ===============================
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"🗑️ Arquivo temporário removido: {image_path}")
            except Exception as e:
                print(f"⚠️ Erro ao remover arquivo: {e}")


# ======================================================
# 🔹 Endpoint alternativo SEM validação
# ======================================================
@app.post("/estimate-body-composition-no-validation")
async def estimate_body_no_validation(
    image: UploadFile = File(...),
    age: int = Form(...),
    weight_kg: float = Form(...),
    height_cm: float = Form(...),
    sex: str = Form(...)
):
    """
    Endpoint SEM validação de foto (comportamento original).
    """
    return await estimate_body(
        image=image,
        age=age,
        weight_kg=weight_kg,
        height_cm=height_cm,
        sex=sex,
        validate_photo=False
    )