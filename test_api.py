import os
from app.services.bf_estimator import estimate_body_composition

# Caminho das imagens
IMAGE_DIR = "test_images"  # coloque suas imagens aqui
os.makedirs(IMAGE_DIR, exist_ok=True)

# Testes
tests = [
    {"file": "magro.jpg", "age": 25, "weight_kg": 60, "height_cm": 175, "sex": "male", "desc": "Corpo magro"},
    {"file": "forte.jpg", "age": 25, "weight_kg": 80, "height_cm": 175, "sex": "male", "desc": "Corpo forte"},
    {"file": "gordo.jpg", "age": 25, "weight_kg": 95, "height_cm": 175, "sex": "male", "desc": "Corpo gordo"},
    {"file": "mulher_jovem.jpg", "age": 22, "weight_kg": 55, "height_cm": 165, "sex": "female", "desc": "Mulher jovem"},
]

for test in tests:
    image_path = os.path.join(IMAGE_DIR, test["file"])
    if not os.path.exists(image_path):
        print(f"⚠️  Imagem não encontrada: {image_path}")
        continue

    print(f"\n=== Teste: {test['desc']} ===")
    result = estimate_body_composition(
        image_path=image_path,
        age=test["age"],
        weight_kg=test["weight_kg"],
        height_cm=test["height_cm"],
        sex=test["sex"]
    )
    print(result)