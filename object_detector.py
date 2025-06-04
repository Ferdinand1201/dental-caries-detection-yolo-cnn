from ultralytics import YOLO
from flask import request, Flask, jsonify, send_from_directory, make_response
from torchvision import transforms
import torch
from PIL import Image
import os
import uuid
from cnn_explainer import generate_gradcam, generate_shap, load_cnn_model, load_background, create_explainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

print("Loading YOLO model...")
yolo_model = YOLO("best.pt")

print("Loading CNN model...")
cnn_model = load_cnn_model("model_cnn.pth")

# Încarcă primele 2 imagini "fara_carie" pentru background SHAP
fara_carie_dir = os.path.join(BASE_DIR, "dataset/train/fara_carie")
background_paths = [
    os.path.join(fara_carie_dir, fname)
    for fname in sorted(os.listdir(fara_carie_dir))
    if fname.lower().endswith((".jpg", ".png", ".jpeg"))
][:2]

background_images = [Image.open(p).convert("RGB") for p in background_paths]

shap_explainer = create_explainer(cnn_model, background_images)

# Foldere pentru crop-uri și explicații
CROPS_DIR = "crops"
EXPLANATIONS_DIR = "explanations"
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)


def crop_yolo_detections(image, detections, output_dir=CROPS_DIR):
    img_width, img_height = image.size
    pad = 30
    cropped_paths = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2, class_name = det
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_width, x2 + pad)
        y2 = min(img_height, y2 + pad)

        crop = image.crop((x1, y1, x2, y2))
        unique_id = uuid.uuid4().hex[:8]
        crop_filename = f"{class_name}_{i}_{unique_id}.png"
        crop_path = os.path.join(output_dir, crop_filename)
        crop.save(crop_path, format='PNG', optimize=False)
        cropped_paths.append((crop_path, i))

    return cropped_paths


def detect_objects_on_image(image):
    orig_width, orig_height = image.size
    resized_image = image.resize((640, 640))
    transform = transforms.ToTensor()
    img_tensor = transform(resized_image).unsqueeze(0)

    results = yolo_model.predict(img_tensor, verbose=False)
    result = results[0]

    scale_x = orig_width / 640
    scale_y = orig_height / 640

    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [x.item() for x in box.xyxy[0]]
        class_id = int(box.cls[0].item())
        if class_id != 0:  # filtrăm doar clasa 'carie' (presupun că este clasa 0)
            continue
        # Opțional: poți salva și probabilitatea dacă dorești
        x1 = round(x1 * scale_x)
        y1 = round(y1 * scale_y)
        x2 = round(x2 * scale_x)
        y2 = round(y2 * scale_y)

        output.append([x1, y1, x2, y2, result.names[class_id]])

    return output


@app.route("/")
def root():
    with open("index.html", encoding="utf-8") as file:
        content = file.read()
    response = make_response(content)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


class_names = {0: "carie", 1: "non-carie"}


@app.route("/detect", methods=["POST"])
def detect():
    if "image_file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    buf = request.files["image_file"]
    image = Image.open(buf.stream).convert("RGB")
    # Salvezi imaginea uploadată (opțional)
    image.save("uploaded.png")

    detections = detect_objects_on_image(image)
    cropped_paths = crop_yolo_detections(image, detections)

    cropped_image_urls = []
    gradcam_image_urls = []
    shap_image_urls = []
    cnn_predictions = []
    cnn_confidences = []

    # Pre-încarcă toate crop-urile pentru a evita I/O repetat
    crop_images = [(Image.open(crop_path).convert("RGB"), crop_path, idx) for crop_path, idx in cropped_paths]

    for crop_image, crop_path, idx in crop_images:
        cropped_url = f"/crops/{os.path.basename(crop_path)}"
        cropped_image_urls.append(cropped_url)

        region_id = os.path.splitext(os.path.basename(crop_path))[0]

        # Grad-CAM (rapid)
        explanation_path, predicted_class, confidence = generate_gradcam(crop_image, cnn_model, region_id=region_id)
        explanation_url = f"/explanations/Grad-CAM/{os.path.basename(explanation_path)}"
        gradcam_image_urls.append(explanation_url)

        # SHAP (mai lent, poate eșua uneori)
        try:
            shap_path = generate_shap(cnn_model, crop_path, output_path=f"explanations/SHAP/shap_{region_id}.png",region_id=region_id)
            shap_url = f"/explanations/SHAP/{os.path.basename(shap_path)}"
        except Exception as e:
            print(f"SHAP failed for {region_id}: {e}")
            shap_url = None

        shap_image_urls.append(shap_url)
        cnn_predictions.append(class_names.get(predicted_class, "necunoscut"))
        cnn_confidences.append(round(confidence * 100, 2))

    return jsonify({
        "detections": detections,
        "cropped_images": cropped_image_urls,
        "gradcam_images": gradcam_image_urls,
        "shap_images": shap_image_urls,
        "cnn_predictions": cnn_predictions,
        "cnn_confidences": cnn_confidences
    })


@app.route('/crops/<path:filename>')
def serve_crop(filename):
    return send_from_directory(CROPS_DIR, filename)


@app.route('/explanations/<path:filename>')
def serve_explanation(filename):
    full_path = os.path.join(EXPLANATIONS_DIR, filename)
    folder = os.path.dirname(full_path)
    file = os.path.basename(full_path)
    return send_from_directory(folder, file)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080, threads=2)
