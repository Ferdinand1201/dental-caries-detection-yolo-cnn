"""
object_detector.py – Backend Flask pentru sistemul de detecție și clasificare a cariilor dentare

Funcționalitate:
- Detecție automată a cariilor folosind YOLOv8
- Clasificare carie / non-carie folosind CNN
- Generare explicații vizuale (Grad-CAM++, Integrated Gradients)
"""

from ultralytics import YOLO
from flask import request, Flask, jsonify, send_from_directory, make_response
from torchvision import transforms
from PIL import Image
from waitress import serve
import os
import uuid
import logging
from cnn_explainer import generate_gradcam, generate_integrated_gradients, load_cnn_model, load_background, create_explainer

# Directorul de bază al aplicației
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inițializare aplicație Flask
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Încărcarea modelului YOLOv8 pentru detecția cariilor
logging.info("Se încarcă modelul YOLOv8...")
yolo_model = YOLO("best.pt")

# Încărcarea modelului CNN antrenat pentru clasificare carie/non-carie
logging.info("Se încarcă modelul CNN pentru clasificare...")
cnn_model = load_cnn_model("model_cnn.pth")

# Selectarea imaginilor de fundal pentru explicabilitate cu Integrated Gradients
fara_carie_dir = os.path.join(BASE_DIR, "dataset/train/fara_carie")
background_paths = [
    os.path.join(fara_carie_dir, fname)
    for fname in sorted(os.listdir(fara_carie_dir))
    if fname.lower().endswith((".jpg", ".png", ".jpeg"))
][:2]

background_images = [Image.open(p).convert("RGB") for p in background_paths]
shap_explainer = create_explainer(cnn_model, background_images)

# Creearea directoarelor în care se vor salva imaginile decupate (YOLO) și explicațiile vizuale (Grad-CAM++, Integrated Gradients).
# Acestea sunt păstrate pentru afișare și analiză ulterioară.
CROPS_DIR = "crops"
EXPLANATIONS_DIR = "explanations"
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)

def detect_objects_on_image(image):
    """
    Redimensionează imaginea, aplică YOLOv8 și extrage coordonatele predicțiilor pentru carii.
    """
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
        if class_id != 0:
            continue
        x1 = round(x1 * scale_x)
        y1 = round(y1 * scale_y)
        x2 = round(x2 * scale_x)
        y2 = round(y2 * scale_y)
        output.append([x1, y1, x2, y2, result.names[class_id]])

    return output

def crop_yolo_detections(image, detections, output_dir=CROPS_DIR):
    """
    Decupează fiecare regiune detectată și salvează două versiuni: 224x224 și 1024x1024.
    """
    img_width, img_height = image.size
    cropped_paths = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2, class_name = det

        # Adăugare padding pentru a păstra contextul vizual
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_width, x2 + pad_x)
        y2 = min(img_height, y2 + pad_y)

        crop = image.crop((x1, y1, x2, y2))
        unique_id = uuid.uuid4().hex[:8]
        crop_filename = f"{class_name}_{i}_{unique_id}.png"

        # Versiune mărită (pentru afișare în interfață)
        full_crop_resized = crop.resize((1024,1024), Image.BICUBIC)
        full_crop_filename = f"full_{crop_filename}"
        full_crop_path = os.path.join(output_dir, full_crop_filename)
        full_crop_resized.save(full_crop_path, format='PNG', optimize=False)

        # Versiune redimensionată pentru clasificare CNN
        resized_crop = crop.resize((224, 224), Image.BICUBIC)
        crop_path = os.path.join(output_dir, crop_filename)
        resized_crop.save(crop_path, format='PNG', optimize=False)

        cropped_paths.append((crop_path, full_crop_path, i))

    return cropped_paths

def sort_detections_by_grid(detections, row_threshold=50):
    """
    Sortează regiunile detectate în funcție de poziția lor verticală și orizontală în imagine.
    """
    if not detections:
        return []

    detections = [list(det) for det in detections]
    detections_with_center_y = [(det, (det[1] + det[3]) / 2) for det in detections]
    detections_with_center_y.sort(key=lambda x: x[1])

    sorted_groups = []
    current_group = [detections_with_center_y[0][0]]
    current_y = detections_with_center_y[0][1]

    for det, y_center in detections_with_center_y[1:]:
        if abs(y_center - current_y) < row_threshold:
            current_group.append(det)
        else:
            current_group.sort(key=lambda d: d[0])
            sorted_groups.extend(current_group)
            current_group = [det]
            current_y = y_center

    current_group.sort(key=lambda d: d[0])
    sorted_groups.extend(current_group)

    return sorted_groups

@app.route("/")
def root():
    """Răspunde cu pagina principală HTML"""

    with open("index.html", encoding="utf-8") as file:
        content = file.read()
    response = make_response(content)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response

# Maparea etichetelor numerice la etichete semantice
class_names = {0: "carie", 1: "non-carie"}

@app.route("/detect", methods=["POST"])
def detect():
    """
    Endpoint principal care primește o imagine, aplică detecție, clasificare și generare explicații.
    """
    if "image_file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    buf = request.files["image_file"]
    image = Image.open(buf.stream).convert("RGB")

    detections = detect_objects_on_image(image)
    detections = sort_detections_by_grid(detections)

    cropped_paths = crop_yolo_detections(image, detections)
    cropped_image_urls = []
    gradcam_image_urls = []
    ig_image_urls = []
    cnn_predictions = []
    cnn_confidences = []
    cropped_full_urls = []

    crop_images = [
        (Image.open(crop_path).convert("RGB"), crop_path, full_crop_path, idx)
        for crop_path, full_crop_path, idx in cropped_paths
    ]

    for crop_image, crop_path, full_crop_path, idx in crop_images:
        cropped_url = f"/crops/{os.path.basename(crop_path)}"
        cropped_image_urls.append(cropped_url)
        cropped_full_url = f"/crops/{os.path.basename(full_crop_path)}"
        cropped_full_urls.append(cropped_full_url)

        region_id = os.path.splitext(os.path.basename(crop_path))[0]

        # Explicație vizuală cu Grad-CAM++
        explanation_path, predicted_class, confidence = generate_gradcam(crop_image, cnn_model, region_id=region_id)
        explanation_url = f"/explanations/Grad-CAM++/{os.path.basename(explanation_path)}"
        gradcam_image_urls.append(explanation_url)

        # Explicație Integrated Gradients
        try:
            ig_path = generate_integrated_gradients(
                cnn_model,
                crop_path,
                output_path=f"explanations/IntegratedGradients/integrated_gradients_{region_id}.png",
                region_id=region_id
            )
            ig_url = f"/explanations/IntegratedGradients/{os.path.basename(ig_path)}"
        except Exception as e:
            print(f"Integrated Gradients failed for {region_id}: {e}")
            ig_url = None

        ig_image_urls.append(ig_url)
        cnn_predictions.append(class_names.get(predicted_class, "necunoscut"))
        cnn_confidences.append(round(confidence * 100, 2))

    return jsonify({
        "detections": detections,
        "cropped_images": cropped_image_urls,
        "cropped_full_images": cropped_full_urls,
        "gradcam_images": gradcam_image_urls,
        "shap_images": ig_image_urls,
        "cnn_predictions": cnn_predictions,
        "cnn_confidences": cnn_confidences
    })

@app.route('/crops/<path:filename>')
def serve_crop(filename):
    """Servește imaginile decupate pentru afișare în interfață"""

    return send_from_directory(CROPS_DIR, filename)

@app.route('/explanations/<path:filename>')
def serve_explanation(filename):
    """Servește hărțile explicative generate"""

    full_path = os.path.join(EXPLANATIONS_DIR, filename)
    folder = os.path.dirname(full_path)
    file = os.path.basename(full_path)
    return send_from_directory(folder, file)

if __name__ == "__main__":
    # Pornirea serverului folosind waitress
    serve(app, host="0.0.0.0", port=8080, threads=2)
