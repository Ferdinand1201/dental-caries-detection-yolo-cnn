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

fara_carie_dir = os.path.join(BASE_DIR, "dataset/train/fara_carie")
background_paths = [
    os.path.join(fara_carie_dir, fname)
    for fname in sorted(os.listdir(fara_carie_dir))
    if fname.lower().endswith((".jpg", ".png", ".jpeg"))
][:2]

background_images = [Image.open(p).convert("RGB") for p in background_paths]
shap_explainer = create_explainer(cnn_model, background_images)
CROPS_DIR = "crops"
EXPLANATIONS_DIR = "explanations"
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)

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


def crop_yolo_detections(image, detections, output_dir=CROPS_DIR):
    img_width, img_height = image.size
    cropped_paths = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2, class_name = det

        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_width, x2 + pad_x)
        y2 = min(img_height, y2 + pad_y)

        crop = image.crop((x1, y1, x2, y2))
        unique_id = uuid.uuid4().hex[:8]
        crop_filename = f"{class_name}_{i}_{unique_id}.png"

        full_crop_resized = crop.resize((1024, 1024), Image.BICUBIC)
        full_crop_filename = f"full_{crop_filename}"
        full_crop_path = os.path.join(output_dir, full_crop_filename)
        full_crop_resized.save(full_crop_path, format='PNG', optimize=False)

        resized_crop = crop.resize((224, 224), Image.BICUBIC)
        crop_path = os.path.join(output_dir, crop_filename)
        resized_crop.save(crop_path, format='PNG', optimize=False)

        cropped_paths.append((crop_path, full_crop_path, i))

    return cropped_paths


def sort_detections_by_grid(detections, row_threshold=50):

    if not detections:
        return []

    detections = [list(det) for det in detections]
    detections_with_center_y = [(det, (det[1] + det[3]) / 2) for det in detections]  # y_center
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


    detections = detect_objects_on_image(image)
    detections = sort_detections_by_grid(detections)

    cropped_paths = crop_yolo_detections(image, detections)
    cropped_image_urls = []
    gradcam_image_urls = []
    shap_image_urls = []
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


        explanation_path, predicted_class, confidence = generate_gradcam(crop_image, cnn_model, region_id=region_id)
        explanation_url = f"/explanations/Grad-CAM++/{os.path.basename(explanation_path)}"
        gradcam_image_urls.append(explanation_url)


        try:
            shap_path = generate_shap(cnn_model, crop_path, output_path=f"explanations/IntegratedGradients/integrated_gradients_{region_id}.png",region_id=region_id)
            shap_url = f"/explanations/IntegratedGradients/{os.path.basename(shap_path)}"
        except Exception as e:
            print(f"Integrated Gradients failed for {region_id}: {e}")
            shap_url = None

        shap_image_urls.append(shap_url)
        cnn_predictions.append(class_names.get(predicted_class, "necunoscut"))
        cnn_confidences.append(round(confidence * 100, 2))


    return jsonify({
        "detections": detections,
        "cropped_images": cropped_image_urls,
        "cropped_full_images": cropped_full_urls,
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
