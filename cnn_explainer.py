import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import shap
import copy
import io
import base64
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import shap

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


#  Funcție pentru încărcarea modelului antrenat
def load_cnn_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model


# === Transformare imagine ===
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# === Grad-CAM ===
def generate_gradcam(pil_image, model, region_id="0"):
    input_tensor = transform(pil_image).unsqueeze(0)

    rgb_image = pil_image.resize((224, 224), resample=Image.BICUBIC)
    rgb_image = np.array(rgb_image) / 255.0
    rgb_image = rgb_image.astype(np.float32)

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    confidence = confidence.item()
    predicted_class = predicted_class.item()

    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True, image_weight=0.5)

    output_path = f"explanations/gradcam_{region_id}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(visualization).save(output_path)

    return output_path, predicted_class, confidence



# === Transformare care include normalizare ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def disable_inplace_relu(model):
    for mod in model.modules():
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False

def load_background(background_paths):
    background_images = []
    for img_path in background_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        background_images.append(img_tensor)
    background = torch.stack(background_images)
    return background

def create_explainer(model, background_images):
    model_copy = copy.deepcopy(model)
    disable_inplace_relu(model_copy)
    model_copy.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Conversie imagini PIL -> tensori, apoi batch
    background_tensors = [preprocess(img) for img in background_images]
    background_batch = torch.stack(background_tensors)  # [N, C, H, W]

    explainer = shap.GradientExplainer(model_copy, background_batch)
    return explainer

def generate_shap(pil_image, explainer, region_id="0"):
    image_tensor = transform(pil_image).unsqueeze(0)
    shap_values = explainer.shap_values(image_tensor)

    shap_values_class = shap_values[0]  # (1, 3, 224, 224)
    shap_values_for_plot = np.transpose(shap_values_class[0], (1, 2, 0))  # (224,224,3)

    original_np = np.array(pil_image.resize((224, 224))) / 255.0

    output_path = f"explanations/shap_{region_id}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    blank_bg = np.ones_like(original_np)
    shap.image_plot(shap_values_for_plot[np.newaxis, ...], blank_bg[np.newaxis, ...], show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path

if __name__ == "__main__":
    model_path = "carie_classifier_resnet18_best.pth"
    image_path = "cale_catre_imagine_cropata.jpg"
    background_paths = [
        "cale_catre_imagine_background1.jpg",
        "cale_catre_imagine_background2.jpg",
        # ... alte imagini background
    ]
    region_id = "test_crop1"

    model = load_cnn_model(model_path)
    background = load_background(background_paths)
    explainer = create_explainer(model, background)

    image = Image.open(image_path).convert("RGB")

    # Presupun că ai funcția generate_gradcam definită undeva
    gradcam_path, pred_class, conf = generate_gradcam(image, model, region_id)

    shap_path = generate_shap(image, explainer, region_id)

    print(f"Clasa prezisă: {pred_class} | Încredere: {conf:.2f}")
    print(f"Grad-CAM salvat la: {gradcam_path}")
    print(f"SHAP salvat la: {shap_path}")

