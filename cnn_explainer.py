import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cnn_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transformare imagine pentru model
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def generate_gradcam(pil_image, model, region_id="0"):
    model.eval()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    rgb_image = pil_image.resize((224, 224), resample=Image.BICUBIC)
    rgb_image = np.array(rgb_image).astype(np.float32) / 255.0

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        confidence = confidence.item()
        predicted_class = predicted_class.item()

    target_layer = model.layer4[-1]
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True, image_weight=0.7)

    output_dir = "explanations/Grad-CAM++"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gradcam++_{region_id}.png")
    Image.fromarray(visualization).resize((1024, 1024), resample=Image.BICUBIC).save(output_path)

    return output_path, predicted_class, confidence

def disable_inplace_relu(model):
    for mod in model.modules():
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False

def load_background(background_paths):
    background_images = []
    for img_path in background_paths:
        img = Image.open(img_path).convert("RGB")
        background_images.append(img)
    return background_images

def create_explainer(model, background_images):
    disable_inplace_relu(model)
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad_(True)

    background_tensors = [transform(img) for img in background_images[:2]]
    background_batch = torch.stack(background_tensors).to(device)

    explainer = shap.GradientExplainer(model, background_batch)
    return explainer


def generate_shap(cnn_model, image_path, output_path="integrated_gradients_output.png", region_id=None):
    """
       Funcție redenumită istoric ca 'generate_shap', care implementează de fapt o metodă personalizată
       de explicabilitate vizuală bazată pe Integrated Gradients, pentru a evita limitările bibliotecii SHAP.
       """
    try:
        print(f"[INFO] Generating explanation for {region_id}...")
        print(f"[DEBUG] image_path type: {type(image_path)}")
        print(f"[DEBUG] image_path value: {image_path}")

        if not isinstance(image_path, (str, bytes, os.PathLike)):
            raise ValueError(f"Invalid image_path type: {type(image_path)}. Expected string path, got {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        pil_image = Image.open(image_path).convert('RGB')
        resized_pil = pil_image.resize((224, 224))
        image_np = np.array(resized_pil) / 255.0

        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        output = cnn_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

        print(f"[INFO] Prediction: Class {predicted_class}, Confidence: {confidence:.3f}")

        baseline = input_tensor * 0.2
        steps = 50
        integrated_grads = torch.zeros_like(input_tensor)

        for i in range(steps):
            alpha = i / steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)

            output = cnn_model(interpolated)
            target_class = predicted_class
            target_score = output[0, target_class]

            grad = torch.autograd.grad(outputs=target_score,
                                       inputs=interpolated,
                                       create_graph=False,
                                       retain_graph=False)[0]

            integrated_grads += grad / steps


        attributions = (input_tensor - baseline) * integrated_grads
        attr_np = attributions.squeeze().detach().cpu().numpy()
        if attr_np.shape[0] == 3:
            attr_np = attr_np.transpose(1, 2, 0)

        attr_sum = np.sum(np.abs(attr_np), axis=2)
        attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np)
        plt.imshow(attr_norm, alpha=0.7, cmap='magma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

        print(f"[INFO] Explanation saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"[ERROR] Explanation generation failed: {e}")
        import traceback
        traceback.print_exc()

        try:
            pil_image = Image.open(image_path).convert('RGB')
            resized_pil = pil_image.resize((224, 224))

            plt.figure(figsize=(6, 6))
            plt.imshow(np.array(resized_pil) / 255.0, interpolation='bilinear')

            plt.title(f'Explanation Failed\nRegion: {region_id}\nShowing original image')
            plt.axis('off')

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Fallback image saved to {output_path}")
            return output_path

        except Exception as fallback_error:
            print(f"[ERROR] Fallback also failed: {fallback_error}")
            return None


# Pentru testare locală, înlocuiește `image_path` cu o imagine crop reală și rulează:
# python cnn_explainer.py

if __name__ == "__main__":
    model_path = "model_cnn.pth"
    image_path = "cale_catre_imagine_cropata.jpg"
    background_paths = [
        "cale_catre_imagine_background1.jpg",
        "cale_catre_imagine_background2.jpg",
    ]
    region_id = "test_crop1"

    model = load_cnn_model(model_path)
    background_images = load_background(background_paths)
    image = Image.open(image_path).convert("RGB")

    gradcam_path, pred_class, conf = generate_gradcam(image, model, region_id)
    shap_path = generate_shap(model, image_path, output_path=f"explanations/IntegratedGradients_{region_id}.png", region_id=region_id)

    print(f"Clasa prezisă: {pred_class} | Încredere: {conf * 100:.2f}%")
    print(f"Grad-CAM++ salvat la: {gradcam_path}")
    print(f"Integrated Gradients salvat la: {shap_path}")