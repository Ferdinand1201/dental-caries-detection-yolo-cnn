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
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

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

    output_dir = "explanations/Grad-CAM"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gradcam_{region_id}.png")
    highres_image = Image.fromarray(visualization).resize((1024, 1024), resample=Image.BICUBIC)
    highres_image.save(output_path)

    return output_path, predicted_class, confidence


def disable_inplace_relu(model):
    """Dezactivează ReLU inplace pentru compatibilitate cu SHAP"""
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
    """Creează SHAP GradientExplainer"""
    disable_inplace_relu(model)
    model.eval()
    model.to(device)

    # Gradienți activați pentru SHAP
    for param in model.parameters():
        param.requires_grad_(True)

    # Preprocess background (aplică transformările definite deja)
    background_tensors = [transform(img) for img in background_images[:2]]
    background_batch = torch.stack(background_tensors).to(device)

    explainer = shap.GradientExplainer(model, background_batch)
    return explainer


def generate_shap(cnn_model, image_path, output_path="shap_output.png", region_id=None):
    """
    Generate SHAP-like explanations using integrated gradients approach
    This avoids SHAP library compatibility issues
    """
    import matplotlib.pyplot as plt

    try:
        print(f"[INFO] Generating explanation for {region_id}...")
        print(f"[DEBUG] image_path type: {type(image_path)}")
        print(f"[DEBUG] image_path value: {image_path}")

        # Check if image_path is valid
        if not isinstance(image_path, (str, bytes, os.PathLike)):
            raise ValueError(f"Invalid image_path type: {type(image_path)}. Expected string path, got {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and preprocess image
        pil_image = Image.open(image_path).convert('RGB')
        resized_pil = pil_image.resize((224, 224))
        image_np = np.array(resized_pil) / 255.0

        # Prepare input tensor
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Get model prediction
        output = cnn_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

        print(f"[INFO] Prediction: Class {predicted_class}, Confidence: {confidence:.3f}")

        # Generate integrated gradients manually
        baseline = torch.zeros_like(input_tensor)
        steps = 50
        integrated_grads = torch.zeros_like(input_tensor)

        for i in range(steps):
            alpha = i / steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)

            output = cnn_model(interpolated)
            target_score = output[0, predicted_class]

            grad = torch.autograd.grad(outputs=target_score,
                                       inputs=interpolated,
                                       create_graph=False,
                                       retain_graph=False)[0]

            integrated_grads += grad / steps

        # Calculate attributions
        attributions = (input_tensor - baseline) * integrated_grads

        # Convert to numpy and process
        attr_np = attributions.squeeze().detach().cpu().numpy()

        # Transpose from (C, H, W) to (H, W, C)
        if attr_np.shape[0] == 3:
            attr_np = attr_np.transpose(1, 2, 0)

        # Create attribution map
        attr_sum = np.sum(np.abs(attr_np), axis=2)
        attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)

        # Create single visualization - just the overlay
        plt.figure(figsize=(8, 8))

        # Afișare imagine originală cu overlay SHAP în nuanțe de verde
        plt.imshow(image_np)
        plt.imshow(attr_norm, alpha=0.6, cmap='magma')

        # Fără titlu, fără axe
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

        # Simple fallback - just show the original image
        try:
            pil_image = Image.open(image_path).convert('RGB')
            resized_pil = pil_image.resize((224, 224))

            plt.figure(figsize=(6, 6))
            plt.imshow(np.array(resized_pil) / 255.0, interpolation='bilinear')

            plt.title(f'Explanation Failed\nRegion: {region_id}\nShowing original image')
            plt.axis('off')

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Fallback image saved to {output_path}")
            return output_path

        except Exception as fallback_error:
            print(f"[ERROR] Fallback also failed: {fallback_error}")
            return None

if __name__ == "__main__":
    model_path = "carie_classifier_resnet18_best.pth"
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
    shap_path = generate_shap(model, image_path, output_path=f"explanations/shap_{region_id}.png", region_id=region_id)

    print(f"Clasa prezisă: {pred_class} | Încredere: {conf * 100:.2f}%")
    print(f"Grad-CAM salvat la: {gradcam_path}")
    print(f"SHAP salvat la: {shap_path}")