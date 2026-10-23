"""Run ThriftNAS inference on a single image."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from thriftnas.models.genotypes import load_genotype
from thriftnas.models.retrain_net import build_network
from thriftnas.utils import get_device


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "results" / "retrain_t10" / "model_weights.pt"
GENOTYPE_PATH = ROOT / "results" / "search_lam0" / "genotype.json"
IMAGE_PATH = ROOT / "test_animals.jpg"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def main():
    for path in (WEIGHTS_PATH, GENOTYPE_PATH, IMAGE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    device = get_device()
    print(f"Device: {device}")

    # Reconstruct the same architecture used for retraining.
    genotype = load_genotype(str(GENOTYPE_PATH))
    model = build_network(
        C=36,
        num_classes=10,
        layers=8,
        genotype=genotype,
        drop_path_prob=0.0,
        is_thriftnas=True,
    ).to(device)

    # Load the weights-only state dictionary.
    state = torch.load(
        WEIGHTS_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    # Use evaluation preprocessing from the project.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    original = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = transform(original).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_idx = probabilities.max(dim=0)

    predicted_class = CLASSES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    print("\n--- ThriftNAS Prediction ---")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence:      {confidence_pct:.2f}%")
    print("\nAll class probabilities:")

    for i in probabilities.argsort(descending=True).tolist():
        print(f"  {CLASSES[i]:12s}: {probabilities[i].item() * 100:6.2f}%")

    # Display the original image and its prediction.
    plt.figure(figsize=(8, 6))
    plt.imshow(original)
    plt.title(
        f"ThriftNAS: {predicted_class} ({confidence_pct:.1f}%)"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()