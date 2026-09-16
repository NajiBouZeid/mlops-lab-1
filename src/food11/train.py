"""
Trains a resnet18 classifier on the Food-11 dataset, logging params/metrics/
model to a local mlflow tracking server.

Run the mlflow server first, in its own terminal:
    uv run mlflow server --host 127.0.0.1 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

Then run training, e.g.:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
"""

import argparse

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_PATHS = {
    "processed": "data/food11_processed",
    "mini": "data/food11_processed_mini",
}

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "food11"


def get_dataloaders(dataset_name: str, batch_size: int):
    data_dir = DATASET_PATHS[dataset_name]

    # ImageNet normalization stats, since we start from an ImageNet-pretrained resnet18
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(f"{data_dir}/training", transform=transform)
    val_ds = datasets.ImageFolder(f"{data_dir}/validation", transform=transform)
    test_ds = datasets.ImageFolder(f"{data_dir}/evaluation", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, len(train_ds.classes)


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size
    )

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "device": str(device),
            }
        )

        for epoch in range(args.epochs):
            model.train()
            running_loss, seen = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                seen += images.size(0)

            train_loss = running_loss / seen
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- val_loss: {val_loss:.4f} "
                f"- val_accuracy: {val_accuracy:.4f}"
            )

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"Final test_accuracy: {test_accuracy:.4f}")

        example_images, _ = next(iter(train_loader))
        input_example = example_images[:1].cpu().numpy()
        mlflow.pytorch.log_model(model, "model", input_example=input_example)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a resnet18 on Food-11 with mlflow tracking")
    parser.add_argument("--dataset", choices=["processed", "mini"], default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
