import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm


class Model(nn.Module):
    # TODO: Implement your own model
    def __init__(self, num_classes=2):
        super(Model, self).__init__()

        # Use ResNet-18 as the backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1-channel input
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )

        # Freeze the early layers and fine-tune the later layers
        for name, param in self.backbone.named_parameters():
            if (
                name.startswith("layer1")
                or name.startswith("bn1")
                or name.startswith("conv1")
            ):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Replace the final fully connected layer to match the number of classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(  # type: ignore
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class CentralizedLearning:
    def __init__(
        self, batch_size, learning_rate, num_epochs, device, class_weights=None
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.class_weights = class_weights

    def calculate_f1_score(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    def train(
        self,
        fold_model,
        train_dataloader,
        k_folds=3,
        class_names=None,
        test_dataloader=None,
    ):
        # TODO: Implement the training process and cross validation.
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device)
            if self.class_weights is not None
            else None
        )

        dataset = train_dataloader.dataset
        kfold = KFold(n_splits=k_folds, shuffle=True)

        val_losses = []
        best_epochs = []

        # Phase 1: Train with K-Fold Cross Validation to find the optimal number of epochs
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold + 1}/{k_folds}")

            train_sub_dataset = torch.utils.data.Subset(dataset, train_ids)  # type: ignore
            val_sub_dataset = torch.utils.data.Subset(dataset, val_ids)  # type: ignore
            train_loader = DataLoader(
                train_sub_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_sub_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            fold_model = Model(num_classes=2).to(self.device)
            optimizer = optim.AdamW(
                fold_model.parameters(), lr=self.learning_rate, weight_decay=1e-4
            )

            best_val_loss = float("inf")
            best_epoch = 0
            patience = 5
            wait = 0

            for epoch in range(self.num_epochs):
                fold_model.train()

                # Set BatchNorm layers to eval mode to prevent them from updating their running statistics
                for name, module in fold_model.backbone.named_modules():
                    if (
                        name.startswith("bn1") or name.startswith("layer1")
                    ) and isinstance(module, nn.BatchNorm2d):
                        module.eval()

                batch_loop = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                    leave=False,
                    unit="batch",
                )
                for data, target in batch_loop:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = fold_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                val_loss, val_acc, val_f1, _, _ = self.evaluate(
                    fold_model, val_loader, class_names
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    torch.save(
                        fold_model.state_dict(),
                        f"outputs/cl/checkpoints/best_model_fold_{fold + 1}.pt",
                    )
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        fold_model.load_state_dict(
                            torch.load(f"outputs/cl/checkpoints/best_model_fold_{fold + 1}.pt")
                        )
                        break

            val_losses.append(best_val_loss)
            best_epochs.append(best_epoch)

        # Phase 2: Use full training data to train the final model with the optimal number of epochs
        optimal_epochs = int(sum(best_epochs) / len(best_epochs))

        final_model = Model(num_classes=2).to(self.device)
        optimizer = optim.AdamW(
            final_model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        for epoch in range(optimal_epochs):
            final_model.train()

            # Set BatchNorm layers to eval mode to prevent them from updating their running statistics
            for name, module in final_model.backbone.named_modules():
                if (name.startswith("bn1") or name.startswith("layer1")) and isinstance(
                    module, nn.BatchNorm2d
                ):
                    module.eval()

            batch_loop = tqdm(
                train_dataloader,
                desc=f"Final Model Epoch {epoch + 1}/{optimal_epochs}",
                leave=False,
                unit="batch",
            )
            for data, target in batch_loop:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = final_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            torch.save(final_model.state_dict(), "outputs/cl/results/cl_model.pt")
            self.evaluate(final_model, train_dataloader, class_names)

        # Evaluate the final model on the test set
        print("\nEvaluating final model on the test set...")
        self.evaluate(final_model, test_dataloader, class_names)
        return

    def evaluate(self, model, test_loader, class_names=None):
        model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()

        all_true_labels = []
        all_pred_labels = []
        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                _, predicted = torch.max(output, 1)

                total_loss += loss.item() * data.size(0)
                total += target.size(0)
                all_true_labels.extend(target.cpu().numpy())
                all_pred_labels.extend(predicted.cpu().numpy())

        report = self.calculate_f1_score(all_true_labels, all_pred_labels)
        f1 = report["weighted avg"]["f1-score"]  # type: ignore
        precision = report["weighted avg"]["precision"]  # type: ignore
        recall = report["weighted avg"]["recall"]  # type: ignore
        acc = sum(int(a == b) for a, b in zip(all_true_labels, all_pred_labels)) / len(
            all_true_labels
        )
        test_loss = total_loss / total
        print(
            f"Test Loss: {test_loss:.4f}, Acc: {acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )
        print(
            classification_report(
                all_true_labels,
                all_pred_labels,
                target_names=class_names,
                zero_division=0,
            )
        )

        # plot_confusion_matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Centralized Learning Confusion Matrix")
        plt.tight_layout()
        # plt.savefig("cl_confusion_matrix.png")
        plt.savefig("outputs/cl/results/cl_confusion_matrix.png")
        plt.close()

        return test_loss, acc, f1, all_pred_labels, all_true_labels


def main():
    os.makedirs("outputs/cl/checkpoints", exist_ok=True)
    os.makedirs("outputs/cl/results", exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            # Additional transforms for data augmentation
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))], p=0.8
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_folder_path = "data/CentralizedLearning/task/train"
    test_folder_path = "data/CentralizedLearning/task/test"

    train_dataset = datasets.ImageFolder(
        root=train_folder_path, transform=train_transform
    )
    test_dataset = datasets.ImageFolder(root=test_folder_path, transform=test_transform)

    # Calculate class weights to handle class imbalance
    class_counts = torch.bincount(torch.tensor(train_dataset.targets))
    total = class_counts.sum().float()
    class_weights = total / (len(class_counts) * class_counts.float())
    print(
        f"Class counts: {class_counts.tolist()}, Class weights: {class_weights.tolist()}"
    )

    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    class_names = list(train_dataset.class_to_idx.keys())

    # Set Hyperparameters
    k_folds = 5
    learning_rate = 5e-4
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    trainer = CentralizedLearning(
        batch_size, learning_rate, num_epochs, device, class_weights
    )
    trainer.train(Model(), train_dataloader, k_folds, class_names, test_dataloader)
    print("\nTraining finished!")


if __name__ == "__main__":
    main()
