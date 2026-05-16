import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
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


class FederatedLearning:
    def __init__(self, clients, server, epochs, batch_size, lr, k_folds):
        self.clients = clients
        self.server = server
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k_folds = k_folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = Model(num_classes=2).to(self.device)
        self.loss_history = {"Client1": [], "Client2": [], "Server": []}
        self.class_names = ["COVID", "Normal"]

    def train_client(self, model, train_loader, optimizer, criterion, client_name):
        model.train()
        total_loss = 0.0
        total = 0

        # Calculate class weights to handle class imbalance
        if hasattr((train_loader.dataset), "targets"):
            targets = train_loader.dataset.targets
        else:
            targets = [y for _, y in train_loader.dataset]

        class_counts = torch.bincount(torch.tensor(targets))
        total_samples = class_counts.sum().float()
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for name, module in model.backbone.named_modules():
            if (name.startswith("bn1") or name.startswith("layer1")) and isinstance(
                module, nn.BatchNorm2d
            ):
                module.eval()

        batch_loop = tqdm(
            train_loader, desc=f"Training {client_name}", leave=False, unit="batch"
        )
        for data, target in batch_loop:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total += target.size(0)

        avg_loss = total_loss / total
        self.loss_history[client_name].append(avg_loss)
        return avg_loss

    def evaluate(self, model, test_loader, criterion=None):
        model.to(self.device)
        model.eval()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                loss = criterion(output, target)
                pred = output.argmax(dim=1)

                total_loss += loss.item() * data.size(0)
                correct += (pred == target).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().tolist())
                all_targets.extend(target.cpu().tolist())

        Test_loss = total_loss / total
        acc = correct / total
        report = self.calculate_f1_score(all_targets, all_preds)
        f1 = report["weighted avg"]["f1-score"]  # type: ignore

        return (
            Test_loss,
            acc,
            f1,
            all_preds,
            all_targets,
        )  # return preds, targets for final_evaluation

    def calculate_f1_score(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    def weight_aggregation(self, client_weights):
        # TODO: Implement the weight aggregation(FedAvg)
        avg_weights = {}
        for key in client_weights[0].keys():
            # Skip non-floating point tensors (e.g., num_batches_tracked is Long)
            if client_weights[0][key].dtype not in [
                torch.float16,
                torch.float32,
                torch.float64,
            ]:
                avg_weights[key] = client_weights[0][key].clone()
            else:
                avg_weights[key] = torch.stack(
                    [client_weights[i][key] for i in range(len(client_weights))]
                ).mean(0)
        return avg_weights

    def round_robin_training(self):
        # TODO: Implement the round-robin training process and cross validation
        criterion = nn.CrossEntropyLoss()
        best_server_loss = float("inf")
        patience = 5
        wait = 0

        for round_idx in range(self.epochs):
            print(f"Round {round_idx + 1}/{self.epochs}")
            client_weights = []

            for client_name in ["Client1", "Client2"]:
                client_model = Model(num_classes=2).to(self.device)
                client_model.load_state_dict(self.global_model.state_dict())

                optimizer = optim.AdamW(
                    client_model.parameters(), lr=self.lr, weight_decay=1e-4
                )

                train_loader = self.clients[client_name]["train"]
                self.train_client(
                    client_model, train_loader, optimizer, criterion, client_name
                )

                # client_weights.append(client_model.state_dict().copy())
                client_weights.append(
                    {k: v.clone() for k, v in client_model.state_dict().items()}
                )

            avg_weights = self.weight_aggregation(client_weights)
            torch.save(
                avg_weights, f"outputs/fl/checkpoints/avg_weighted_model_round_{round_idx + 1}.pt"
            )
            self.global_model.load_state_dict(avg_weights)

            server_test_loader = self.server["Server"]["test"]
            test_loss, test_acc, test_f1, _, _ = self.evaluate(
                self.global_model, server_test_loader
            )
            print(
                f"Round {round_idx + 1} - Server Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}"
            )

            if test_loss < best_server_loss:
                best_server_loss = test_loss
                torch.save(
                    self.global_model.state_dict(),
                    "outputs/fl/checkpoints/fl_model.pt",
                )
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at round {round_idx + 1}")
                    self.global_model.load_state_dict(
                        torch.load("outputs/fl/checkpoints/fl_model.pt")
                    )
                    break

            self.loss_history["Server"].append(test_loss)

            for client_name in ["Client1", "Client2"]:
                test_loader = self.clients[client_name]["test"]
                test_loss, test_acc, test_f1, preds, targets = self.evaluate(
                    self.global_model, test_loader
                )
                self.plot_confusion_matrix(targets, preds, name=client_name)

            server_test_loader = self.server["Server"]["test"]
            test_loss, test_acc, test_f1, preds, targets = self.evaluate(
                self.global_model, server_test_loader
            )
            self.plot_confusion_matrix(targets, preds, name="Server")
            self.plot_loss()
        return

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        for client_name, losses in self.loss_history.items():
            plt.plot(range(len(losses)), losses, label=f"{client_name} Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Federated Learning Training Loss per Client")
        plt.legend()
        plt.tight_layout()
        # plt.savefig("federated_learning_loss.png")
        plt.savefig("outputs/fl/results/federated_learning_loss.png")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, name="Model"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        filename = f"outputs/fl/results/{name.lower()}_fl_confusion_matrix.png"
        plt.savefig(filename)
        plt.close()


def main():
    os.makedirs("outputs/fl/checkpoints", exist_ok=True)
    os.makedirs("outputs/fl/results", exist_ok=True)

    client_dataset_paths = {
        "Client1": {
            "train": "data/FederatedLearning/Client1/train/",
            "test": "data/FederatedLearning/Client1/test/",
        },
        "Client2": {
            "train": "data/FederatedLearning/Client2/train/",
            "test": "data/FederatedLearning/Client2/test/",
        },
    }

    server_dataset_path = {
        "Server": {"test": "data/FederatedLearning/Server/test/"},
    }

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

    batch_size = 32

    client_dataloaders = {
        client: {
            "train": DataLoader(
                datasets.ImageFolder(path["train"], transform=train_transform),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            ),
            "test": DataLoader(
                datasets.ImageFolder(path["test"], transform=test_transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            ),
        }
        for client, path in client_dataset_paths.items()
    }

    server_dataloaders = {
        "Server": {
            "test": DataLoader(
                datasets.ImageFolder(
                    server_dataset_path["Server"]["test"], transform=test_transform
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        }
    }

    # Hyperparameters
    epoch = 20
    lr = 1e-4
    k_folds = 5

    print(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    fl = FederatedLearning(
        client_dataloaders,
        server_dataloaders,
        epochs=epoch,
        batch_size=batch_size,
        lr=lr,
        k_folds=k_folds,
    )
    fl.round_robin_training()
    print("\nTraining finished!")


if __name__ == "__main__":
    main()
