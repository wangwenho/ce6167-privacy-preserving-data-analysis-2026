import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class Model(nn.Module):
    # TODO: Implement your own model
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
            avg_weights[key] = torch.stack(
                [client_weights[i][key] for i in range(len(client_weights))]
            ).mean(0)
        return avg_weights
        return

    def round_robin_training(self):
        # TODO: Implement the round-robin training process and cross validation
        criterion = nn.CrossEntropyLoss()

        for round_idx in range(self.epochs):
            print(f"Round {round_idx + 1}/{self.epochs}")
            client_weights = []

            for client_name in ["Client1", "Client2"]:
                client_model = Model(num_classes=2).to(self.device)
                client_model.load_state_dict(self.global_model.state_dict())

                optimizer = optim.Adam(client_model.parameters(), lr=self.lr)

                train_loader = self.clients[client_name]["train"]
                self.train_client(
                    client_model, train_loader, optimizer, criterion, client_name
                )

                # client_weights.append(client_model.state_dict().copy())
                client_weights.append({k: v.clone() for k, v in client_model.state_dict().items()})

            avg_weights = self.weight_aggregation(client_weights)
            self.global_model.load_state_dict(avg_weights)

            server_test_loader = self.server["Server"]["test"]
            test_loss, test_acc, test_f1, _, _ = self.evaluate(
                self.global_model, server_test_loader
            )
            print(
                f"Round {round_idx + 1} - Server Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}"
            )

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
        os.makedirs("outputs/fl", exist_ok=True)
        plt.savefig("outputs/fl/federated_learning_loss.png")
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
        os.makedirs("outputs/fl", exist_ok=True)
        filename = f"outputs/fl/{name.lower()}_fl_confusion_matrix.png"
        plt.savefig(filename)
        os.makedirs("outputs/fl", exist_ok=True)
        plt.close()


def main():
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

    basic_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    batch_size = 32

    client_dataloaders = {
        client: {
            "train": DataLoader(
                datasets.ImageFolder(path["train"], transform=basic_transform),
                batch_size=batch_size,
                shuffle=True,
            ),
            "test": DataLoader(
                datasets.ImageFolder(path["test"], transform=basic_transform),
                batch_size=batch_size,
                shuffle=False,
            ),
        }
        for client, path in client_dataset_paths.items()
    }

    server_dataloaders = {
        "Server": {
            "test": DataLoader(
                datasets.ImageFolder(
                    server_dataset_path["Server"]["test"], transform=basic_transform
                ),
                batch_size=batch_size,
                shuffle=False,
            )
        }
    }

    # Hyperparameters
    epoch = 20
    lr = 0.0005
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
