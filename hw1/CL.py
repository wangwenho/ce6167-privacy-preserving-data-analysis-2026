import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets
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


class CentralizedLearning:
    def __init__(self, batch_size, learning_rate, num_epochs, device):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device

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
        criterion = nn.CrossEntropyLoss()

        from sklearn.model_selection import KFold

        dataset = train_dataloader.dataset
        kfold = KFold(n_splits=k_folds, shuffle=True)

        val_losses = []
        best_epochs = []

        # Phase 1: Early stopping based on validation loss
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold + 1}/{k_folds}")
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)  # type: ignore
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)  # type: ignore
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=train_subsampler
            )
            val_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=val_subsampler
            )

            fold_model = Model(num_classes=2).to(self.device)
            optimizer = optim.Adam(fold_model.parameters(), lr=self.learning_rate)

            best_val_loss = float("inf")
            best_epoch = 0
            patience = 5
            wait = 0

            for epoch in range(self.num_epochs):
                fold_model.train()
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
                    best_epoch = epoch
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            val_losses.append(best_val_loss)
            best_epochs.append(best_epoch)

        # Phase 2: Use full training data to train the final model with the optimal number of epochs
        optimal_epochs = int(sum(best_epochs) / len(best_epochs))
        print(optimal_epochs)

        final_model = Model(num_classes=2).to(self.device)
        optimizer = optim.Adam(final_model.parameters(), lr=self.learning_rate)

        for epoch in range(optimal_epochs):
            final_model.train()
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

            # self.evaluate(model, val_loader, class_names)

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
        os.makedirs("outputs/cl", exist_ok=True)
        plt.savefig("outputs/cl/cl_confusion_matrix.png")
        plt.close()

        return test_loss, acc, f1, all_pred_labels, all_true_labels


def main():

    basic_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_folder_path = "data/CentralizedLearning/task/train"
    test_folder_path = "data/CentralizedLearning/task/test"

    train_dataset = datasets.ImageFolder(
        root=train_folder_path, transform=basic_transform
    )
    test_dataset = datasets.ImageFolder(
        root=test_folder_path, transform=basic_transform
    )

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = list(train_dataset.class_to_idx.keys())

    # Set Hyperparameters
    k_folds = 2
    learning_rate = 0.001
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    trainer = CentralizedLearning(batch_size, learning_rate, num_epochs, device)
    trainer.train(Model(), train_dataloader, k_folds, class_names, test_dataloader)
    print("\nTraining finished!")


if __name__ == "__main__":
    main()
