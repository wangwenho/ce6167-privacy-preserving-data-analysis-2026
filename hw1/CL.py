import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
import seaborn as sns
from tqdm import tqdm


class Model(nn.Module):
    # TODO: Implement your own model
    def __init__(self, num_classes=2):
        super(Model, self).__init__()

    def forward(self, x):
        return


class CentralizedLearning:
    def __init__(self, batch_size, learning_rate, num_epochs, device):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device

    def calculate_f1_score(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    def train(self, model, train_dataloader, k_folds=3, class_names=None, test_dataloader=None):
        # TODO: Implement the training process and cross validation.
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
        f1 = report["weighted avg"]["f1-score"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        acc = sum(int(a == b) for a, b in zip(all_true_labels, all_pred_labels)) / len(all_true_labels)
        Test_loss = total_loss / total
        print(f"Test Loss: {Test_loss:.4f}, Acc: {acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(classification_report(all_true_labels, all_pred_labels, target_names=class_names, zero_division=0))
        
        # plot_confusion_matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Centralized Learning Confusion Matrix")
        plt.tight_layout()        
        plt.savefig("cl_confusion_matrix.png")
        plt.close()        
        return


def main():

    basic_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


    train_folder_path = "data/CentralizedLearning/task/train"
    test_folder_path = "data/CentralizedLearning/task/test"

    train_dataset = datasets.ImageFolder(root=train_folder_path, transform=basic_transform)
    test_dataset = datasets.ImageFolder(root=test_folder_path, transform=basic_transform)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = list(train_dataset.class_to_idx.keys())

    # Set Hyperparameters
    k_folds = 5  
    learning_rate = 0.001
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    trainer = CentralizedLearning(batch_size, learning_rate, num_epochs, device)
    trainer.train(Model, train_dataloader, k_folds, class_names, test_dataloader)
    print("\nTraining finished!")

if __name__ == "__main__":
    main()
