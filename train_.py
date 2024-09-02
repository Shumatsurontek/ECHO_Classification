import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    mlflow.set_experiment("Experience 1")

    # Initialiser MLflow
    with mlflow.start_run():
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        example_input = None
        example_output = None

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # Stocker un exemple d'entrée et de sortie pour la signature du modèle
                        if example_input is None:
                            example_input = inputs.detach().cpu().numpy()
                            example_output = outputs.detach().cpu().numpy()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # Enregistrer les métriques dans MLflow
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc.item())
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("train_acc", epoch_acc.item(), step=epoch)
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc.item())
                    mlflow.log_metric("val_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("val_acc", epoch_acc.item(), step=epoch)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            print()

        # Tracer les courbes de perte et de précision
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(num_epochs), train_losses, label='Train Loss')
        plt.plot(range(num_epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(num_epochs), train_accs, label='Train Accuracy')
        plt.plot(range(num_epochs), val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()

        # Sauvegarder les figures dans MLflow
        plt.savefig("training_curves.png")
        mlflow.log_artifact("training_curves.png")

        # Déduire et loguer la signature du modèle avec un exemple d'entrée
        signature = infer_signature(example_input, example_output)
        mlflow.pytorch.log_model(model, "model", signature=signature, input_example=example_input, pip_requirements="pip_requirements.txt")

    return model

def evaluate_model(model, dataloaders, device):
    model.eval()
    corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = corrects.double() / total
    print(f'Accuracy on validation set: {accuracy:.4f}')
