import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import random
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class FaceVerificationMLP(nn.Module):
    def __init__(self, input_dim=512):
        super(FaceVerificationMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


class FaceVerificationSystem:
    def __init__(self, img_dir, identity_file):
        self.img_dir = img_dir
        self.identity_file = identity_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Używam urządzenia: {self.device}")

        self.mtcnn = MTCNN(image_size=160, margin=20, device=self.device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.load_identity_data()

        self.embedding_cache = {}

    def load_identity_data(self):
        self.identity_data = {}
        self.person_images = defaultdict(list)

        with open(self.identity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, person_id = parts
                    self.identity_data[img_name] = int(person_id)
                    self.person_images[int(person_id)].append(img_name)

        self.person_images = {pid: imgs for pid, imgs in self.person_images.items() if len(imgs) >= 2}
        print(f"Wczytano {len(self.person_images)} osób z co najmniej 2 zdjęciami")

    def get_embedding(self, image_path):
        if image_path in self.embedding_cache:
            return self.embedding_cache[image_path]

        try:
            full_path = os.path.join(self.img_dir, image_path)
            img = Image.open(full_path).convert('RGB')

            face = self.mtcnn(img)
            if face is None:
                return None

            with torch.no_grad():
                face_embedding = self.facenet(face.unsqueeze(0).to(self.device))
                embedding = face_embedding.cpu().squeeze()

            self.embedding_cache[image_path] = embedding
            return embedding

        except Exception as e:
            print(f"Błąd przetwarzania {image_path}: {e}")
            return None

    def create_pairs(self, num_pairs, excluded_persons=None):
        if excluded_persons is None:
            excluded_persons = set()

        positive_pairs = []
        negative_pairs = []
        used_persons = set()
        available_persons = [pid for pid in self.person_images.keys() if pid not in excluded_persons]
        random.shuffle(available_persons)

        # Pary pozytywne (ta sama osoba)
        pos_needed = num_pairs // 2
        pos_count = 0

        for person_id in available_persons:
            if pos_count >= pos_needed:
                break

            person_imgs = self.person_images[person_id]

            if len(person_imgs) >= 2:
                img1, img2 = random.sample(person_imgs, 2)
                emb1 = self.get_embedding(img1)
                emb2 = self.get_embedding(img2)

                if emb1 is not None and emb2 is not None:
                    diff_vector = torch.abs(emb1 - emb2)
                    positive_pairs.append((diff_vector, 1))
                    used_persons.add(person_id)
                    pos_count += 1

        # Pary negatywne (różne osoby)
        neg_needed = num_pairs - pos_count
        neg_count = 0

        remaining_persons = [pid for pid in available_persons if pid not in used_persons]

        while neg_count < neg_needed and len(remaining_persons) > 1:
            person1_id, person2_id = random.sample(remaining_persons, 2)

            img1 = random.choice(self.person_images[person1_id])
            img2 = random.choice(self.person_images[person2_id])

            emb1 = self.get_embedding(img1)
            emb2 = self.get_embedding(img2)

            if emb1 is not None and emb2 is not None:
                diff_vector = torch.abs(emb1 - emb2)
                negative_pairs.append((diff_vector, 0))
                used_persons.add(person1_id)
                used_persons.add(person2_id)
                neg_count += 1

        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        print(f"Utworzono {len(positive_pairs)} par pozytywnych i {len(negative_pairs)} par negatywnych")

        return all_pairs, used_persons

    def create_test_pairs(self, num_test_pairs=200, excluded_persons=None):
        if excluded_persons is None:
            excluded_persons = set()

        available_persons = [pid for pid in self.person_images.keys() if pid not in excluded_persons]

        if len(available_persons) < 20:
            print("Zbyt mało osób dostępnych do testów!")
            return []

        test_pairs = []
        used_persons = set()

        # Pary pozytywne
        pos_needed = num_test_pairs // 2
        pos_count = 0

        for person_id in available_persons:
            if pos_count >= pos_needed:
                break

            person_imgs = self.person_images[person_id]
            if len(person_imgs) >= 2:
                img1, img2 = random.sample(person_imgs, 2)
                emb1 = self.get_embedding(img1)
                emb2 = self.get_embedding(img2)

                if emb1 is not None and emb2 is not None:
                    diff_vector = torch.abs(emb1 - emb2)
                    test_pairs.append((diff_vector, 1))
                    used_persons.add(person_id)
                    pos_count += 1

        # Pary negatywne
        neg_needed = num_test_pairs - pos_count
        neg_count = 0

        remaining_persons = [pid for pid in available_persons if pid not in used_persons]

        while neg_count < neg_needed and len(remaining_persons) > 1:
            person1_id, person2_id = random.sample(remaining_persons, 2)

            img1 = random.choice(self.person_images[person1_id])
            img2 = random.choice(self.person_images[person2_id])

            emb1 = self.get_embedding(img1)
            emb2 = self.get_embedding(img2)

            if emb1 is not None and emb2 is not None:
                diff_vector = torch.abs(emb1 - emb2)
                test_pairs.append((diff_vector, 0))
                used_persons.add(person1_id)
                used_persons.add(person2_id)
                neg_count += 1

        random.shuffle(test_pairs)
        print(f"Utworzono {len(test_pairs)} par testowych ({pos_count} pozytywnych, {neg_count} negatywnych)")

        return test_pairs, used_persons

    def train_model(self, train_pairs, lr=0.001, epochs=50):
        if not train_pairs:
            print("Brak danych treningowych!")
            return None, []

        X_train = torch.stack([pair[0] for pair in train_pairs])
        y_train = torch.tensor([pair[1] for pair in train_pairs], dtype=torch.long)

        model = FaceVerificationMLP().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        return model, train_losses

    def evaluate_model(self, model, test_pairs):
        if not test_pairs:
            print("Brak danych testowych!")
            return {}

        model.eval()

        X_test = torch.stack([pair[0] for pair in test_pairs])
        y_test = torch.tensor([pair[1] for pair in test_pairs], dtype=torch.long)

        with torch.no_grad():
            X_test = X_test.to(self.device)
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
            y_test = y_test.numpy()

        metrics = {
            'accuracy': accuracy_score(y_test, predicted),
            'precision': precision_score(y_test, predicted, zero_division=0),
            'recall': recall_score(y_test, predicted, zero_division=0),
            'f1': f1_score(y_test, predicted, zero_division=0)
        }

        return metrics

    def plot_results(self, results, experiment_type, save_path=None, x_log=False):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Wyniki eksperymentu: {experiment_type}', fontsize=16)

        x_values = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = {metric: [results[x][metric] for x in x_values] for metric in metrics}

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            ax.plot(x_values, metric_values[metric], 'o-', linewidth=2, markersize=8)
            ax.set_title(f'{metric.capitalize()}')
            ax.set_xlabel(experiment_type)
            ax.set_ylabel('Wartość')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            if x_log:
                ax.set_xscale('log')

            for j, v in enumerate(metric_values[metric]):
                ax.annotate(f'{v:.3f}', (x_values[j], v), textcoords="offset points",
                            xytext=(0, 10), ha='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def run_dataset_size_experiment(self):
        print("=== EKSPERYMENT 1: Wpływ rozmiaru zbioru danych ===")

        dataset_sizes = [10, 100, 500, 1000, 5000]
        results = {}

        test_pairs, ex_p = self.create_test_pairs(200)

        for size in dataset_sizes:
            print(f"\n--- Trenowanie z {size} parami ---")

            train_pairs, _ = self.create_pairs(size, ex_p)

            if len(train_pairs) < size * 0.8:
                print(f"Nie udało się utworzyć wystarczająco par dla rozmiaru {size}")
                continue

            model, losses = self.train_model(train_pairs, epochs=50)

            if model is None:
                continue

            metrics = self.evaluate_model(model, test_pairs)
            results[size] = metrics

            print(f"Wyniki dla {size} par:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        if results:
            self.plot_results(results, "Rozmiar zbioru treningowego",
                              "dataset_size_experiment.png")

        return results

    def run_learning_rate_experiment(self):
        print("\n=== EKSPERYMENT 2: Wpływ learning rate ===")

        learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
        results = {}

        train_pairs, ex_p = self.create_pairs(1000)
        test_pairs, _ = self.create_test_pairs(200, ex_p)

        for lr in learning_rates:
            print(f"\n--- Learning rate: {lr} ---")

            model, losses = self.train_model(train_pairs, lr=lr, epochs=50)

            if model is None:
                continue

            metrics = self.evaluate_model(model, test_pairs)
            results[lr] = metrics

            print(f"Wyniki dla lr={lr}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        if results:
            self.plot_results(results, "Learning Rate", "learning_rate_experiment.png", x_log=True)

        return results

    def run_epochs_experiment(self):
        print("\n=== EKSPERYMENT 3: Wpływ liczby epok ===")

        epoch_values = [5, 10, 30, 50, 100]
        results = {}

        train_pairs, ex_p = self.create_pairs(1000)
        test_pairs, _ = self.create_test_pairs(200, ex_p)

        for epochs in epoch_values:
            print(f"\n--- Liczba epok: {epochs} ---")

            model, losses = self.train_model(train_pairs, lr=0.001, epochs=epochs)

            if model is None:
                continue

            metrics = self.evaluate_model(model, test_pairs)
            results[epochs] = metrics

            print(f"Wyniki dla {epochs} epok:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        if results:
            self.plot_results(results, "Liczba epok", "epochs_experiment.png")

        return results

    def run_best_parameters_from_experiments(self):
        print("\n=== Best parameters from experimetns 1-3 ===")

        train_pairs, ex_p = self.create_pairs(1000)
        test_pairs, _ = self.create_test_pairs(200, ex_p)

        model, losses = self.train_model(train_pairs, lr=0.01, epochs=50)

        metrics = self.evaluate_model(model, test_pairs)
        results = metrics

        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return results


def main():
    img_dir = "img_align_celeba"
    identity_file = "identity_CelebA.txt"

    if not os.path.exists(img_dir):
        print(f"Błąd: Folder {img_dir} nie istnieje!")
        return

    if not os.path.exists(identity_file):
        print(f"Błąd: Plik {identity_file} nie istnieje!")
        return

    system = FaceVerificationSystem(img_dir, identity_file)

    print("Rozpoczynam eksperymenty...")

    # Eksperyment 1: Rozmiar zbioru danych
    dataset_results = system.run_dataset_size_experiment()

    # Eksperyment 2: Learning rate
    lr_results = system.run_learning_rate_experiment()

    # Eksperyment 3: Liczba epok
    epochs_results = system.run_epochs_experiment()

    # Podsumowanie
    print("\n=== PODSUMOWANIE WYNIKÓW ===")

    if dataset_results:
        print("\nNajlepsze wyniki dla różnych rozmiarów zbiorów:")
        best_size = max(dataset_results.keys(), key=lambda x: dataset_results[x]['accuracy'])
        print(f"Najlepsza accuracy ({dataset_results[best_size]['accuracy']:.4f}) dla {best_size} par")

    if lr_results:
        print("\nNajlepsze wyniki dla różnych learning rates:")
        best_lr = max(lr_results.keys(), key=lambda x: lr_results[x]['accuracy'])
        print(f"Najlepsza accuracy ({lr_results[best_lr]['accuracy']:.4f}) dla lr={best_lr}")

    if epochs_results:
        print("\nNajlepsze wyniki dla różnych liczb epok:")
        best_epochs = max(epochs_results.keys(), key=lambda x: epochs_results[x]['accuracy'])
        print(f"Najlepsza accuracy ({epochs_results[best_epochs]['accuracy']:.4f}) dla {best_epochs} epok")

    _ = system.run_best_parameters_from_experiments()


if __name__ == "__main__":
    main()
