import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif


class CTGAnalysis:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.X = None
        self.y = None
        self.results = {}

    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Dane wczytane pomyślnie. Rozmiar: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"Nie znaleziono pliku: {self.csv_path}")
            return False

    def data_processing(self, X_train, X_test, y_train, y_test):

        X_train_base = X_train.copy()
        X_test_base = X_test.copy()

        scaler_standard = StandardScaler()
        X_train_standard = pd.DataFrame(
            scaler_standard.fit_transform(X_train_base),
            columns=X_train_base.columns
        )
        X_test_standard = pd.DataFrame(
            scaler_standard.transform(X_test_base),
            columns=X_test_base.columns
        )

        scaler_minmax = MinMaxScaler()
        X_train_normalized = pd.DataFrame(
            scaler_minmax.fit_transform(X_train_base),
            columns=X_train_base.columns
        )
        X_test_normalized = pd.DataFrame(
            scaler_minmax.transform(X_test_base),
            columns=X_test_base.columns
        )

        selector_8 = SelectKBest(score_func=f_classif, k=8)
        X_train_selected_8 = pd.DataFrame(
            selector_8.fit_transform(X_train_base, y_train),
            columns=X_train_base.columns[selector_8.get_support()]
        )
        X_test_selected_8 = pd.DataFrame(
            selector_8.transform(X_test_base),
            columns=X_train_base.columns[selector_8.get_support()]
        )

        selector_15 = SelectKBest(score_func=f_classif, k=15)
        X_train_selected_15 = pd.DataFrame(
            selector_15.fit_transform(X_train_base, y_train),
            columns=X_train_base.columns[selector_15.get_support()]
        )
        X_test_selected_15 = pd.DataFrame(
            selector_15.transform(X_test_base),
            columns=X_train_base.columns[selector_15.get_support()]
        )

        pca_8 = PCA(n_components=8)
        X_train_pca_8 = pd.DataFrame(
            pca_8.fit_transform(X_train_standard),
            columns=[f'PC{i + 1}' for i in range(8)]
        )
        X_test_pca_8 = pd.DataFrame(
            pca_8.transform(X_test_standard),
            columns=[f'PC{i + 1}' for i in range(8)]
        )

        pca_15 = PCA(n_components=15)
        X_train_pca_15 = pd.DataFrame(
            pca_15.fit_transform(X_train_standard),
            columns=[f'PC{i + 1}' for i in range(15)]
        )
        X_test_pca_15 = pd.DataFrame(
            pca_15.transform(X_test_standard),
            columns=[f'PC{i + 1}' for i in range(15)]
        )

        print(f"Variance explained by PCA (8): {pca_8.explained_variance_ratio_.sum():.3f}")
        print(f"Variance explained by PCA (15): {pca_15.explained_variance_ratio_.sum():.3f}")

        return {
            'original': (X_train_base, X_test_base, y_train, y_test),
            'standardized': (X_train_standard, X_test_standard, y_train, y_test),
            'normalized': (X_train_normalized, X_test_normalized, y_train, y_test),
            'selected (8)': (X_train_selected_8, X_test_selected_8, y_train, y_test),
            'selected (15)': (X_train_selected_15, X_test_selected_15, y_train, y_test),
            'pca (8)': (X_train_pca_8, X_test_pca_8, y_train, y_test),
            'pca (15)': (X_train_pca_15, X_test_pca_15, y_train, y_test),
        }

    def data_preparation(self):
        """2. DATA PREPARATION"""
        print("\n" + "=" * 60)
        print("2. DATA PREPARATION - PRZYGOTOWANIE DANYCH")
        print("=" * 60)

        self.X = self.data.drop('CLASS', axis=1)
        self.y = self.data['CLASS']

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"Rozmiar zbioru treningowego: {X_train.shape}")
        print(f"Rozmiar zbioru testowego: {X_test.shape}")

        print("\n--- Metody uzupełniania brakujących wartości ---")

        data_removed = self.data.dropna()
        print(f"Po usunięciu wierszy z brakującymi wartościami: {data_removed.shape}")
        X_train_removal = X_train.dropna()
        y_train_removal = y_train.loc[X_train_removal.index]

        X_test_removal = X_test.dropna()
        y_test_removal = y_test.loc[X_test_removal.index]

        imputer_mean = SimpleImputer(strategy='mean')
        X_train_imputed_mean = pd.DataFrame(
            imputer_mean.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_imputed_mean = pd.DataFrame(
            imputer_mean.transform(X_test),
            columns=X_test.columns
        )

        imputer_knn = KNNImputer(n_neighbors=5)
        X_train_imputed_knn = pd.DataFrame(
            imputer_knn.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_imputed_knn = pd.DataFrame(
            imputer_knn.transform(X_test),
            columns=X_test.columns
        )

        X_train_interpolation = X_train.interpolate()
        X_test_interpolation = X_test.interpolate()

        X_train_interpolation = X_train_interpolation.dropna()
        y_train_interpolation = y_train.loc[X_train_interpolation.index]

        X_test_interpolation = X_test_interpolation.dropna()
        y_test_interpolation = y_test.loc[X_test_interpolation.index]

        processed_removed = self.data_processing(X_train_removal, X_test_removal, y_train_removal, y_test_removal)
        processed_mean = self.data_processing(X_train_imputed_mean, X_test_imputed_mean, y_train, y_test)
        processed_knn = self.data_processing(X_train_imputed_knn, X_test_imputed_knn, y_train, y_test)
        processed_interpolation = self.data_processing(X_train_interpolation, X_test_interpolation, y_train_interpolation, y_test_interpolation)

        return {
            'processed_removed': processed_removed,
            'processed_mean': processed_mean,
            'processed_knn': processed_knn,
            'processed_interpolation': processed_interpolation,
        }

    def plot_best_confusion_matrix(self, results_df, classification_results):
        """Znajdź najlepszy klasyfikator wg F1-score i pokaż jego confusion matrix."""

        best_row = results_df.loc[results_df['F1-Score'].idxmax()]

        imputation = best_row['Imputation Method']
        processing = best_row['Data Processing']
        clf_family = best_row['Classifier Family']
        clf_name = best_row['Classifier']

        print(f"\nNajlepszy klasyfikator wg F1-score:")
        print(f" Imputacja: {imputation}")
        print(f" Przetwarzanie: {processing}")
        print(f" Rodzina klasyfikatora: {clf_family}")
        print(f" Klasyfikator: {clf_name}")
        print(f" F1-score: {best_row['F1-Score']:.3f}\n")

        try:
            conf_matrix = classification_results[imputation][processing][clf_family][clf_name]['confusion_matrix']

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix – {clf_name}\n({imputation} / {processing})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()

        except KeyError:
            print("Nie znaleziono confusion matrix dla najlepszego klasyfikatora – sprawdź zgodność nazw.")

    def classification_experiments(self, data_variants):
        """3. KLASYFIKACJA - eksperymenty z klasyfikatorami"""
        print("\n" + "=" * 60)
        print("3. KLASYFIKACJA - EKSPERYMENTY")
        print("=" * 60)

        results = {}

        classifiers = {
            'Naive Bayes': [
                ('NB_default', GaussianNB()),
                ('NB_var_smoothing_1e-10', GaussianNB(var_smoothing=1e-10)),
                ('NB_var_smoothing_1e-15', GaussianNB(var_smoothing=1e-15))
            ],
            'Decision Tree': [
                ('DT_default', DecisionTreeClassifier(random_state=42)),
                ('DT_max_depth_5', DecisionTreeClassifier(max_depth=5, random_state=42)),
                ('DT_max_depth_10', DecisionTreeClassifier(max_depth=10, random_state=42)),
                ('DT_max_depth_10_min_samples_10', DecisionTreeClassifier(
                    max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42
                ))
            ],
            'Random Forest': [
                ('RF_default', RandomForestClassifier(random_state=42)),
                ('RF_150_trees', RandomForestClassifier(n_estimators=150, random_state=42)),
                ('RF_200_trees_max_depth_10', RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42
                ))
            ],
            'SVM': [
                ('SVM_rbf', SVC(kernel='rbf', random_state=42)),
                ('SVM_linear', SVC(kernel='linear', random_state=42)),
                ('SVM_poly', SVC(kernel='poly', degree=3, random_state=42))
            ]
        }

        for imputation_method, processing_variants in data_variants.items():
            print(f"\n=== Metoda uzupełniania: {imputation_method} ===")
            results[imputation_method] = {}

            for processing_name, (X_train, X_test, y_train, y_test) in processing_variants.items():
                print(f"\n--- Testowanie na danych: {imputation_method} -> {processing_name} ---")
                results[imputation_method][processing_name] = {}

                for clf_family, clf_variants in classifiers.items():
                    results[imputation_method][processing_name][clf_family] = {}

                    for clf_name, clf in clf_variants:
                        try:
                            clf.fit(X_train, y_train)

                            y_pred = clf.predict(X_test)

                            acc = accuracy_score(y_test, y_pred)
                            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                            cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

                            results[imputation_method][processing_name][clf_family][clf_name] = {
                                'accuracy': acc,
                                'precision': prec,
                                'recall': rec,
                                'f1_score': f1,
                                'cv_mean': cv_scores.mean(),
                                'cv_std': cv_scores.std(),
                                'confusion_matrix': confusion_matrix(y_test, y_pred)
                            }

                            print(
                                f"{clf_name}: Acc={acc:.3f}, F1={f1:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

                        except Exception as e:
                            print(f"Błąd dla {clf_name}: {e}")
                            continue

        return results

    def overfitting_mitigation(self, data_variants):
        """metody przeciwdziałania przeuczeniu"""
        print("\n" + "=" * 60)
        print("PRZECIWDZIAŁANIE PRZEUCZENIU")
        print("=" * 60)

        X_train, X_test, y_train, y_test = data_variants['processed_mean']['standardized']

        dt_overfit = DecisionTreeClassifier(random_state=42)
        dt_regularized = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_leaf_nodes=50,
            random_state=42
        )

        results = {}

        for name, clf in [('Bez regularyzacji', dt_overfit), ('Z regularyzacją', dt_regularized)]:
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)

            y_test_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_gap': train_acc - test_acc
            }

            print(f"{name}:")
            print(f"  Dokładność treningowa: {train_acc:.3f}")
            print(f"  Dokładność testowa: {test_acc:.3f}")
            print(f"  Różnica (overfitting): {train_acc - test_acc:.3f}")
            print(f"  Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        return results

    def evaluate_results(self, classification_results):
        """4. OCENA WYNIKÓW - analiza i interpretacja"""
        print("\n" + "=" * 60)
        print("4. OCENA WYNIKÓW - ANALIZA I INTERPRETACJA")
        print("=" * 60)

        results_table = []

        for imputation_method, processing_results in classification_results.items():
            for processing_name, data_results in processing_results.items():
                for clf_family, clf_results in data_results.items():
                    for clf_name, metrics in clf_results.items():
                        results_table.append({
                            'Imputation Method': imputation_method,
                            'Data Processing': processing_name,
                            'Classifier Family': clf_family,
                            'Classifier': clf_name,
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1-Score': metrics['f1_score'],
                            'CV Mean': metrics['cv_mean'],
                            'CV Std': metrics['cv_std']
                        })

        results_df = pd.DataFrame(results_table)

        # Sortowanie według F1-Score
        results_df_sorted = results_df.sort_values('F1-Score', ascending=False)

        print("Top 20 najlepszych konfiguracji (według F1-Score):")
        print(results_df_sorted.head(20).to_string(index=False))

        print("\n--- Analiza wpływu metod imputacji ---")
        imputation_analysis = results_df.groupby('Imputation Method').agg({
            'Accuracy': ['mean', 'std'],
            'F1-Score': ['mean', 'std']
        }).round(3)
        print(imputation_analysis)

        print("\n--- Analiza wpływu metod przetwarzania danych ---")
        data_prep_analysis = results_df.groupby('Data Processing').agg({
            'Accuracy': ['mean', 'std'],
            'F1-Score': ['mean', 'std']
        }).round(3)
        print(data_prep_analysis)

        print("\n--- Analiza klasyfikatorów ---")
        classifier_analysis = results_df.groupby('Classifier Family').agg({
            'Accuracy': ['mean', 'std'],
            'F1-Score': ['mean', 'std']
        }).round(3)
        print(classifier_analysis)

        print("\n--- Analiza poszczególnych klasyfikatorów ---")
        individual_classifier_analysis = results_df.groupby('Classifier').agg({
            'Accuracy': ['mean', 'std'],
            'F1-Score': ['mean', 'std'],
            'CV Mean': ['mean', 'std']
        }).round(3)
        print(individual_classifier_analysis)

        print("\n--- Top 10 najlepszych poszczególnych klasyfikatorów (średni F1-Score) ---")
        individual_clf_means = results_df.groupby('Classifier')['F1-Score'].mean().sort_values(ascending=False)
        print(individual_clf_means.head(10))

        print("\n--- Najstabilniejsze klasyfikatory (najmniejsze CV Std) ---")
        individual_clf_stability = results_df.groupby('Classifier')['CV Std'].mean().sort_values(ascending=True)
        print(individual_clf_stability.head(10))

        # ------------------ WIZUALIZACJA CZĘŚĆ 1 ------------------
        plt.figure(figsize=(20, 14))

        # Wykres 1: Porównanie metod imputacji
        plt.subplot(2, 3, 1)
        imputation_means = results_df.groupby('Imputation Method')['F1-Score'].mean()
        imputation_means.plot(kind='bar')
        plt.title('Średni F1-Score dla metod imputacji')
        plt.xlabel('Metoda imputacji')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)

        # Wykres 2: Porównanie metod przetwarzania danych
        plt.subplot(2, 3, 2)
        data_prep_means = results_df.groupby('Data Processing')['F1-Score'].mean()
        data_prep_means.plot(kind='bar')
        plt.title('Średni F1-Score dla metod przetwarzania danych')
        plt.xlabel('Metoda przetwarzania')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)

        # Wykres 3: Porównanie klasyfikatorów (rodzin)
        plt.subplot(2, 3, 3)
        clf_means = results_df.groupby('Classifier Family')['F1-Score'].mean()
        clf_means.plot(kind='bar')
        plt.title('Średni F1-Score dla rodzin klasyfikatorów')
        plt.xlabel('Rodzina klasyfikatora')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)

        # Wykres 4: Accuracy vs F1-Score
        plt.subplot(2, 3, 4)
        plt.scatter(results_df['Accuracy'], results_df['F1-Score'], alpha=0.6)
        plt.xlabel('Accuracy')
        plt.ylabel('F1-Score')
        plt.title('Accuracy vs F1-Score')

        # Wykres 5: Stabilność cross-validation
        plt.subplot(2, 3, 5)
        plt.scatter(results_df['CV Mean'], results_df['CV Std'], alpha=0.6)
        plt.xlabel('CV Mean')
        plt.ylabel('CV Std')
        plt.title('Stabilność cross-validation')

        # Wykres 6: Rozkład wyników
        plt.subplot(2, 3, 6)
        results_df['F1-Score'].hist(bins=20)
        plt.title('Rozkład wyników F1-Score')
        plt.xlabel('F1-Score')
        plt.ylabel('Częstość')

        plt.tight_layout()
        plt.show()

        # ------------------ WIZUALIZACJA CZĘŚĆ 2 ------------------
        plt.figure(figsize=(22, 12))

        # Wykres 7: Heatmapa wyników - Imputacja vs Klasyfikator
        plt.subplot(2, 2, 1)
        pivot_table_imputation = results_df.pivot_table(
            values='F1-Score',
            index='Imputation Method',
            columns='Classifier Family',
            aggfunc='mean'
        )
        sns.heatmap(pivot_table_imputation, annot=True, cmap='viridis', fmt='.3f')
        plt.title('F1-Score: Imputacja vs Klasyfikator')

        # Wykres 8: Heatmapa wyników - Przetwarzanie vs Klasyfikator
        plt.subplot(2, 2, 2)
        pivot_table_processing = results_df.pivot_table(
            values='F1-Score',
            index='Data Processing',
            columns='Classifier Family',
            aggfunc='mean'
        )
        sns.heatmap(pivot_table_processing, annot=True, cmap='viridis', fmt='.3f')
        plt.title('F1-Score: Przetwarzanie vs Klasyfikator')

        # Wykres 9: Najstabilniejsze klasyfikatory (CV Std)
        plt.subplot(2, 2, 3)
        individual_clf_stability = results_df.groupby('Classifier')['CV Std'].mean().sort_values()
        individual_clf_stability.head(15).plot(kind='bar')
        plt.title('Najstabilniejsze klasyfikatory (średni CV Std)')
        plt.xlabel('Klasyfikator')
        plt.ylabel('Średni CV Std')
        plt.xticks(rotation=90)

        # Wykres 10: Najmniejsza wariancja F1-Score
        plt.subplot(2, 2, 4)
        individual_clf_variance = results_df.groupby('Classifier')['F1-Score'].std().sort_values()
        individual_clf_variance.head(15).plot(kind='bar')
        plt.title('Najmniejsza wariancja F1-Score')
        plt.xlabel('Klasyfikator')
        plt.ylabel('Odchylenie standardowe F1-Score')
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

        print("\n--- INTERPRETACJA WYNIKÓW ---")

        best_config = results_df_sorted.iloc[0]
        print(f"Najlepsza konfiguracja:")
        print(f"  Metoda imputacji: {best_config['Imputation Method']}")
        print(f"  Metoda przetwarzania: {best_config['Data Processing']}")
        print(f"  Klasyfikator: {best_config['Classifier']}")
        print(f"  F1-Score: {best_config['F1-Score']:.3f}")
        print(f"  Accuracy: {best_config['Accuracy']:.3f}")

        best_imputation = imputation_means.idxmax()
        print(f"\nNajlepsza metoda imputacji: {best_imputation}")
        print(f"Średni F1-Score: {imputation_means[best_imputation]:.3f}")

        best_data_prep = data_prep_means.idxmax()
        print(f"\nNajlepsza metoda przetwarzania danych: {best_data_prep}")
        print(f"Średni F1-Score: {data_prep_means[best_data_prep]:.3f}")

        best_classifier = clf_means.idxmax()
        print(f"\nNajlepszy klasyfikator: {best_classifier}")
        print(f"Średni F1-Score: {clf_means[best_classifier]:.3f}")

        best_individual_classifier = individual_clf_means.idxmax()
        print(f"\nNajlepszy poszczególny klasyfikator: {best_individual_classifier}")
        print(f"Średni F1-Score: {individual_clf_means[best_individual_classifier]:.3f}")

        self.plot_best_confusion_matrix(results_df, classification_results)

        return {
            'results_table': results_df_sorted,
            'best_config': best_config,
            'imputation_analysis': imputation_analysis,
            'data_prep_analysis': data_prep_analysis,
            'classifier_analysis': classifier_analysis
        }

    def run_complete_analysis(self):
        """Uruchomienie pełnej analizy"""
        print("ROZPOCZYNANIE PEŁNEJ ANALIZY DANYCH CTG")
        print("=" * 60)

        if not self.load_data():
            return

        data_variants = self.data_preparation()

        classification_results = self.classification_experiments(data_variants)

        overfitting_results = self.overfitting_mitigation(data_variants)

        evaluation_results = self.evaluate_results(classification_results)

        print("\n" + "=" * 60)
        print("ANALIZA ZAKOŃCZONA POMYŚLNIE!")
        print("=" * 60)

        return {
            'data_variants': data_variants,
            'classification': classification_results,
            'overfitting': overfitting_results,
            'evaluation': evaluation_results
        }


if __name__ == "__main__":
    analyzer = CTGAnalysis('cardiotocography_v2.csv')

    results = analyzer.run_complete_analysis()
