import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib.colors import ListedColormap
from collections import Counter

#  Реализация KNN с нуля
class KNN:
    def __init__(self, k=5):
        """ Инициализация модели с параметром k (количество ближайших соседей). """
        self.k = k

    def fit(self, X_train, y_train):
        """ Запоминаем обучающие данные. """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        """ Для каждого объекта из тестового набора находим предсказанный класс. """
        predictions = [self._predict(x) for x in np.array(X_test)]
        return np.array(predictions)

    def _predict(self, x):
        """ Находим класс для одного объекта x. """
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def _euclidean_distance(self, x1, x2):
        """ Вычисление евклидова расстояния между двумя точками. """
        return np.sqrt(np.sum((x1 - x2) ** 2))

#  Загрузка данных
print("\n================= 1. Загрузка данных =================")
try:
    train = pd.read_csv("C:\\Users\\Алексей\\Desktop\\8 Семестр\\(Корнеева) Анализ больших данных\\Lab\\BigData_Lab_3\\data\\train.csv")
    test = pd.read_csv("C:\\Users\\Алексей\\Desktop\\8 Семестр\\(Корнеева) Анализ больших данных\\Lab\\BigData_Lab_3\\data\\test.csv")
    print(f"Размер train: {train.shape}")
    print(f"Размер test: {test.shape}")
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit()

print("\n================= 2. Информация о данных =================")
print(train.info() if not train.empty else "Данные train отсутствуют.")

print("\n================= 3. Проверка пропущенных значений =================")
if not train.empty:
    missing_values = train.isnull().sum()
    print(missing_values if missing_values.sum() > 0 else "Пропущенные значения отсутствуют.")
else:
    print("Данные train отсутствуют.")

print("\n================= 4. Анализ числовых признаков =================")
numerical_features = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
if numerical_features:
    numerical_stats = train[numerical_features].agg(['min', 'median', 'mean', 'max']).T
    numerical_stats["25%"] = train[numerical_features].apply(lambda x: np.percentile(x.dropna(), 25))
    numerical_stats["75%"] = train[numerical_features].apply(lambda x: np.percentile(x.dropna(), 75))
    print(numerical_stats)
else:
    print("Числовые признаки отсутствуют.")

print("\n================= 5. Анализ категориальных признаков =================")
categorical_features = train.select_dtypes(include=['object']).columns.tolist()
if categorical_features:
    categorical_modes = train[categorical_features].mode().iloc[0] if not train[categorical_features].mode().empty else None
    categorical_frequencies = train[categorical_features].apply(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
    categorical_stats = pd.DataFrame({"Mode": categorical_modes, "Frequency": categorical_frequencies})
    print(categorical_stats)
else:
    print("Категориальные признаки отсутствуют.")

print("\n================= 6. Подготовка данных =================")
if not train.empty:
    train.fillna(train.median(), inplace=True)
    print("Пропущенные значения обработаны.")
else:
    print("Данные train отсутствуют.")

print("\n================= 7. Выбор признаков =================")
selected_features = ["gravity", "ph"]
if all(feature in train.columns for feature in selected_features):
    X_selected = train[selected_features].values
    y_selected = train["target"].values
else:
    print("Ошибка: Выбранные признаки отсутствуют в данных.")
    exit()

print("\n================= 8. Нормализация данных =================")
scaler = StandardScaler()
X_selected_scaled = scaler.fit_transform(X_selected)
print("Нормализация выполнена.")

print("\n================= 9. Разделение данных на train/test =================")
X_train, X_val, y_train, y_val = train_test_split(X_selected_scaled, y_selected, test_size=0.2, random_state=42)
print("Разделение выполнено.")

print("\n================= 10. Обучение моделей =================")
models = {
    "KNN (собственная реализация)": KNN(k=5),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True)
}

results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if name == "KNN (собственная реализация)":
            y_prob = np.zeros_like(y_pred, dtype=float)  # KNN не даёт вероятности
        else:
            y_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob) if name != "KNN (собственная реализация)" else "N/A"

        results[name] = [acc, prec, rec, f1, auc]

        print(f"\n{name}:\n Accuracy: {acc:.4f}\n Precision: {prec:.4f}\n Recall: {rec:.4f}\n F1: {f1:.4f}\n AUC: {auc}")
    except Exception as e:
        print(f"Ошибка при обучении {name}: {e}")


print("\n================= 11. Выбор лучшей модели =================")
if results:
    best_model = max(results, key=lambda x: results[x][-1] if isinstance(results[x][-1], float) else results[x][-2])
    print(f"Лучшая модель: {best_model}")
else:
    print("Не удалось выбрать лучшую модель.")

#print("\n================= 12. Визуализация границ классов =================")
def plot_decision_boundary(model, X, y, feature_names):
    try:
        h = 0.02  # Шаг сетки
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
        cmap_bold = ListedColormap(["#FF0000", "#AAFFAA"])

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f"Границы классов для {feature_names[0]} и {feature_names[1]}")
        plt.show()
    except Exception as e:
        print(f"Ошибка при построении границ классов: {e}")

knn_selected = KNeighborsClassifier(n_neighbors=5)
knn_selected.fit(X_selected_scaled, y_selected)
plot_decision_boundary(knn_selected, X_selected_scaled, y_selected, selected_features)
