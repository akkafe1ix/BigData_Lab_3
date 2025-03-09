from sqlalchemy import create_engine
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
from sklearn.metrics import roc_curve

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

# Данные для подключения
host = "povt-cluster.tstu.tver.ru"
port = 5432
user = "mpi"
password = "135a1"
database = "leonov"

# Функция подключения к базе данных
def connect_to_database():
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}?client_encoding=utf8")
        print("Подключение успешно")
        return engine
    except Exception as e:
        print("Ошибка подключения:", e)
        exit()

# Функция загрузки полной базы данных
def load_full_data(engine):
    query = '''
    select 
        k.*, 
        g."adi" as "группа крови",
        ilce."adi" as "район рождения",
        il."adi" as "город рождения",
        fak."adi" as "факультет",
        bol."adi" as "отделение",
        d."adi" as "курс",
        dao."vize" as "оценка за промежуточный экзамен",
        dao."final" as "оценка за финальный экзамен",
        dao."harf" as "оценка (буквенная)",
        yo."ucret" as "оплата за летнюю школу",
        case 
            when k."cinsiyet" = true then 'мужчина'
            when k."cinsiyet" = false then 'женщина'
            else 'не указан'
        end as "пол"
    from 
        "tkullanicilar" k
    left join 
        "tkangruplari" g on k."kangrubu_tkangruplariid" = g."id"
    left join 
        "tilceler" ilce on k."dogumyeri_tilcelerid" = ilce."id"
    left join 
        "tiller" il on ilce."il_tillerid" = il."id"
    left join 
        "togrenciler" ogr on k."id" = ogr."ogrenci_tkullanicilarid"
    left join 
        "tbolumler" bol on ogr."bolum_tbolumlerid" = bol."id"
    left join 
        "tfakulteler" fak on bol."fakulte_tfakultelerid" = fak."id"
    left join 
        "tdersialanogrenciler" dao on k."id" = dao."ogrenci_tkullanicilarid"
    left join 
        "tdersler" d on dao."ders_tderslerid" = d."id"
    left join 
        "tyazokuluucretleri" yo on k."id" = yo."ogrenci_tkullanicilarid"
    limit 10000;
    '''
    return pd.read_sql_query(query, engine)

# Подключение к базе
engine = connect_to_database()
df = load_full_data(engine)

print("\n================= 1. Загрузка данных =================")
print(f"Размер данных: {df.shape}")

print("\n================= 2. Информация о данных =================")
print(df.info())

print("\n================= 3. Проверка пропущенных значений =================")
missing_values = df.isnull().sum()
print(missing_values if missing_values.sum() > 0 else "Пропущенные значения отсутствуют.")

print("\n================= 4. Анализ числовых признаков =================")
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_stats = df[numerical_features].agg(['min', 'median', 'mean', 'max']).T
numerical_stats["25%"] = df[numerical_features].apply(lambda x: np.percentile(x.dropna(), 25))
numerical_stats["75%"] = df[numerical_features].apply(lambda x: np.percentile(x.dropna(), 75))
print(numerical_stats)

print("\n================= 5. Анализ категориальных признаков =================")
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
if categorical_features:
    categorical_modes = df[categorical_features].mode().iloc[0]
    categorical_frequencies = df[categorical_features].apply(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 'Нет данных')
    categorical_stats = pd.DataFrame({"Mode": categorical_modes, "Frequency": categorical_frequencies})
    print(categorical_stats)
else:
    print("Категориальные признаки отсутствуют.")

print("\n================= 6. Подготовка данных =================")
df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)
print("Пропущенные значения обработаны.")
# Преобразование целевой переменной в числовой формат (1 - мужчина, 0 - женщина)
df["пол"] = df["пол"].map({"мужчина": 1, "женщина": 0})

print("\n================= 7. Выбор признаков =================")
selected_features = ["оценка за промежуточный экзамен", "оценка за финальный экзамен"]
if all(feature in df.columns for feature in selected_features):
    X_selected = df[selected_features].values
    y_selected = df["пол"].values
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

#print("\n================= 13. Функция для построения ROC-кривой =================")
def plot_roc_curve(y_train, y_train_prob, y_test, y_test_prob,name):
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='green', label='ROC curve Train')
    plt.plot(fpr_test, tpr_test, color='blue', label='ROC curve Test')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({name})')
    plt.legend()
    plt.show()

knn_selected = KNeighborsClassifier(n_neighbors=5)
knn_selected.fit(X_selected_scaled, y_selected)
plot_decision_boundary(knn_selected, X_selected_scaled, y_selected, selected_features)
# Вызов функции построения ROC-кривой после обучения всех моделей
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_val)[:, 1]
        plot_roc_curve(y_train, y_train_prob, y_val, y_test_prob,name)

