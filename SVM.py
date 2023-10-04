import os
import json
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load PATH variable
with open("config.json", "r") as file:
    config = json.load(file)
DATASETS = config["paths"]["DATASETS"]
DATABASES = config["paths"]["DATABASES"]

# Import Data
dataset_path = os.path.join(DATASETS, "lbp_BND radius=5 n_points=40 n_samples=5301.hdf5") #GridSearch: clf = SVC(kernel='rbf', C=100, gamma=1)
#dataset_path = os.path.join(DATASETS, "log_lbp|radius=5|n_points=40|n_samples=3616.hdf5") #GridSearch: clf = SVC(kernel='rbf', C=100, gamma=0.1)
hdf5 = tb.open_file(dataset_path)
dataset_table = hdf5.get_node("/dataset")
descriptors = dataset_table.col("descriptor")
labels = dataset_table.col("label")
hdf5.close()

# Contar a quantidade de casos por classe
class_counts = dict(zip(*np.unique(labels, return_counts=True)))
print(class_counts)


# split dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(descriptors, labels, test_size=0.20)


# Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Grid Search
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
# }
# grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3, n_jobs=6)  # n_jobs=-1 usa todos os CPUs disponíveis
# grid_search.fit(x_train, y_train)
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# print(best_params)
# print(best_score)

# Train
clf = SVC(kernel='rbf', C=100, gamma=1)
clf.fit(x_train, y_train)

# Valoração
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')  # Use 'macro' for multiclass; change if binary classification
recall = recall_score(y_test, y_pred, average='macro')  # Use 'macro' for multiclass; change if binary classification
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Recall:", recall)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['UNCERTAIN',  'COVID', 'NORMAL'])
disp.plot()
plt.show()

# Mapeando labels para números para vizualizações
label_to_number_mapping = {b'NORMAL': 2, b'BND': 0, b'COVID': 1}
y_train_numeric = np.array([label_to_number_mapping[label] for label in y_train])


plt.grid(True)
plt.show()
