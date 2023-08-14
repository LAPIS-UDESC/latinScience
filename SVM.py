import os
import json
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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
print(accuracy)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BND',  'COVID', 'NORMAL'])
disp.plot()
plt.show()
