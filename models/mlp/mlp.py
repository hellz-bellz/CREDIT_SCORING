from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from preprocessing.create_dummy import create_dummy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv('../data/filled.csv')
data_to_model = create_dummy(df)
X = data_to_model.drop(columns="BAD")

for col in X.columns:
    min_ = X[col].min()
    max_ = X[col].max()
    X[col] -= min_
    X[col] /= max_


y = data_to_model["BAD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

parameters = {'max_iter': [1000, 1500, 2000],
              'alpha': [0.1, 0.01, 0.001, 0],
              'hidden_layer_sizes' : [[256,], [256, 256, 256,], [256, 256, 256, 256, 256,]],
              'learning_rate' : ['constant'],
              'learning_rate_init' : [1e-2, 1e-3, 1e-4],
              'activation': ['relu']
              }
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.best_params_)
y_pred = clf.predict(X_test)


print("Accuracy score : ", accuracy_score(y_test, y_pred))
print("F1 score : ", f1_score(y_test, y_pred))
print("Precision score : ", precision_score(y_test, y_pred))
print("Recall score : ", recall_score(y_test, y_pred))

filename = 'mlp_model.sav'
pickle.dump(clf, open(filename, 'wb'))
