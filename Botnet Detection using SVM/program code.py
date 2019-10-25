#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA as PCA

#importing our botnet dataset
dataset = pd.read_csv('combo.csv')
X = dataset.iloc[1:30, 1:30].values
Y = dataset.iloc[1:30, 10].values
dataset.head()
print("Botnet data set dimensions : {}".format(dataset.shape))


dataset.isnull().sum()
dataset.isna().sum()
print("Success 1")

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("Success2")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
print("Success3")


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("Success4")

pca = PCA(n_components=2)# adjust yourself
pca.fit(X_train)
X_t_train = pca.transform(X_train)
print("Training set size:",X_t_train.shape)
X_t_test = pca.transform(X_test)
print("Testing set size:",X_t_test.shape)

#Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_t_train, Y_train)
print("Training done..")
Y_pred = classifier.predict(X_t_test)
#print("The predicted values are:",Y_pred)


# Visualising the Training set results for 'rbf' kernel
from matplotlib.colors import ListedColormap
X_set, y_set = X_t_train, Y_train
print("copied")

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
print("predicting")

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
print("prediction done...")

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'black'))(i))
plt.title('SVM (Training set)')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = X_t_test, Y_test
print("copied")

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
print("predicting")
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

print("prediction done...")

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

print("limit set")
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'black'))(i))
plt.title('SVM (Test set)')
plt.legend()
plt.show()
