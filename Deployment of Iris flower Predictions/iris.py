from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings("ignore")

iris = load_iris()
iris.data
iris.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
from sklearn.svm import SVC
model = SVC()
model = model.fit(X_train, y_train)


import pickle
pickle.dump(model, open('iris.pkl', 'wb'))
model=pickle.load(open('iris.pkl','rb'))
print(model.predict([[1.5, 4.5, 3.5, 2.3]])