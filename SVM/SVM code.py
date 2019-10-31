import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")


X = np.array([[1,2],[2,3],[10,7],[5,4],[8,6],
            [5,8],
            [1.5,1.8],
            [8,8],
            [1,0.6],
            [9,11]])
y = [0,0,1,0,1,1,0,1,0,1]

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


clf= svm.SVC(kernel='linear')
clf.fit(X,y)


w= clf.coef_[0]

a=-w[0]/w[1]

xx=np.linspace(0,13)
yy=a*xx-clf.intercept_[0]/w[1]



h0=plt.plot(xx,yy,'k-',label="non weighted div")


plt.scatter(X[:,0],X[:,1],c=y)
plt.legend()
plt.show()

print("Enter your numbers")
a=input()
b=input()
print(clf.predict([[a,b]]))
