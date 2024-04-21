from sklearn.datasets import load_iris
import Functions as fs

iris = load_iris()
X = iris.data[:, :2]
k = 5

result  = fs.bisecting_kmeans(X, k, 10)
print("\n RESULT:")
print(result)