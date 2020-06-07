import numpy
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics

n_samples = 200
n_features = 6
seed = 42

data, targets = sklearn.datasets.make_regression(
    n_samples=n_samples, n_features=n_features, random_state=seed)

merged_data = numpy.zeros((n_samples, n_features + 1))

for i in range(len(data)):
    merged_data[i] = numpy.append(data[i], targets[i])

slicing_index = int(n_samples * 0.75)
train_data = merged_data[:slicing_index]
eval_data = merged_data[slicing_index:]

numpy.savetxt("train_data.csv", train_data, delimiter=",")
numpy.savetxt("eval_data.csv", eval_data, delimiter=",")


model = sklearn.linear_model.LinearRegression()
model.fit(data[:slicing_index], targets[:slicing_index])
predictions = model.predict(data[slicing_index:])

print("sklearn LinearRegression MSE: ")
print(sklearn.metrics.mean_squared_error(predictions, targets[slicing_index:]))
