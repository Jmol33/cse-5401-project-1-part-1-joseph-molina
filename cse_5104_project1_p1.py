import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def get_data(file):
    # Extract data from csv file
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    # Extract y from last column
    y = data[:, -1]
    # extract X features from all but last column
    X_features = data[:, :-1]
    # Add column for bias
    X = np.c_[np.ones(X_features.shape[0]), X_features]
    return X, y

def Standardize(X):
    # Z-score normalization
    X_features = X[:, 1:]
    mean = X_features.mean(axis=0)
    std = X_features.std(axis=0)
    X_features_normalized = (X_features - mean) / std
    return np.c_[np.ones(X_features_normalized.shape[0]), X_features_normalized], mean, std

def StandardizeTest(X, mean, std):
    # Standardize given X (testing set) using mean and std of a different X (training set)
    X_features = X[:, 1:]
    X_features_normalized = (X_features - mean) / std
    return np.c_[np.ones(X_features_normalized.shape[0]), X_features_normalized]

def LogScale(X):
    X_features = X[:, 1:]
    return np.c_[np.ones(X_features.shape[0]), np.log(X_features + 1)]

def R2(X, w, y):
    # Calculate R^2
    predictions = X @ w
    MSE = np.mean((y - predictions) ** 2)
    Variance = np.mean((y - np.mean(y)) ** 2)
    return 1 - MSE / Variance

def R2_outputs(y_pred, y):
    MSE = np.mean((y - y_pred) ** 2)
    Variance = np.mean((y - np.mean(y)) ** 2)
    return 1 - MSE / Variance

def grad_f(XTX, w, XTy, size):
    return (-2/size) * (XTy - XTX @ w)

def MSE(X, w, y):
    return np.mean((y-(X @ w))**2)

def gradient_descent(cur_w, X, y, learning_rate, precision, max_iters):
    history = []
    history.append(cur_w.copy())
    XTX = X.T @ X
    XTy = X.T @ y
    size = len(y)
    for i in range(max_iters):
        # Calculate new gradient and change weights
        grad_current = grad_f(XTX, cur_w, XTy, size)
        last_w = cur_w.copy()
        cur_w = cur_w - learning_rate * grad_current
        history.append(cur_w.copy())
        # Stop if MSE does not change enough
        if abs(MSE(X, last_w, y) - MSE(X, cur_w, y)) < precision:
            print(f"Converged after {i} iterations")
            break
    print(f"Training MSE Local Minima weights: {cur_w}")
    print(f"Training MSE: {MSE(X, cur_w, y)}")
    print(f"Training R^2: {R2(X, cur_w, y)}\n")
    return cur_w, history

def gradient_descent_positive_R2(cur_w, X, y, learning_rate, precision, max_iters):
    history = []
    history.append(cur_w.copy())
    XTX = X.T @ X
    XTy = X.T @ y
    size = len(y)
    for i in range(max_iters):
        # Calculate new gradient and change weights
        grad_current = grad_f(XTX, cur_w, XTy, size)
        last_w = cur_w.copy()
        cur_w = cur_w - learning_rate * grad_current
        history.append(cur_w.copy())
        # Stop if MSE does not change enough
        if abs(MSE(X, last_w, y) - MSE(X, cur_w, y)) < precision:
            print(f"Converged after {i} iterations")
            break
        if (R2(X, cur_w, y) > 0):
            print(f"Reached positive R^2 after {i} iterations")
            print(f"Training weights at first positive R^2: {cur_w}")
            print(f"Training MSE at first positive R^2: {MSE(X, cur_w, y)}")
            print(f"Training R^2 at first positive R^2: {R2(X, cur_w, y)}\n")
            break
    return cur_w, history

# Load the data
file = 'Concrete_Data.csv'
X, y = get_data(file)
# Split the data into training, standardized training, and testing sets
X_train = np.concatenate((X[:500],X[630:]))
X_train_standardized, X_train_mean, X_train_std = Standardize(X_train)
y_train = np.concatenate((y[:500],y[630:]))
X_test = X[500:630]
X_test_standardized = StandardizeTest(X_test, X_train_mean, X_train_std)
y_test = y[500:630]

# Initialize starting weights. I chose the 0 vector
start_w = np.zeros(X.shape[1])

# The bias term is of a much larger scale in the unstandardized, so I use a separate learning rate for it so it converges faster
multiple_learning_rates = np.array([0.1, 0.0000005, 0.0000005, 0.0000005, 0.0000005, 0.0000005, 0.0000005, 0.0000005, 0.0000005])

# Train unstandardized
weights_1, history_1 = gradient_descent(start_w.copy(), X_train, y_train, multiple_learning_rates, 1e-10, 1000000)
history_1 = np.array(history_1)
mse_1 = [np.log10(MSE(X_train, w, y_train)) for w in history_1]
iterations_1 = np.arange(len(history_1))
# Train until positive R
weights_1R, history_1R = gradient_descent_positive_R2(start_w.copy(), X_train, y_train, multiple_learning_rates, 1e-10, 1000000)
history_1R = np.array(history_1R)
mse_1R = [MSE(X_train, w, y_train) for w in history_1R]
iterations_1R = np.arange(len(history_1R))
# Train standardized
weights_2, history_2 = gradient_descent(start_w.copy(), X_train_standardized, y_train, 0.1, 1e-6, 10000)
history_2 = np.array(history_2)
mse_2 = [np.log10(MSE(X_train_standardized, w, y_train)) for w in history_2]
iterations_2 = np.arange(len(history_2))
# Train until positive R
weights_2R, history_2R = gradient_descent_positive_R2(start_w.copy(), X_train_standardized, y_train, 0.1, 1e-6, 10000)
history_2R = np.array(history_2R)
mse_2R = [MSE(X_train_standardized, w, y_train) for w in history_2R]
iterations_2R = np.arange(len(history_2R))

print(f"GD Testing MSE: {MSE(X_test, weights_1, y_test)}")
print(f"GD Testing R^2: {R2(X_test, weights_1, y_test)}")
print(f"GD Standardized Testing MSE: {MSE(X_test_standardized, weights_2, y_test)}")
print(f"GD Standardized Testing R^2: {R2(X_test_standardized, weights_2, y_test)}\n")

# Plot until converge
plt.figure(figsize=(10, 6))
plt.plot(iterations_1, mse_1, label='Unstandardized Data')
plt.plot(iterations_2, mse_2, label='Standardized Data')
plt.xlabel("Iteration")
plt.ylabel("MSE (log base 10 scale)")
plt.title("MSE over Iterations")
plt.legend()
plt.grid(True)
plt.show()
# Plot until positive R^2
plt.figure(figsize=(10, 6))
plt.plot(iterations_1R, mse_1R, label='Unstandardized Data')
plt.plot(iterations_2R, mse_2R, label='Standardized Data')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("MSE over Iterations until positive R^2")
plt.legend()
plt.grid(True)
plt.show()

# We can calculate the optimal values directly with OLS, which should be the same as using sklearn
# But this isn't a 3rd party tool
closed_form = np.linalg.inv((X_train.T @ X_train)) @ X_train.T @ y_train
print(f"Closed form sol: {closed_form}")

# Use LinearRegression from sklearn as 3rd party comparison
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)
print(f"Sklearn Regression Coefficients: {reg.coef_[1:]}")
print(f"Sklearn Regression Intercept: {reg.intercept_}")
reg.fit(X_train_standardized, y_train)
y_pred_train_standardized = reg.predict(X_train_standardized)
y_pred_test_standardized = reg.predict(X_test_standardized)
print(f"Sklearn Regression Standardized Coefficients: {reg.coef_[1:]}")
print(f"Sklearn Regression Standardized Intercept: {reg.intercept_}")
print(f"Sklearn Regression Training MSE: {np.mean((y_pred_train-y_train)**2)}")
print(f"Sklearn Regression Testing MSE: {np.mean((y_pred_test - y_test) ** 2)}")
print(f"Sklearn Regression Training R^2: {R2_outputs(y_pred_train,y_train)}")
print(f"Sklearn Regression Testing R^2: {R2_outputs(y_pred_test,y_test)}")
print(f"Sklearn Regression Standardized Training MSE: {np.mean((y_pred_train_standardized - y_train) ** 2)}")
print(f"Sklearn Regression Standardized Testing MSE: {np.mean((y_pred_test_standardized - y_test) ** 2)}")
print(f"Sklearn Regression Training R^2: {R2_outputs(y_pred_train_standardized, y_train)}")
print(f"Sklearn Regression Testing R^2: {R2_outputs(y_pred_test_standardized, y_test)}\n")

# Log scaling & training
X_train_log = LogScale(X_train)
X_test_log = LogScale(X_test)
weights_3, history_3 = gradient_descent(start_w.copy(), X_train_log, y_train, 0.005, 1e-6, 100000)
history_3 = np.array(history_3)
mse_3 = [np.log10(MSE(X_train_log, w, y_train)) for w in history_3]
iterations_3 = np.arange(len(history_3))
print(f"GD Log Testing MSE: {MSE(X_test_log, weights_3, y_test)}")
print(f"GD Log Testing R^2: {R2(X_test_log, weights_3, y_test)}")

plt.figure(figsize=(10, 6))
plt.plot(iterations_3, mse_3, label='Log(x+1) Data')
plt.xlabel("Iteration")
plt.ylabel("MSE (log base 10 scale)")
plt.title("MSE over Iterations")
plt.legend()
plt.grid(True)
plt.show()

reg.fit(X_train_log, y_train)
y_pred_train_log = reg.predict(X_train_log)
y_pred_test_log = reg.predict(X_test_log)
print(f"Sklearn Regression Log Training MSE: {np.mean((y_pred_train_log - y_train) ** 2)}")
print(f"Sklearn Regression Log Testing MSE: {np.mean((y_pred_test_log - y_test) ** 2)}")
print(f"Sklearn Regression Log Training R^2: {R2_outputs(y_pred_train_log, y_train)}")
print(f"Sklearn Regression Log Testing R^2: {R2_outputs(y_pred_test_log, y_test)}")

# Train univariate models
finding_convergence = False
start_w_univariate = np.zeros(2)
learning_rates = [0.00001, 0.00005, 0.0001, 0.00003, 0.01, 0.000001, 0.000001, 0.0001]
# [1000000, 200000, 100000, ?, 1000, 1000, ?, 100000] for convergence
# 4 needs like 800k to reach positive R^2 at this learning rate, 1000 to save time since it's not in fastest 2
max_iterations = [10, 50000, 20000, 1000, 200, 200, 1000, 20000]
reg_univariate = LinearRegression()
for i in range(1, 9):
    print(f"Feature {i}:")
    X_feature_train = X_train[:, i]
    X_univariate_train = np.c_[np.ones(X_feature_train.shape[0]), X_feature_train]
    X_feature_test = X_train[:, i]
    X_univariate_test = np.c_[np.ones(X_feature_test.shape[0]), X_feature_test]
    reg_univariate.fit(X_univariate_train, y_train)
    print(f"Univariate Coefficients: {reg_univariate.coef_}")
    print(f"Univariate Intercept: {reg_univariate.intercept_}")

    if (finding_convergence):
        weights_univariate, history_univariate = gradient_descent(start_w_univariate.copy(), X_univariate_train, y_train, learning_rates[i-1], 1e-6, max_iterations[i-1])
    # Train until positive R^2
    weights_univariate_R, history_univariate_R = gradient_descent_positive_R2(start_w_univariate.copy(), X_univariate_train,
                                                              y_train, learning_rates[i - 1], 1e-6,
                                                              max_iterations[i - 1])

# get p-values
model_untransformed = sm.OLS(y_train, X_train).fit()
model_standardized = sm.OLS(y_train, X_train_standardized).fit()
model_logarithmic = sm.OLS(y_train, X_train_log).fit()

print("Untransformed:")
print(model_untransformed.summary())
print(model_untransformed.pvalues)
print("Standardized:")
print(model_standardized.summary())
print(model_standardized.pvalues)
print("Logarithmic Scale:")
print(model_logarithmic.summary())
print(model_logarithmic.pvalues)