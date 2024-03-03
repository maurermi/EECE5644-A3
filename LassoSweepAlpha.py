import numpy as np
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix
import lift


def print_params(model):
    print("Model parameters:")
    print("\t Intercept: %3.5f" % model.intercept_,end="")
    for i,val in enumerate(model.coef_):
        print(", β%d: %3.5f" % (i,val), end="")
    print("\n")

N = 1000 # Number of samples
sigma = 0.01 # Noise variance
d = 40 # Feature dimension

psfx = postfix(N,d,sigma)

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

# Lift the dataset
X = lift.liftDataset(X)

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))


# Values of alpha under test
alphas = [2**i for i in range(-20,11)]

mean_RMSEs = []
std_RMSEs = []
coeffs = []
for alpha in alphas:
        model = Lasso(alpha = alpha)
        # Cross validation
        cv = KFold(
                n_splits=5,
                random_state=42,
                shuffle=True
                )
        scores = cross_val_score(
                model, X_train, y_train, cv=cv,scoring="neg_root_mean_squared_error")
        mean_RMSEs.append(-np.mean(scores))
        std_RMSEs.append(np.std(scores))
        print("Cross-validation RMSE for α=%f : %f ± %f" % (alpha,-np.mean(scores),np.std(scores)) )
        model.fit(X_train, y_train)
        coeffs.append(model.coef_)


plt.errorbar(alphas, mean_RMSEs, yerr=std_RMSEs, fmt='o')
plt.xscale('log')
plt.xlabel("α")
plt.ylabel("Mean RMSE")
plt.title("Mean RMSE vs. α (Lasso Regression)")
plt.savefig("Q4.png")
# plt.show()
plt.cla()

for i in range(len(coeffs[0])):
    coeff_data = [coeff[i] for coeff in coeffs]
    plt.plot(alphas, coeff_data,label="feature %d" % i)

#plt.legend()
plt.xscale('log')
plt.xlabel("α")
plt.ylabel("Coefficient of feature")
plt.savefig("coefficients_vs_alpha.png")

best_alpha = alphas[np.argmin(mean_RMSEs)]
best_mean_RMSE = np.min(mean_RMSEs)

# Get the best alpha
print("Best alpha,", best_alpha, "gives a RMSE of", best_mean_RMSE)
model = Lasso(alpha=best_alpha)
print("Fitting linear model over entire training set...",end="")
model.fit(X_train, y_train)
print(" done")

# print("Coefficients of optimal model:", model.coef_)
# print("Intercept of optimal model:", model.intercept_)

# Compute RMSE
rmse_train = rmse(y_train,model.predict(X_train))
rmse_test = rmse(y_test,model.predict(X_test))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))

print_params(model)
