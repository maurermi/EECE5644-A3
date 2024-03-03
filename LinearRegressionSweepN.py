import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from data_generator import postfix
import lift

# For testing -- if defined as false skips the given question's code
Q2 = True
Q3 = True

def print_params(model):
    print("Model parameters:")
    print("\t Intercept: %3.5f" % model.intercept_,end="")
    for i,val in enumerate(model.coef_):
        print(", β%d: %3.5f" % (i,val), end="")
    print("\n")

##################################################################
######################### QUESTION 2 #############################
##################################################################


if(Q2):
    # Number of samples
    N = 1000

    # Noise variance
    sigma = 0.01

    # Feature dimension
    d = 5


    psfx = postfix(N,d,sigma)
    X = np.load("X"+psfx+".npy")
    y = np.load("y"+psfx+".npy")

    print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1)

    print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

    model = LinearRegression()

    # Train a model using a fraction of the training set each time
    print("Question 2.3")
    fracs = np.arange(0.1, 1.1, 0.1).tolist()
    RMSEs_d5 = np.zeros(shape=(len(fracs), 2))
    for i in range(len(fracs)):
        fr = fracs[i]
        X_train_i = X_train[:int(fr*X_train.shape[0]), :]
        y_train_i = y_train[:int(fr*y_train.shape[0])]


        print("Training on %f percent of the dataset..." % fr)
        model.fit(X_train_i, y_train_i)
        print("... done")

        train_rmse = rmse(y_train_i, model.predict(X_train_i))
        RMSEs_d5[i, 0] = train_rmse
        test_rmse = rmse(y_test, model.predict(X_test))
        RMSEs_d5[i, 1] = test_rmse
        print("Train rmse (fr = %f): " % fr, train_rmse)
        print("Test rmse (fr = %f): " % fr, test_rmse)
        print()

    x_vals = [fr * N for fr in fracs]
    plt.plot(x_vals, RMSEs_d5[:, 0], label='Train RMSE')
    plt.plot(x_vals, RMSEs_d5[:, 1], label='Test RMSEs')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Training Samples: d = 5")
    plt.legend()
    plt.grid()
    plt.savefig("Q2_3.png")
    # plt.show()
    print_params(model)

    print("Question 2.4")

    d = 40
    psfx = postfix(N,d,sigma)
    X = np.load("X"+psfx+".npy")
    y = np.load("y"+psfx+".npy")
    print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    fracs = np.arange(0.1, 1.1, 0.1).tolist()
    RMSEs_d40 = np.zeros(shape=(len(fracs), 2))
    for i in range(len(fracs)):
        fr = fracs[i]
        X_train_i = X_train[:int(fr*X_train.shape[0]), :]
        y_train_i = y_train[:int(fr*y_train.shape[0])]


        print("Training on %f percent of the dataset..." % fr)
        model.fit(X_train_i, y_train_i)
        print("... done")

        train_rmse = rmse(y_train_i, model.predict(X_train_i))
        RMSEs_d40[i, 0] = train_rmse
        test_rmse = rmse(y_test, model.predict(X_test))
        RMSEs_d40[i, 1] = test_rmse
        print("Train rmse (fr = %f): " % fr, train_rmse)
        print("Test rmse (fr = %f): " % fr, test_rmse)
        print()

    print_params(model)

    x_vals = [fr * N for fr in fracs]
    plt.cla()
    plt.plot(x_vals, RMSEs_d40[:, 0], label='Train RMSE')
    plt.plot(x_vals, RMSEs_d40[:, 1], label='Test RMSEs')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("RMSE vs. Training Samples: d = 40")
    plt.grid()
    plt.savefig("Q2_4.png")
    # plt.show()

    plt.cla()
    plt.plot(x_vals, RMSEs_d5[:, 0], label='Train RMSE (d=5)')
    plt.plot(x_vals, RMSEs_d5[:, 1], label='Test RMSEs (d=5)')
    plt.plot(x_vals, RMSEs_d40[:, 0], label='Train RMSE (d=40)')
    plt.plot(x_vals, RMSEs_d40[:, 1], label='Test RMSEs (d=40)')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Training Samples")
    plt.legend()
    plt.grid()
    plt.savefig("Q2_overlay.png")
    # plt.show()

    ##########################################################

    # Number of samples
    N = 10000

    # Noise variance
    sigma = 0.01

    # Feature dimension
    d = 5
    psfx = postfix(N,d,sigma)
    X = np.load("X"+psfx+".npy")
    y = np.load("y"+psfx+".npy")

    print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1)

    print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

    model = LinearRegression()

    # Train a model using a fraction of the training set each time
    print("Question 2.3 (N=10000)")
    fracs = np.arange(0.1, 1.1, 0.1).tolist()
    RMSEs_d5 = np.zeros(shape=(len(fracs), 2))
    for i in range(len(fracs)):
        fr = fracs[i]
        X_train_i = X_train[:int(fr*X_train.shape[0]), :]
        y_train_i = y_train[:int(fr*y_train.shape[0])]


        print("Training on %f percent of the dataset..." % fr)
        model.fit(X_train_i, y_train_i)
        print("... done")

        train_rmse = rmse(y_train_i, model.predict(X_train_i))
        RMSEs_d5[i, 0] = train_rmse
        test_rmse = rmse(y_test, model.predict(X_test))
        RMSEs_d5[i, 1] = test_rmse
        print("Train rmse (fr = %f): " % fr, train_rmse)
        print("Test rmse (fr = %f): " % fr, test_rmse)
        print()

    print("2.3 Coefficients:", model.coef_)
    print("2.3 Intercept:", model.intercept_)
    x_vals = [fr * N for fr in fracs]
    plt.cla()
    plt.plot(x_vals, RMSEs_d5[:, 0], label='Train RMSE')
    plt.plot(x_vals, RMSEs_d5[:, 1], label='Test RMSEs')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Training Samples: d = 5")
    plt.legend()
    plt.grid()
    plt.savefig("Q2_3_10000.png")
    # plt.show()


    print("Question 2.4")

    d = 40
    psfx = postfix(N,d,sigma)
    X = np.load("X"+psfx+".npy")
    y = np.load("y"+psfx+".npy")
    print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    fracs = np.arange(0.1, 1.1, 0.1).tolist()
    RMSEs_d40 = np.zeros(shape=(len(fracs), 2))
    for i in range(len(fracs)):
        fr = fracs[i]
        X_train_i = X_train[:int(fr*X_train.shape[0]), :]
        y_train_i = y_train[:int(fr*y_train.shape[0])]


        print("Training on %f percent of the dataset..." % fr)
        model.fit(X_train_i, y_train_i)
        print("... done")

        train_rmse = rmse(y_train_i, model.predict(X_train_i))
        RMSEs_d40[i, 0] = train_rmse
        test_rmse = rmse(y_test, model.predict(X_test))
        RMSEs_d40[i, 1] = test_rmse
        print("Train rmse (fr = %f): " % fr, train_rmse)
        print("Test rmse (fr = %f): " % fr, test_rmse)
        print()

    print("2.4 Coefficients:", model.coef_)
    print("2.4 Intercept:", model.intercept_)

    x_vals = [fr * N for fr in fracs]
    plt.cla()
    plt.plot(x_vals, RMSEs_d40[:, 0], label='Train RMSE')
    plt.plot(x_vals, RMSEs_d40[:, 1], label='Test RMSEs')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("RMSE vs. Training Samples: d = 40")
    plt.grid()
    plt.savefig("Q2_4_10000.png")
    # plt.show()


    plt.cla()
    plt.plot(x_vals, RMSEs_d5[:, 0], label='Train RMSE (d=5)')
    plt.plot(x_vals, RMSEs_d5[:, 1], label='Test RMSEs (d=5)')
    plt.plot(x_vals, RMSEs_d40[:, 0], label='Train RMSE (d=40)')
    plt.plot(x_vals, RMSEs_d40[:, 1], label='Test RMSEs (d=40)')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Training Samples")
    plt.legend()
    plt.grid()
    plt.savefig("Q2_overlay_10000.png")
    # plt.show()


##################################################################
######################### QUESTION 3 #############################
##################################################################

if(Q3):
    # Question 3
    print("Question 3")

    # Number of samples
    N = 1000
    # Noise variance
    sigma = 0.01
    # Feature dimension
    d = 5

    psfx = postfix(N,d,sigma)
    X = np.load("X"+psfx+".npy")
    y = np.load("y"+psfx+".npy")

    X = lift.liftDataset(X)
    print(np.shape(X))

    print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1)

    print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

    model = LinearRegression()

    # Train a model using a fraction of the training set each time
    print("Question 3.4")
    fracs = np.arange(0.1, 1.1, 0.1).tolist()
    RMSEs_d5 = np.zeros(shape=(len(fracs), 2))
    for i in range(len(fracs)):
        fr = fracs[i]
        X_train_i = X_train[:int(fr*X_train.shape[0]), :]
        y_train_i = y_train[:int(fr*y_train.shape[0])]


        print("Training on %f percent of the dataset..." % fr)
        model.fit(X_train_i, y_train_i)
        print("... done")

        train_rmse = rmse(y_train_i, model.predict(X_train_i))
        RMSEs_d5[i, 0] = train_rmse
        test_rmse = rmse(y_test, model.predict(X_test))
        RMSEs_d5[i, 1] = test_rmse
        print("Train rmse (fr = %f): " % fr, train_rmse)
        print("Test rmse (fr = %f): " % fr, test_rmse)
        print()


    x_vals = [fr * N for fr in fracs]
    plt.cla()
    plt.plot(x_vals, RMSEs_d5[:, 0], label='Train RMSE')
    plt.plot(x_vals, RMSEs_d5[:, 1], label='Test RMSEs')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Training Samples: d = 5")
    plt.legend()
    plt.grid()
    plt.savefig("Q3_4.png")
    # plt.show()

    print_params(model)


    ####
    print("Question 3.5")

    d = 40
    psfx = postfix(N,d,sigma)
    X = np.load("X"+psfx+".npy")
    y = np.load("y"+psfx+".npy")
    X = lift.liftDataset(X)

    print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    fracs = np.arange(0.1, 1.1, 0.1).tolist()
    RMSEs_d40 = np.zeros(shape=(len(fracs), 2))
    for i in range(len(fracs)):
        fr = fracs[i]
        X_train_i = X_train[:int(fr*X_train.shape[0]), :]
        y_train_i = y_train[:int(fr*y_train.shape[0])]


        print("Training on %f percent of the dataset..." % fr)
        model.fit(X_train_i, y_train_i)
        print("... done")

        train_rmse = rmse(y_train_i, model.predict(X_train_i))
        RMSEs_d40[i, 0] = train_rmse
        test_rmse = rmse(y_test, model.predict(X_test))
        RMSEs_d40[i, 1] = test_rmse
        print("Train rmse (fr = %f): " % fr, train_rmse)
        print("Test rmse (fr = %f): " % fr, test_rmse)
        print()

    print_params(model)

    x_vals = [fr * N for fr in fracs]
    plt.cla()
    plt.plot(x_vals, RMSEs_d40[:, 0], label='Train RMSE')
    plt.plot(x_vals, RMSEs_d40[:, 1], label='Test RMSEs')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("RMSE vs. Training Samples: d = 40")
    plt.grid()
    plt.savefig("Q3_5.png")
    # plt.show()

    plt.cla()
    plt.plot(x_vals, RMSEs_d5[:, 0], label='Train RMSE (d=5)')
    plt.plot(x_vals, RMSEs_d5[:, 1], label='Test RMSEs (d=5)')
    plt.plot(x_vals, RMSEs_d40[:, 0], label='Train RMSE (d=40)')
    plt.plot(x_vals, RMSEs_d40[:, 1], label='Test RMSEs (d=40)')
    plt.xlabel("N training samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Training Samples")
    plt.legend()
    plt.grid()
    plt.savefig("Q3_overlay.png")
    # plt.show()

