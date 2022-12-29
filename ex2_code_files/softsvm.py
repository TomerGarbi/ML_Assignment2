import cvxopt
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """
    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    d = len(trainX[0])
    m = len(trainX)
    H = np.eye(d + m) * 2 * l
    for j in range(d, d + m):
        H[j, j] = 0

    u = np.ones(d + m) / m
    for j in range(d):
        u[j] = 0

    v = np.zeros(2 * m)
    for j in range(m):
        v[j] = 1

    eyes = np.vstack([np.eye(m), np.eye(m)])
    Z = np.zeros((m, d))
    X = np.array(trainX)
    for r in range(m):
        X[r] = trainy[r] * X[r]
    XZ = np.vstack([X, Z])
    A = np.hstack([XZ, eyes])
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['feastol'] = 1e-6
    sol = cvxopt.solvers.qp(matrix(H), matrix(u), matrix(-A), matrix(-v))
    return np.array(sol["x"])[:d]


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"


    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")



def select_random(trainX, trainY, size):
    X_random = []
    Y_random = []
    for i in range(size):
        random_index = np.random.randint(0, len(trainX),size=1)
        X_random.append(trainX[random_index])
        Y_random.append(trainY[random_index])
    return np.array(X_random), np.array(Y_random)


def Q2a():
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    sample_size = 100
    reps = 10
    lambdas = [L for L in range(1, 11)]
    avg_train_error = [0 for _ in lambdas]
    # max_train_error = [0 for _ in lambdas]
    # min_train_error = [1 for _ in lambdas]
    avg_test_error = [0 for _ in lambdas]
    max_test_error = [0 for _ in lambdas]
    min_test_error = [1 for _ in lambdas]

    for i in range(reps):
        indices = np.random.permutation(trainX.shape[0])
        X_exp = trainX[indices[:sample_size]]
        Y_exp = trainy[indices[:sample_size]]
        for l in lambdas:
            w = softsvm(10 ** l, X_exp, Y_exp)
            train_error_counter = 0
            for j in range(len(X_exp)):
                predicty = np.sign(X_exp[j] @ w)
                if predicty != Y_exp[j]:
                    train_error_counter += 1
            error = train_error_counter / len(X_exp)
            avg_train_error[l - 1] += error / reps
            test_error_counter = 0
            for j in range(len(testX)):
                predicty = np.sign(testX[j] @ w)
                if predicty != testy[j]:
                    test_error_counter += 1
            error = test_error_counter / len(testX)
            if max_test_error[l - 1] < error:
                max_test_error[l - 1] = error
            if min_test_error[l - 1] > error:
                min_test_error[l - 1] = error
            avg_test_error[l - 1] += error / reps
    plt.plot(lambdas, avg_test_error, color="#ff0000", label="avg test error", zorder=1)
    plt.plot(lambdas, avg_train_error, color="#0000ff", label="avg train error", zorder=1)
    plt.bar(lambdas, max_test_error, color="#000000", label="max error", zorder=1)
    plt.bar(lambdas, min_test_error, color="#00ff00", label="min error", zorder=1)
    plt.title("Q2b")
    plt.xlabel("lambda")
    plt.ylabel("error rate")


    sample_size = 1000
    reps = 10
    lambdas = [1, 3, 5, 8]
    avg_train_error = [0 for _ in lambdas]
    # max_train_error = [0 for _ in lambdas]
    # min_train_error = [1 for _ in lambdas]
    avg_test_error = [0 for _ in lambdas]
    max_test_error = [0 for _ in lambdas]
    min_test_error = [1 for _ in lambdas]
    map = {1: 0, 3: 1, 5: 2, 8: 3}
    for i in range(reps):
        indices = np.random.permutation(trainX.shape[0])
        X_exp = trainX[indices[:sample_size]]
        Y_exp = trainy[indices[:sample_size]]
        for l in lambdas:
            w = softsvm(10 ** l, X_exp, Y_exp)
            train_error_counter = 0
            for j in range(len(X_exp)):
                predicty = np.sign(X_exp[j] @ w)
                if predicty != Y_exp[j]:
                    train_error_counter += 1
            error = train_error_counter / len(X_exp)
            avg_train_error[map[l]] += error / reps
            test_error_counter = 0
            for j in range(len(testX)):
                predicty = np.sign(testX[j] @ w)
                if predicty != testy[j]:
                    test_error_counter += 1
            error = test_error_counter / len(testX)
            if max_test_error[map[l]] < error:
                max_test_error[map[l]] = error
            if min_test_error[map[l]] > error:
                min_test_error[map[l]] = error
            avg_test_error[map[l]] += error / reps

    plt.scatter(lambdas, avg_train_error, color="#0000ff", label="large sample train error", zorder=2)
    plt.scatter(lambdas, avg_test_error, color="#ff0000", label="large sample test error", zorder=2)

    plt.legend()
    plt.show()







if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    Q2a()

    # here you may add any code that uses the above functions to solve question 2
