import cvxopt
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt

import softsvm


def plot(data, labels, title):
    groups = []
    ind = 0
    labels_seen = []
    N = len(data)
    while(ind < N):
        label = labels[ind]
        if label in labels_seen:
            groups[labels_seen.index(label)].append(data[ind])
            ind = ind + 1
        else:
            labels_seen.append(label)
            groups.append([])
    plot_groups(groups, title)


def plot_groups(groups, title):
    i = 0
    while i < len(groups):
        x = []
        y = []
        j = 0
        while j < len(groups[i]):
            point = groups[i][j]
            x.append(point[0])
            y.append(point[1])
            j = j + 1
        plt.plot(x, y, ".")
        i = i + 1
    plt.title(title)
    plt.show()


def K(Xi, Xj, k):
    return (1 + Xi @ Xj) ** k


def predict(trainX, x, k, alpha):
    return np.sign(sum([alpha[j] * K(trainX[j], x, k) for j in range(len(trainX))]))


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = len(trainX)
    d = len(trainX[0])
    G = np.zeros((m, m))
    # calculate G matrix:
    for i in range(m):
        for j in range(m):
            G[i][j] = K(trainX[i], trainX[j], k)
    zeroMM = np.zeros((m, m))
    epsilon = 1e-04
    H = 2 * l * np.hstack([np.vstack([G, zeroMM]), np.vstack([zeroMM, zeroMM])])
    H = H + np.eye(len(H)) * epsilon
    v = np.hstack([np.ones(m), np.zeros(m)])
    u = np.hstack([np.zeros(m), (1 / m) * np.ones(m)])

    eyes = np.vstack([np.eye(m), np.eye(m)])
    X = np.array(G)
    for r in range(m):
        X[r] = trainy[r] * X[r]
    XZ = np.vstack([X, zeroMM])
    A = np.hstack([XZ, eyes])
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['feastol'] = 1e-6
    sol = cvxopt.solvers.qp(matrix(H), matrix(u), matrix(-A), matrix(-v))
    return np.array(sol["x"])[:m]


def simple_test():
    # load question 2 data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

    c = 0
    for j in range(1000):
        # get a random example from the test set, and classify it
        i = np.random.randint(0, testX.shape[0])
        predicty = predict(_trainX, testX[i], 5, w)

        # this line should print the classification of the i'th test sample (1 or -1).
        print(f"The {i}'th test sample was classified as {predicty}")
        if predicty != testy[i]:
            c += 1
    print("err: ", c / 1000)


def Q4A():
    data = np.load("ex2q4_data.npz")
    X_train = data["Xtrain"]
    Y_train = data["Ytrain"]
    X_test = data["Xtest"]
    Y_test = data["Ytest"]
    plot(X_train, Y_train, "Q4a")
    simple_test()




def divide(X_train: np.ndarray, k):
    m = len(X_train)
    indices = np.array([i for i in range(m)])
    np.random.shuffle(indices)
    X_divided = []
    for i in range(k - 1):
        X_divided.append(indices[i * (m//k):(i+1) * (m // k)])
    X_divided.append(indices[(k - 1) * (m // k):])
    return X_divided


def calculate_kernel_error(alpha, k, X_train, X_test, Y_test):
    c = 0
    m = len(X_test)
    for i in range(m):
        predicty = predict(X_train, X_test[i], k, alpha)
        if predicty != Y_test[i]:
            c += 1
    return c / m


def Q4b_kernel():
    data = np.load("ex2q4_data.npz")
    X_train = data["Xtrain"]
    Y_train = data["Ytrain"]
    X_test = data["Xtest"]
    Y_test = data["Ytest"]
    K_FOLD = 5
    X_train_divided = divide(X_train, K_FOLD)
    lambdas = [1, 10, 100]
    ks = [2, 5, 8]
    m, d = X_train.shape
    total_errors = []
    for l in lambdas:
        for k in ks:
            total_err = 0
            for i in range(K_FOLD):
                V = np.array([X_train[l] for l in X_train_divided[i]])
                Y_V = np.array([Y_train[l] for l in X_train_divided[i]])
                _S = []
                _Y = []
                for j in range(K_FOLD):
                    if j != i:
                        _S.extend([X_train[l] for l in X_train_divided[j]])
                        _Y.extend([Y_train[l] for l in X_train_divided[j]])
                _S = np.array(_S)
                _Y = np.array(_Y)
                alpha = softsvmpoly(l, k, _S, _Y)
                err = calculate_kernel_error(alpha, k, _S, V, Y_V)
                total_err += err
                # print(f"lambda = {l}, k={k}, err={err}")
            print(f"lambda = {l}, k={k}, avg_err={total_err / K_FOLD}")
            total_errors.append((l, k, total_err / K_FOLD))
    (min_l, min_k, _) = min(total_errors, key=lambda e: e[2])
    print(f"min l: {min_l}, min k: {min_k}")
    alpha = softsvmpoly(min_l, min_k, X_train, Y_train)
    print(f"minimal error: {calculate_kernel_error(alpha, min_k, X_train, X_test, Y_test)}")
    return alpha


def calculate_soft_error(w, X_test, Y_test):
    c = 0
    m = len(X_test)
    for i in range(m):
        predicty = np.sign(X_test[i] @ w)
        if predicty != Y_test[i]:
            c += 1
    return c / m


def Q4b_soft():
    data = np.load("ex2q4_data.npz")
    X_train = data["Xtrain"]
    Y_train = data["Ytrain"]
    X_test = data["Xtest"]
    Y_test = data["Ytest"]
    K_FOLD = 5
    X_train_divided = divide(X_train, K_FOLD)
    lambdas = [1, 10, 100, 10000]
    m, d = X_train.shape
    total_errors = []
    for l in lambdas:
        total_err = 0
        for i in range(K_FOLD):
            V = np.array([X_train[l] for l in X_train_divided[i]])
            Y_V = np.array([Y_train[l] for l in X_train_divided[i]])
            _S = []
            _Y = []
            for j in range(K_FOLD):
                if j != i:
                    _S.extend([X_train[l] for l in X_train_divided[j]])
                    _Y.extend([Y_train[l] for l in X_train_divided[j]])
            _S = np.array(_S)
            _Y = np.array(_Y)
            w = softsvm.softsvm(l, _S, _Y)
            err = calculate_soft_error(w, V, Y_V)
            total_err += err
        print(f"lambda = {l} avg_err={total_err / K_FOLD}")
        total_errors.append((l, total_err / K_FOLD))
    (min_l, _) = min(total_errors, key=lambda e: e[1])
    print(f"min l: {min_l}")
    w = softsvm.softsvm(min_l, X_train, Y_train)
    print(f"minimal error: {calculate_soft_error(w, X_test, Y_test)}")
    print(w)
    return w


def Q4e():
    data = np.load("ex2q4_data.npz")
    X_train = data["Xtrain"]
    Y_train = data["Ytrain"]
    X_test = data["Xtest"]
    Y_test = data["Ytest"]
    l = 1
    ks = [3, 5, 8]
    for k in ks:
        alpha = softsvmpoly(l, k, X_train, Y_train)
        img = np.zeros((200, 200))
        for i in range(200):
            for j in range(200):
                x = (i - 100) / 100, (-j + 100) / 100
                img[i][j] = predict(X_train, x, k, alpha) + 1
        plt.title(f"k = {k}")
        plt.imshow(img.T)
        plt.show()



from scipy.special import factorial



def multinomial_coeff(K, c):
    return factorial(K) / factorial(c).prod()


def Q4F():
    data = np.load("ex2q4_data.npz")
    X_train = data["Xtrain"]
    Y_train = data["Ytrain"]
    X_test = data["Xtest"]
    Y_test = data["Ytest"]
    l = 1
    K = 5
    T = []
    for i in range(6):
        for j in range(6 - i):
            T.append(np.array([5 - i - j, j, i]))
    w = np.zeros(len(T))
    alpha = softsvmpoly(l, K, X_train, Y_train)

    for i_x, x in enumerate(X_train):
        psi_x = np.zeros(len(T))
        for i_t, t in enumerate(T):
            bkt = multinomial_coeff(K, t)
            psi_x[i_t] = np.sqrt(bkt) * x[0] ** t[1] * x[1] ** t[2]
        w += alpha[i_x] * psi_x
    print(T)
    print(w)
    Y_w = np.zeros(len(Y_train))
    for i in range(len(X_train)):
        x = X_train[i]
        psi_x = np.zeros(len(T))
        for i_t, t in enumerate(T):
            bkt = multinomial_coeff(K, t)
            psi_x[i_t] = np.sqrt(bkt) * x[0] ** t[1] * x[1] ** t[2]
        w_pred = np.sign(w @ psi_x)
        Y_w[i] = w_pred

    plot(X_train, Y_w, "predictions by W")

    for i_t,t in enumerate(T):
        print(f"{'-' if np.sign(w[i_t]) < 0 else '+'}{abs(w[i_t])} * {multinomial_coeff(5, t)} * x(1)^{t[1]} * x(2) ^ {t[2]}")














if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    Q4F()
    # here you may add any code that uses the above functions to solve question 4
