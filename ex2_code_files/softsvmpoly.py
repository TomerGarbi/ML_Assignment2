import cvxopt
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt



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
    return np.sign(sum(alpha[j] * K(trainX[j], x, k) for j in range(len(trainX))))

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
    epsilon = np.finfo(float).eps
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

    sol = cvxopt.solvers.qp(matrix(H), matrix(u), matrix(-A), matrix(-v))
    print(np.array(sol["x"])[:m].shape)
    return np.array(sol["x"])[:m]


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
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
    assert w.shape[0] == 100 and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

    c = 0
    for i in range(100):

        # get a random example from the test set, and classify it
        i = np.random.randint(0, testX.shape[0])
        predicty = predict(_trainX, testX[i], 5, w)

        # this line should print the classification of the i'th test sample (1 or -1).
        print(f"The {i}'th test sample was classified as {predicty}")
        if predicty != testy[i]:
            c += 1
    print("err: ", c / 100)


def Q4A():
    data = np.load("ex2q4_data.npz")
    X_train = data["Xtrain"]
    Y_train = data["Ytrain"]
    X_test = data["Xtest"]
    Y_test = data["Ytest"]
    plot(X_test, Y_test, ".")








if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    Q4A()
    # here you may add any code that uses the above functions to solve question 4
