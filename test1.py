import numpy as np


y_actual = np.array([[0,1,0],
[0,1,1],
[1,0,1],
[0,0,1]])

y_hat = np.array([[0,1,1],
[0,1,1],
[0,1,0],
[0,0,0]])



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
    for i in range(len(y_hat)):
        if y_hat[i]==1 and y_actual!=y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==0:
            TN += 1
    for i in range(len(y_hat)):
        if y_hat[i]==0 and y_actual!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)


if __name__ == "__main__":
        print('Hamming score: {0}'.format(perf_measure(y_actual, y_hat)))
