import numpy as np

y_true = np.array([[0,1,0],
                   [0,1,1],
                   [1,0,1],
                   [0,0,1]])

y_pred = np.array([[0,1,1],
                   [0,1,1],
                   [0,1,0],
                   [0,0,0]])

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

if __name__ == "__main__":
    print('Hamming score: {0}'.format(hamming_score(y_true, y_pred)))
