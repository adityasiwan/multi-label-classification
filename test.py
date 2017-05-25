import numpy as np


y_actual = np.array([[0,1,0],
[0,1,1],
[1,0,1],
[0,0,1]])

y_pred = np.array([[0,1,1],
[0,1,1],
[0,1,0],
[0,0,0]])



def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i] and y_pred[i]==1:
            TP += 1
    for i in range(len(y_pred)):
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
            FP += 1
    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i] and y_pred[i]==0:
            TN += 1
    for i in range(len(y_pred)):
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
            FN += 1

    return(TP, FP, TN, FN)




""" def accuracy(self):
    return sum([self.TP(c) for c in self.classes]) / float(len(self))

    def count(self, classe):
    return sum(self._classes[classe].values())

    def TP(self, classe):
    return self._classes[classe][classe]

    def FP(self, classe):
        fp = 0
            for k,v in self._classes.items():
                if k != classe:
                    fp+=v.get(classe, 0)
    return fp

    def FN(self, classe):
        fn = 0
            for k,v in self._classes[classe].items():
                if k != classe:
                    fn+=v
    return fn

    def TN(self, classe):
    return len(self) - self.TP(classe) - self.FP(classe) - self.FN(classe) """

def precision(self):
    tp = float(self.TP)
    fp = self.FP
    return tp/(tp+fp)

def recall(self):
    tp = float(self.TP)
    if tp == 0:
        return 0

    fn= self.FN
    return tp/(tp+fn)

def fmeasure(self, beta=1.0):
    P = self.precision
    R = self.recall
    return (1.0 + beta) * (P*R) / (beta*P + R)

""" def macrofmeasure(self, beta=1.0):
    return sum([self.fmeasure(c, beta=beta) for c in self.classes])/len(self.training)

    def microfmeasure(self, beta=1.0):
    return sum([self.fmeasure(c, beta=beta)*self.training[c]/sum(self.training.values()) for c in self.training.keys()]) """


if __name__ == "__main__":
    print('fmeasure: {0}'.format(perf_measure(y_actual, y_pred)))
