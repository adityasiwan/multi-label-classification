
from sklearn.metrics import confusion_matrix
import numpy as np
y_true = np.array([(1,0,0), (1,0,0), (0,0,1), (1,0,0), (0,1,0)])
y_pred = np.array([(1,0,0), (0,1,0), (0,0,1), (0,1,0), (1,0,0)])
macrofmeasure = 0
for i in range(y_true.shape[1]):
    """ print("Col {}".format(i))
    print(confusion_matrix(y_true[:,i], y_pred[:,i]))
    print("") """
    b = confusion_matrix(y_true[:,i], y_pred[:,i])
    pi1 = b[0,0]/(b[0,0]+b[0,1])
    rho1 = b[0,0]/(b[0,0]+b[1,0])
    macrofmeasure = macrofmeasure + (2*pi1*rho1)/(pi1+rho1)


print (macrofmeasure)

a = confusion_matrix(y_true.flatten(), y_pred.flatten())
"print (a)"
pi = a[0,0]/(a[0,0]+a[0,1])
rho = a[0,0]/(a[0,0]+a[1,0])

microfmeasure = (2*pi*rho)/(pi+rho)

print (microfmeasure)
