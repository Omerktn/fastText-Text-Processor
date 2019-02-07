import h5py
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import random
from sklearn.metrics import accuracy_score

vecdir = '/home/user/Documents/vectors/'
with h5py.File(vecdir + "tw_FULL.h5", 'r') as hf:
    xval = hf['tw'][:]
with h5py.File(vecdir + "tw_FULL_train.h5", 'r') as hf:
    train_ind = hf['tw'][:]
with h5py.File(vecdir + "tw_FULL_test.h5", 'r') as hf:
    test_ind = hf['tw'][:]

twNums = [3695,3184,6462,5798,1114,2768,1927,1351,1801,8259,761,1898,630]
twLabel = []
for i in range(twNums[0]):
  twLabel.append("dunya")
for i in range(twNums[1]):
  twLabel.append("ekonomi")
for i in range(twNums[2]):
  twLabel.append("genel")
for i in range(twNums[3]):
  twLabel.append("guncel")
for i in range(twNums[4]):
  twLabel.append("kultur-sanat")
for i in range(twNums[5]):
  twLabel.append("magazin")
for i in range(twNums[6]):
  twLabel.append("planet")
for i in range(twNums[7]):
  twLabel.append("saglik")
for i in range(twNums[8]):
  twLabel.append("siyaset")
for i in range(twNums[9]):
  twLabel.append("spor")
for i in range(twNums[10]):
  twLabel.append("teknoloji")
for i in range(twNums[11]):
  twLabel.append("turkiye")
for i in range(twNums[12]):
  twLabel.append("yasam")

print(len(xval))
print(len(twLabel))

# DELETE NAN VALUES
index = 0
del_list = []
inn = 0
for i in xval[:]:
    if not np.all(np.isfinite(i)):
        del_list.append(index)
    index +=1
print("found " + str(len(del_list)) + " NaN values, and deleted.")
xval = np.delete(xval,del_list,0)
yval = np.delete(twLabel,del_list)

print(len(xval))
print(len(yval))

# ORDER
x_train = [xval[index] for index in train_ind]
y_train = [yval[index] for index in train_ind]
x_test = [xval[index] for index in test_ind]
y_test = [yval[index] for index in test_ind]


clf = svm.SVC(gamma=0.001, C=100)
x = x_train[:]
y = y_train[:]
clf.fit(x,y)

y_pred = clf.predict(x_test[:])
y_true = y_test[:]

print("Accuracy:")
print(accuracy_score(y_true,y_pred))
