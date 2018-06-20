import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas

filename = 'midterm_data.csv'
head = ['X1','X2','Class']
fptr = open(filename,'w', newline='')
writer = csv.writer(fptr,delimiter=',')
writer.writerow(head)
    
    
N=100
x0=0.5*np.random.randn(2,N)

fig = plt.figure()  #  axis([-10 10 -10 10]),hold on;
ax0 = fig.add_subplot(1,1,1)
ax0.set_xlim(-10., 10.)
ax0.set_ylim(-10., 10.)

x1=np.zeros((2,N))
x2=np.zeros((2,N))
x3=np.zeros((2,N))

for j in range(N):
   x1[:,j] = x0[:,j] + [-1,1]
   x2[:,j] = x0[:,j] + [2,3]
   x3[:,j] = x0[:,j] + [2,6]
#
#
ax0.plot(x1[0,:],x1[1,:],'r+')
ax0.plot(x2[0,:],x2[1,:],'bo')
ax0.plot(x3[0,:],x3[1,:],'g*')


for j in range(N):
   writer.writerow([x1[0,j],x1[1,j],'1'])
for j in range(N):
   writer.writerow([x2[0,j],x2[1,j],'2'])
for j in range(N):
   writer.writerow([x3[0,j],x3[1,j],'3'])


fptr.close()


#Preprocessing
import pandas as pd

df = pd.read_csv(filename, names=["X1", "X2", "Class"], header=0)
X = df.iloc[:,0:2]
y = df.iloc[:,2]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

