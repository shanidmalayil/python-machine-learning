# coding: utf-8
from numpy import genfromtxt, zeros

# read the first 4 columns
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
# read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)
# print data.shape(150,4)
print 'DATA IS :'
print data
print 'END OF DATA\n \n'
print '\n\n\ntarget is',target


#================+classifier ================================++ 
#3.1 convert string_classnames 'setosa,ver,virg' into integer class names as 1,2 and 3, declare new 1D array as t[len],
t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3
print '\n \n \nt is',t

#instantiate and train classifier
from sklearn.naive_bayes import GaussianNB
data=data.reshape((150,4))
t=t.reshape((150))
classifier = GaussianNB()  #get classifier object of GauaprioriAssianNB_Classifier
classifier.fit(data,t) # training with iris full dataset, with 150 records

#predict method
#print 'Classified as :', classifier.predict([data[0]])
#print 'Classified as :', classifier.predict([[1.2,.2,3.2,.2]])
#print 'Classified as :', classifier.predict([data[2],data[4]])
print '\n \n \n INPUT DATA TO PREDICT USING CLASSIFIER: enter 4 values of data :'
d1=float(input('d1:'))
d2=float(input('d2:'))
d3=float(input('d3:'))
d4=float(input('d4:'))
print 'By Naive Bayes Classifier, Classified as class :', classifier.predict([[d1,d2,d3,d4]])
#===================================END OF CLASSIFICATTION=======================



from sklearn import svm
clf = svm.SVC()
clf.fit(data,t)
print ' By SVM Classifier , Classified as class  ',clf.predict([[d1,d2,d3,d4]])


#==============TO TEST ACCURACY OF OUR CLASSIFIER==============================
'''
#TEST DATA-training and classiffication
#split to 60 percent-train and 40 perccent-test of total data
from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t,test_size=0.4, random_state=0)
print '\n \n Number of records used for training',train.shape
print '\n \n Number of records used for testing',test.shape

#TRAIN THE CLASSIFIER USING 60 percent data of 150 and test
classifier.fit(train,t_train) # train with 1st part:60 percent
print '\n \n Accuracy of the classifier is =',classifier.score(test,t_test) # test with 2nd part:40 percent

#CONFUSION MATRICS TO SHOW ACCURACY
from sklearn.metrics import confusion_matrix
print '\n \n \n CONFUSION MATRIX IS '
print confusion_matrix ,confusion_matrix(classifier.predict(test),t_test)

#Function that gives us a complete report on the performance
from sklearn.metrics import classification_report
print '\n \n CLASSIFICATION REPORT IS :'
print classification_report(classifier.predict(test),t_test,target_names=['setosa', 'versicolor','virginica'])

#Sophisticated evaluation model like Cross Validation. The idea behind the model is simple: the data is split into train and test sets several consecutive times and the averaged value of the prediction scores obtained with the different sets is the evaluation of the classifier
from sklearn.cross_validation import cross_val_score
# cross validation with 6 iterations
scores = cross_val_score(classifier, data, t, cv=20)
#print scores

from numpy import mean
print '\n \n \n Average Accuracy from 16 iterations',mean(scores)*100,' percent'

'''
#==============END OF  TEST ACCURACY OF OUR CLASSIFIER==============================



#GENERAL EXAMPLES
'''
#To plot simple x-y 2d relation
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.xlabel('x labels')
plt.ylabel('y labels')
plt.show()
'''



'''
# plotting continious values x,y axis, x varies continously
import matplotlib.pyplot as pylab
import numpy as np
x = np.linspace(0, 20, 2000)  # 100 evenly-spaced values from 0 to 50
y = np.sin(x)
pylab.plot(x,y)
pylab.xlim(5,15)
pylab.ylim(-1.2,1.2)
pylab.xlabel('x values')
pylab.ylabel('y =sin(x)')
pylab.show()
'''



'''
#Multiple plottings of different shapes and colour
import numpy as np
import matplotlib.pyplot as plt
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.3)
# red dashes, blue squares and green triangles
plt.plot(t, t, 'y--', t, t**2, 'bs', t, t**3, 'r^',t,t**4,'b^')
plt.show()
'''



'''

# Histogram plotting: we can plot the distribution of the first feature of our data (sepal_length) for each class:
from pylab import figure, subplot, hist, xlim, show
xmin = min(data[:,0])
xmax = max(data[:,0])
figure()
subplot(411) # distribution of the setosa class (1st, on the top)
hist(data[target=='setosa',0],color='b',alpha=.9) #data[0] where data[4]=0;
xlim(xmin,xmax)
subplot(412) # distribution of the versicolor class (2nd)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)

subplot(413) # distribution of the virginica class (3rd)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)

subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
show()


'''

'''

#import pandas as pd
#data.head()
#print target
#print data.shape
#print target.shape
#print type(data) # <type 'numpy.ndarray'>
#print data[0][0]
#print target[0]
#print set(target) # build a collection of unique elements o/p= set([‘setosa’, ‘versicolor’, ‘virginica’])

#plot sepal_length x sepal_width  in x and y axis of 3 different classess seperatly in same plane graph
#from pylab import plot, show
#print data[target=='setosa',0] --> data[0] where data[4]= 'setosa' 
#attributes :sepal_l,sepal_b,petal_l,petal_b,flower_class
#plot(data[target=='setosa',0],data[target=='setosa',0])
#plot(data[target=='setosa',0],data[target=='setosa',2],'go')
#plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
#plot(data[target=='virginica',0],data[target=='virginica',2],'bo')
#show()

'''

