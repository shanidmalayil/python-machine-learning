# coding: utf-8
from numpy import genfromtxt, zeros

#"","Name","PClass","Age","Sex","Survived","SexCode"
#"PClass","Age","SexCode","Survived"

# read the first 4 columns
data = genfromtxt('Titanic1.csv',delimiter=',',usecols=(0,1,2))
# read the fifth column
target = genfromtxt('Titanic1.csv',delimiter=',',usecols=(3),dtype=str)
print data
#print data
print target
print data.shape
print target.shape
#print type(data) # <type 'numpy.ndarray'>
#print data[0][0]
#print target[0]
print set(target) # build a collection of unique elements o/p= set([‘setosa’, ‘versicolor’, ‘virginica’])


#plot sepal_length x sepal_width  in x and y axis of 3 different classess seperatly in same plane graph
from pylab import plot, show
print data[target=='2',0] #-->data[0] where data[4]= 'setosa' 
#attributes :sepal_l,sepal_b,petal_l,petal_b,flower_class
#plot(data[target=='setosa',0],data[target=='setosa',0])
#plot(data[target=='1',0],data[target=='0',0],'go') #dead 1st class
#plot(data[target=='1',0],data[target=='2',0],'ro')
#plot(data[1],data[2],'go')
show()


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




#3.1 convert string_classnames 'setosa,ver,virg' into integer class names as 1,2 and 3, declare new 1D array as t[len],
t = zeros(len(target))
t[target == '0'] = 0
t[target == '1'] = 1
#t[target == 'virginica'] = 3
#instantiate and train classifier
from sklearn.naive_bayes import GaussianNB
data=data.reshape((1313,3))
t=t.reshape((1313))
classifier = GaussianNB()
classifier.fit(data,t) # training with iris full dataset, with 150 records

#predict method
print " 1 means survived, 0 means not survived"
print 'Classified as :', classifier.predict([data[0]])
print 'Classified as :', classifier.predict([[3,27,0]])
print 'Classified as :', classifier.predict([data[2],data[4]])

#TEST DATA-training and classiffication
#split to 60 percent-train and 40 perccent-test of total data
from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t,test_size=0.4, random_state=0)
print 'Number of records used for training',train.shape
print 'Number of records used for testing',test.shape

#train and test
classifier.fit(train,t_train) # train with 1st part:60 percent
print 'Accuracy is =',classifier.score(test,t_test) # test with 2nd part:40 percent

#CONFUSION MATRICS TO SHOW ACCURACY
from sklearn.metrics import confusion_matrix
print 'confusion matrix\n',confusion_matrix(classifier.predict(test),t_test)

#Function that gives us a complete report on the performance
from sklearn.metrics import classification_report
print classification_report(classifier.predict(test),t_test,target_names=['Survived', 'Not Survived'])

#Sophisticated evaluation model like Cross Validation. The idea behind the model is simple: the data is split into train and test sets several consecutive times and the averaged value of the prediction scores obtained with the different sets is the evaluation of the classifier
from sklearn.cross_validation import cross_val_score
# cross validation with 6 iterations
scores = cross_val_score(classifier, data, t, cv=20)
#print scores

from numpy import mean
print 'Average Accuracy from 16 iterations',mean(scores)*100,' percent'
#================ END OF CLASSIFICATION(GuassianNB)====================



#=========================4 CLUSTERING==============================
#kmeans
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=3,init='random') # initialization
kmeans.fit(data) # actual execution
c = kmeans.predict(data) # predict or do clustering now
print c
#we can evaluate the results of clustering, comparing it with the labels that we already have using the completeness and the homogeneity score
 #   homogeneity: each cluster contains only members of a single class.
 #   completeness: all members of a given class are assigned to the same cluster.

from sklearn.metrics import completeness_score, homogeneity_score
print completeness_score(t,c)
print homogeneity_score(t,c)

from pylab import figure, subplot, show,plot
figure()
subplot(211) # top figure with the real classes
plot(data[t==0,2],data[t==0,0],'bo')#p_class vs p_sex --survived
plot(data[t==1,2],data[t==1,0],'ro')#p_class vs p_sex --non survived
subplot(212) # bottom figure with classes assigned automatically
plot(data[c==1,0],data[c==1,2],'bo',alpha=.7)
plot(data[c==2,0],data[c==2,2],'go',alpha=.7)
show()
#=========================END OF CLUSTERING=========================



#=================REGRESSION-prediction using models==================
'''
#Regression is a method for investigating functional relationships among variables that can be used to make predictions.

from numpy.random import rand
x = rand(40,1) # explanatory variable
y = x*x*x+rand(40,1)/5 # depentend variable
#print x
#print y

#simple plotting
import matplotlib.pyplot as pylab
import numpy as np
pylab.xlabel('x variable _random')
pylab.ylabel('y dependent variable, y=x/5')
pylab.plot(x,y,'bo')
pylab.title('Simple plot of x and y')
pylab.show()

#calculates the best-fitting line for the observed data by minimizing the sum of the squares of the vertical deviations from each data point to the line.
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x,y)

import matplotlib.pyplot as plt
#plot this line over the actual data points to evaluate the result
from numpy import linspace, matrix
xx = linspace(0,1,40)
#plt.plot(xx,linreg.predict(matrix(xx).T),'r^')#test only
plt.plot(x,y,'g^',xx,linreg.predict(matrix(xx).T),'--r')#real lr plot on xy plane
plt.show()

#quantify how the model fits the original data using the mean squared error:T is metric measures the expected squared distance between the prediction and the true data. It is 0 when the prediction is perfect
from sklearn.metrics import mean_squared_error
print 'Mean sqrd error :', mean_squared_error(linreg.predict(x),y)
'''
#========================END REGRESSION-Prediction using models======================


'''
The overall idea of regression is to examine two things: (1) does a set of predictor variables do a good job in predicting an outcome (dependent) variable?  (2) Which variables in particular are significant predictors of the outcome variable, and in what way do they–indicated by the magnitude and sign of the beta estimates–impact the outcome variable?  These regression estimates are used to explain the relationship between one dependent variable and one or more independent variables.  The simplest form of the regression equation with one dependent and one independent variable is defined by the formula y = c + b*x, where y = estimated dependent variable score, c = constant, b = regression coefficient, and x = score on the independent variable.'''
#===========================================xxx==========================================


#========================CORRELATION===================

#***If the correlation is positive, when one variable increases, so does the other. example :hight-weight(+),examstudytime-faliurechance(- )
#**If a correlation is negative, when one variable increases, the other variable descreases. This means there is an inverse or negative relationship between the two variables. For example, as study time increases, the number of errors on an exam decreases.

#**Linear means straight line. Correlation means co-relation, or the degree that two variables "go together". Linear correlation means to go together in a straight line. The correlation coefficient is a number that summarizes the direction and degree (closeness) of linear relations between two variables. The correlation coefficient is also known as the Pearson Product-Moment Correlation Coefficient. 

#**We study the correlation to understand whether and how strongly pairs of variables are related. This kind of analysis helps us in locating the critically important variables on which others depend. The best correlation measure is the Pearson product-moment correlation coefficient. It’s obtained by dividing the covariance of the two variables by the product of their standard deviations. We can compute this index between each pair of variables for the iris dataset as follows:

#This time, sklearn provides us all we need to perform our analysis: from sklearn.decomposition import PCA ca = PCA(n_components=2) .**In the snippet above we instantiated a PCA object which we can use to compute the first two PCs. The transform is computed as follows:



from numpy import corrcoef
corr = corrcoef(data.T) # .T gives the transpose
print corr

#***The function corrcoef returns a symmetric matrix of correlation coefficients calculated from an input matrix in which rows are variables and columns are observations. Each element of the matrix represents the correlation between two variables. 
##**Correlation is positive when the values increase together. It is negative when one value decreases as the other increases. In particular we have that : 
 #1 is a perfect positive correlation, 
 #0 is no correlation
 #-1 is a perfect negative correlation.


#plot
from pylab import pcolor, colorbar, xticks, yticks,show
from numpy import arange
pcolor(corr)
colorbar() # add
# arranging the names of the variables on the axis
xticks(arange(0.5,4.5),['sepal_length', 'sepal_width', 'petal_length','petal_width'],rotation=-20)
yticks(arange(0.5,4.5),['sepal_length', 'sepal_width','petal_length','petal_width'],rotation=-20)
show()

#======================== XXX END CORRELATION XXX ==========================


#======================== XXX DIMENSION REDUCTION XXX =======================

#we have a view of only a part of the dataset. Since the maximum number of dimensions that we can plot at the same time is 3, to have a global view of the data it’s necessary to embed the whole data in a number of dimensions that we can visualize. This embedding process is called dimensionality reduction. One of the most famous techniques for dimensionality reduction is the Principal Component Analysis (PCA). This technique transforms the variables of our data into an equal or smaller number of uncorrelated variables called principal components (PCs).

from pylab import plot ,show
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pcad = pca.fit_transform(data)##first 2 principle components

plot(pcad[target=='setosa',0],pcad[target=='setosa',1],'bo')
plot(pcad[target=='versicolor',0],pcad[target=='versicolor',1],'ro')
plot(pcad[target=='virginica',0],pcad[target=='virginica',1],'go')
show()

#the separation between the versicolor specie (in red) and the virginica specie (in green) is more clear.
#The PCA projects the data into a space where the variance is maximized and we can determine how much information is stored in the PCs looking at the variance ratio:
print 'variance ratio :',pca.explained_variance_ratio_ 

#print how much information we lost during the transformation process
print 1-sum(pca.explained_variance_ratio_)


#we can apply the inverse transformation to get the original data back:
data_inv = pca.inverse_transform(pcad)
#Arguably, the inverse transformation doesn’t give us exactly the original data due to the loss of information. We can estimate how much the result of the inverse is likely to the original data as follows:
print abs(sum(sum(data - data_inv)))


#We have that the difference between the original data and the approximation computed with the inverse transform is close to zero. In this network each node represents a character of the novel and the connection between two characters represents the coappearance in the same chapter. It’s easy to see that the graph is not really helpful. Most of the details of the network are still hidden and it’s impossible to understand which are the most important nodes. In order to gain some insights about our data we can study the degree of the nodes. The degree of a node is considered one of the simplest centrality measures and it consists of the number of connections a node has. We can summarize the degrees distribution of a network looking at its maximum, minimum, median, first quartile and third quartile: It’s interesting to note how much information we can preserve by varying the number of principal components:

print 'Information preserve percentages by variing components : \n'
for i in range(1,5):
	pca = PCA(n_components=i)
	pca.fit(data)
	print 'n_Components=',i,'Sum=',sum(pca.explained_variance_ratio_) * 100,'%'
#nb:by using 3 components we can asmot get 100%

#======================== XXX END DIMENSION REDUCTION XXX ===================


#=======================NETWORK===========================

'''
import sys
import matplotlib.pyplot as plt
import networkx as nx
G = nx.read_gml('lesmiserables.gml')
print G
#G = nx.read_gml(‘lesmiserables.gml’,relabel=True)
#G = nx.read_gml('lesmiserables.gml',relabel=True)
#G = nx.read_gml('lesmiserables.gml', label='label')
nx.draw(G,node_size=0,edge_color='b',alpha=.2,font_size=7)
plt.show()



deg = nx.degree(G)
print deg
#error down blow
from numpy import percentile, mean, median
print min(deg.values())
print percentile(deg.values(),25) # computes the 1st quartile
print median(deg.values())
print percentile(deg.values(),75) # computes the 3rd quartile
print max(deg.values())

Gt = G.copy()
dn = nx.degree(Gt)
for n in Gt.nodes():
	if dn[n] <= 10:
		Gt.remove_node(n)
nx.draw(Gt,node_size=0,edge_color='b',alpha=.2,font_size=12)

from networkx import find_cliques
cliques = list(find_cliques(G))

#print max(cliques,key=lambda l: len(l))

'''
#=======================NETWORK===========================



#=================EXAMPLE NETWORK==================
'''
import sys

import matplotlib.pyplot as plt
import networkx as nx

G = nx.grid_2d_graph(5, 5)  # 5x5 grid
try:  # Python 2.6+
    nx.write_adjlist(G, sys.stdout)  # write adjacency list to screen
except TypeError:  # Python 3.x
    nx.write_adjlist(G, sys.stdout.buffer)  # write adjacency list to screen
# write edgelist to grid.edgelist
nx. write_edgelist(G, path="grid.edgelist", delimiter=":")
# read edgelist from grid.edgelist
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

nx.draw(H)
plt.show()
'''
#=================END EXAMPLE NETWORK==================


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
