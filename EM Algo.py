#!/usr/bin/env python
# coding: utf-8

# In[59]:


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
import numpy as np
import pandas as pd


# In[60]:


iris = datasets.load_iris()

x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

y= pd.DataFrame(iris.target)
y.columns = ['Targets']



# In[61]:


x


# In[62]:


y


# In[67]:


colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1,3,1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Clustering')

model1 = KMeans(n_clusters=3)
model1.fit(x)

plt.subplot(1,3,2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model1.labels_], s=40)
plt.title('KMeans Clustering')
model2 = GaussianMixture(n_components=3)
model2.fit(x)

plt.subplot(1,3,3)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model2.predict(x)], s=40)
plt.title('EM Clustering')
plt.show()


# In[71]:


print("Real Target : ", iris.target)


# In[72]:


print("K-Means : ", model1.labels_)


# In[73]:


print("EM : ", model2.predict(x))


# In[78]:


print("Accuracy Score of K-Means with respect to Real clustreing  => ", sm.accuracy_score(y, model1.labels_ ) )


# In[80]:


print("Accuracy Score of EM Algo with respect to Real clustreing  => ", sm.accuracy_score(y, model2.predict(x) ) )


# In[81]:


km = sm.accuracy_score(y, model1.labels_ )
em = sm.accuracy_score(y, model2.predict(x))

if km > em:
    print("K-Means have better Clustering Capability")    
else: 
    print("EM have better Clustering Capability") 


# In[ ]:




