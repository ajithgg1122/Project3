#!/usr/bin/env python
# coding: utf-8

# # DATA DECSRIPTION

# No usiness can thrive without customers. on the flip side, customers leaving the business is a nightmare that every business owner dreads.
# In fact one of the key metrics to measure a business sucess is by measuring its customer chum rate - the lower the chum thye more loves the company is.
# 
# Dataset
# 
# The dataset cosists of parameters such as the users dempgraphic and PII detailsmembership account details, duration and frequency of thier visits to the website reoported grievances and feefdback are the like.
# The binefit  of practicing thias problem isd by using machine learning techniques are, This challenges encourage you to apply your machine learning skills to build a model that predicts a users chum score.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\AJITH JIJI\\Desktop\\AJITH JIJI\\train.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df[['Name','internet_option']]


# In[9]:


df['internet_option'].unique()


# In[10]:


df['internet_option'].value_counts()


# In[ ]:





# # HANDLING MISSING DATA

# In[11]:


df.info()


# In[12]:


#filling the catogorical column.


# In[13]:


df.isnull().sum()


# In[14]:


df['region_category'].mode()


# In[15]:


df['region_category']=df['region_category'].fillna(df['region_category'].mode()[0])


# In[16]:


df['region_category'].isnull().sum()


# In[17]:


df['preferred_offer_types'].mode()


# In[18]:


df['preferred_offer_types']=df['preferred_offer_types'].fillna(df['preferred_offer_types'].mode()[0])


# In[19]:


df['preferred_offer_types'].isnull().sum()


# In[20]:


#handling numerical missing column


# In[21]:


df['points_in_wallet'].mean()


# In[22]:


df['points_in_wallet']=df['points_in_wallet'].fillna(df['points_in_wallet'].mean())


# In[23]:


df.isnull().sum()


# In[24]:


df.describe()


# # #DATA ANALYSIS

# In[25]:


df.head()


# In[26]:


c = df.corr()
c


# In[27]:



import seaborn as sns

df1 = df.copy()



plt.figure(figsize=(16, 6))

heatmap = sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=24);


# In[28]:


df.columns


# In[29]:


df1 = df[['age', 'days_since_last_login',  'avg_time_spent', 'avg_transaction_value', 'points_in_wallet', 'churn_risk_score']]


# In[30]:



import seaborn as sns




for col in df1.columns:
    sns.displot(df1[col])


# In[31]:


#EXPLORATORY DATA ANALYSIS


# In[ ]:





# In[32]:


df['joining_date']


# In[33]:


df['joinig_date']=pd.to_datetime(df.joining_date, format="%Y/%m/%d").dt.day
df['joinig_month']=pd.to_datetime(df.joining_date, format="%Y/%m/%d").dt.month
df['joinig_year']=pd.to_datetime(df.joining_date, format="%Y/%m/%d").dt.year


# In[34]:


df.head()


# In[35]:


df.drop(["joining_date"], axis=1, inplace=True)


# In[36]:


df.head()


# In[37]:


df['last_visit_time'].head()
df.drop(columns=['past_complaint', 'complaint_status'])


# In[38]:


#to extract hour, mint and sec


# In[39]:


import pandas as pd
from datetime import datetime

# Create a DataFrame with a single column of time strings


# Define a function to convert a time string to hour, minute, and second
def extract_time(time_string):
    time_object = datetime.strptime(time_string, "%H:%M:%S")
    return pd.Series({'hour': time_object.hour, 'minute': time_object.minute, 'second': time_object.second})

# Apply the function to the 'time' column and create new columns for hour, minute, and second
df[['hour', 'minute', 'second']] = df['last_visit_time'].apply(extract_time)

df


# In[40]:


#handle catogorical columns


# In[41]:


df.info()


# In[42]:


df['gender'].value_counts()


# In[43]:


gender = pd.get_dummies(df[['gender']], drop_first=True)
gender.head()


# In[44]:


df['region_category'].value_counts()


# In[45]:


region=pd.get_dummies(df[['region_category']], drop_first=True)
region


# In[46]:


df['membership_category'].value_counts()


# In[47]:


membership =pd.get_dummies(df[['membership_category']], drop_first=True)
membership


# In[48]:


df['joined_through_referral'].value_counts()


# In[49]:


referal = pd.get_dummies(df[['joined_through_referral']], drop_first=True)
referal


# In[50]:


df['preferred_offer_types'].value_counts()


# In[51]:


offer = pd.get_dummies(df[['preferred_offer_types']], drop_first=True)
offer


# In[52]:


df['medium_of_operation'].value_counts()


# In[53]:


medium = pd.get_dummies(df[['medium_of_operation']], drop_first=True)
medium


# In[54]:


df['internet_option'].value_counts()


# In[55]:


internet = pd.get_dummies(df[['internet_option']], drop_first=True)
internet


# In[56]:


df['used_special_discount'].value_counts()


# In[57]:


discount = pd.get_dummies(df[['used_special_discount']], drop_first=True)
discount


# In[58]:


df['offer_application_preference'].value_counts()


# In[59]:


application = pd.get_dummies(df[['offer_application_preference']], drop_first=True)
application


# In[60]:


df['past_complaint'].value_counts()


# In[61]:


complaint = pd.get_dummies(df[['past_complaint']], drop_first=True)
complaint


# In[62]:


df['feedback'].value_counts()


# In[63]:


feedback = pd.get_dummies(df[['feedback']], drop_first=True)
feedback


# In[64]:


data_train=pd.concat([df, gender, region, membership, referal, offer, internet, discount, application, complaint, feedback ], axis = 1)
data_train.head()


# In[65]:


data_train.columns


# In[66]:


df['churn_risk_score']


# In[67]:


data_train.drop(["customer_id"], axis=1, inplace =True)


# In[68]:


data_train.drop(["preferred_offer_types"], axis=1, inplace=True)


# In[69]:


data_train.drop(["last_visit_time"], axis=1, inplace=True)


# In[70]:


data_train.head()


# In[71]:


data_train.info()


# In[72]:


data_train.drop(["complaint_status"], axis=1, inplace=True)

df[' internet_option_Mobile_Data'].unique()
# In[73]:


data_train.shape


# In[74]:


data_train['internet_option_Mobile_Data'].unique()


# In[75]:


data_train['churn_risk_score']


# In[76]:


data_train.drop(['Name'], axis = 1, inplace =True)


# In[77]:


data_train.columns


# In[78]:


data_train.drop(['gender','security_no'], axis=1, inplace =True)


# In[79]:


x = data_train.loc[:, ['age', 'region_category', 'membership_category',
       'joined_through_referral', 'referral_id', 'medium_of_operation',
       'internet_option', 'days_since_last_login', 'avg_time_spent',
       'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet',
       'used_special_discount', 'offer_application_preference',
       'past_complaint', 'feedback',  'joinig_date',
       'joinig_month', 'joinig_year', 'hour', 'minute', 'second', 'gender_M',
       'gender_Unknown', 'region_category_Town', 'region_category_Village',
       'membership_category_Gold Membership',
       'membership_category_No Membership',
       'membership_category_Platinum Membership',
       'membership_category_Premium Membership',
       'membership_category_Silver Membership', 'joined_through_referral_No',
       'joined_through_referral_Yes',
       'preferred_offer_types_Gift Vouchers/Coupons',
       'preferred_offer_types_Without Offers', 'internet_option_Mobile_Data',
       'internet_option_Wi-Fi', 'used_special_discount_Yes',
       'offer_application_preference_Yes', 'past_complaint_Yes',
       'feedback_Poor Customer Service', 'feedback_Poor Product Quality',
       'feedback_Poor Website', 'feedback_Products always in Stock',
       'feedback_Quality Customer Care', 'feedback_Reasonable Price',
       'feedback_Too many ads', 'feedback_User Friendly Website']]


# In[80]:


x.head()


# In[81]:


y = data_train[['churn_risk_score']]
y.head()


# In[82]:


x.drop(['used_special_discount', 'offer_application_preference', 'past_complaint', 'feedback'], axis=1, inplace =True)


# In[83]:


x.drop(['avg_frequency_login_days','internet_option', 'region_category', 'membership_category', 'joined_through_referral', 'referral_id', 'medium_of_operation'], axis=1, inplace =True)


# In[ ]:





# In[84]:


x.info()


# In[85]:


from sklearn.preprocessing import StandardScaler


# Standardizing the features
x = StandardScaler().fit_transform(x)
x


# In[86]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=13)
x = pca.fit_transform(x)
x


# In[ ]:





# In[87]:


from sklearn.model_selection import train_test_split#Importing the train_test split to split the train data into train data and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)


# In[ ]:





# In[88]:


from sklearn.preprocessing import StandardScaler#importing libraries for feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test,X_train


# # Non-Parametric Test(RandomForestClassifier)

# In[89]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 123)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(predictions, y_test)


# In[ ]:





# # Parametric Test(svm)

# In[90]:


from sklearn import svm


# In[91]:


model_svm = svm.SVC(kernel ="rbf")
model_svm.fit(X_train, y_train)
predictsvm = model_svm.predict(X_test)
accuracy_score(predictsvm, y_test)


# # SVC

# In[92]:


from sklearn.svm import SVC
# Instantiate the Support Vector Classifier (SVC)
svc = SVC(C=1.0, random_state=1, kernel='linear')
 
# Fit the model
svc.fit(X_train, y_train)


# In[93]:


predictsvc = svc.predict(X_test)
accuracy_score(predictsvc, y_test)


# In[94]:


#Gaussian Kernel Radial Basis Function (RBF)


# In[95]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel ='rbf') # radial basis function Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# In[96]:


#Polynomial Kernel


# In[97]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel ='poly') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# In[98]:


#Sigmoid Kernel


# In[99]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel ='sigmoid') # radial basis function Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)


# ## K MEANS

# In[100]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
print(kmeans.fit(X_train, y_train))
labels = kmeans.predict(X_test)
centroids = kmeans.cluster_centers_
print("labels", labels, "\n", "centroids", centroids)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
print(kmeans.fit(X_train, y_train))
labels = kmeans.predict(X_test)
centroids = kmeans.cluster_centers_
print("labels", labels, "\n", "centroids", centroids)


# In[101]:


import numpy as nm    
import matplotlib.pyplot as mtp    
import pandas as pd    
from sklearn.cluster import KMeans  
wcss_list= []   
  

for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(X_train, y_train)  
    wcss_list.append(kmeans.inertia_)  
mtp.plot(range(1, 11), wcss_list)  
mtp.title('The Elobw Method Graph')  
mtp.xlabel('Number of clusters(k)')  
mtp.ylabel('wcss_list')  
mtp.show()  


# In[102]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()


# In[ ]:





# The output you provided includes the cluster labels and the cluster centroids for each of the clusters in the k-means clustering algorithm.
# 
# The cluster labels indicate which cluster each data point belongs to. For example, the first data point was assigned to cluster 4, the second data point was assigned to cluster 2, and so on.
# 
# The cluster centroids are the mean values of the feature values for each cluster. Each centroid represents the "center" of the cluster in the feature space.
# 
# It's important to note that the interpretation of the clusters and the centroids will depend on the specific dataset and the features that were used for clustering.
# 
# In general, you can use the cluster labels and centroids to gain insights into the structure of the data. For example, you can use the centroids to identify which features are most important in distinguishing between the clusters. You can also use the cluster labels to identify which data points are similar to each other based on their feature values.

# ## ADVANTAGES

# K-means is a simple and computationally efficient algorithm, making it easy to implement on large datasets.
# It is highly scalable and can work well with high-dimensional datasets.
# K-means is a good option when the number of clusters is known in advance, as it requires the user to specify the number of clusters to be formed.
# K-means produces clusters that are easy to interpret and visualize.

# ## DISADVATAGES

# K-means assumes that clusters are spherical, equally-sized, and have similar density, which may not always be true for real-world datasets.
# The results of k-means clustering can be sensitive to the initial placement of cluster centroids, which can lead to different results when the algorithm is run multiple times.
# K-means can be sensitive to outliers, as it tries to minimize the sum of squares between data points and their assigned centroids, which can be heavily influenced by outliers.
# K-means requires a priori knowledge of the number of clusters, which can be difficult to determine in some cases.

# ## BAGGING AND BOOSTING

# In[103]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score



# Create a decision tree classifier
dtc = DecisionTreeClassifier(random_state=42)

# Create a bagging classifier using the decision tree classifier as the base estimator
bagging_clf = BaggingClassifier(base_estimator=dtc, n_estimators=10, random_state=42)

# Train the bagging classifier on the training data
bagging_clf.fit(X_train, y_train)

# Make predictions on the testing data using the bagging classifier
y_pred_bagging = bagging_clf.predict(X_test)

# Calculate the accuracy of the bagging classifier
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print("Accuracy of bagging classifier:", accuracy_bagging)

# Create an AdaBoost classifier using the decision tree classifier as the base estimator
adaboost_clf = AdaBoostClassifier(base_estimator=dtc, n_estimators=10, random_state=42)

# Train the AdaBoost classifier on the training data
adaboost_clf.fit(X_train, y_train)

# Make predictions on the testing data using the AdaBoost classifier
y_pred_adaboost = adaboost_clf.predict(X_test)

# Calculate the accuracy of the AdaBoost classifier
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print("Accuracy of AdaBoost classifier:", accuracy_adaboost)

# hyrarcial clustering



Hierarchical clustering is a type of clustering algorithm that aims to build a hierarchy of clusters, with larger clusters containing smaller ones. In this method, each data point is initially considered as its own cluster, and the algorithm then iteratively merges the closest clusters based on a similarity metric, until all data points belong to a single cluster.

There are two main types of hierarchical clustering: agglomerative and divisive. In agglomerative clustering, each data point starts as its own cluster, and clusters are successively merged until a single cluster containing all data points is obtained. In divisive clustering, all data points start in the same cluster, and the cluster is recursively divided into smaller subclusters.

Hierarchical clustering can be useful for exploring the structure of data and identifying natural groupings within it. It is commonly used in fields such as biology, sociology, and market research, among others.

# In[104]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage



# separate the target variable (churn) from the features


# perform hierarchical clustering
Z = linkage(X_test, 'ward')

# plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()


# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage



# separate the target variable (churn) from the features


# perform hierarchical clustering
Z = linkage(x, 'ward')

# plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()


# In[111]:


from sklearn.cluster import DBSCAN

# initialize DBSCAN model with epsilon and minimum sample parameters
dbscan = DBSCAN(eps=0.3, min_samples=5)

# fit the model to the data
dbscan.fit(x)

# get the labels and number of clusters
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# print the results
print('Estimated number of clusters: %d' % n_clusters)
print('Cluster labels: %s' % labels)


# In[112]:


from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# initialize DBSCAN model with epsilon and minimum sample parameters
dbscan = DBSCAN(eps=0.3, min_samples=5)

# fit the model to the train data
dbscan.fit(X_train)

# get the labels and number of clusters
train_labels = dbscan.labels_
n_clusters = len(set(train_labels)) - (1 if -1 in train_labels else 0)

# print the results
print('Estimated number of clusters: %d' % n_clusters)
print('Cluster labels: %s' % train_labels)


# In[113]:


# Scatter plot of the train data with different colors for each cluster
plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels, cmap='jet')
plt.title('DBSCAN Clustering of Train Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[114]:


# Histogram of cluster sizes for train data
unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
plt.bar(unique_train_labels, train_counts)
plt.title('Cluster Sizes for Train Data')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Points')
plt.show()



# In[115]:


# fit the model to the test data
dbscan.fit(X_test)

# get the labels and number of clusters for test data
test_labels = dbscan.labels_
n_clusters = len(set(test_labels)) - (1 if -1 in test_labels else 0)

# print the results
print('Estimated number of clusters: %d' % n_clusters)
print('Cluster labels: %s' % test_labels)

# Scatter plot of the test data with different colors for each cluster
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_labels, cmap='jet')
plt.title('DBSCAN Clustering of Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[116]:




# Histogram of cluster sizes for test data
unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)
plt.bar(unique_test_labels, test_counts)
plt.title('Cluster Sizes for Test Data')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Points')
plt.show()


# The first line imports the DBSCAN class from the sklearn.cluster module and the matplotlib.pyplot module for data visualization.
# 
# The next line creates an instance of the DBSCAN model with eps=0.3 and min_samples=5. The eps parameter determines the radius of the neighborhood around each point, and min_samples determines the minimum number of points required to form a dense region.
# 
# The next three lines fit the DBSCAN model to the X_train data, get the predicted cluster labels for each point in X_train, and calculate the number of clusters. The -1 label represents noise points that do not belong to any cluster.
# 
# The print statements output the estimated number of clusters and the predicted cluster labels.
# 
# The next line creates a scatter plot of the X_train data with different colors for each cluster. The c parameter sets the color for each point based on its predicted cluster label, and the cmap parameter specifies the colormap to use for coloring the points.
# 
# The next two lines add a title and axis labels to the plot, and the plt.show() command displays the plot.
# 
# The next two lines calculate the size of each cluster in the X_train data and create a bar plot of the cluster sizes with the cluster label on the x-axis and the number of points in the cluster on the y-axis.

# ## DB SCAN
The first line imports the DBSCAN class from the sklearn.cluster module and the matplotlib.pyplot module for data visualization.

The next line creates an instance of the DBSCAN model with eps=0.3 and min_samples=5. The eps parameter determines the radius of the neighborhood around each point, and min_samples determines the minimum number of points required to form a dense region.

The next three lines fit the DBSCAN model to the X_train data, get the predicted cluster labels for each point in X_train, and calculate the number of clusters. The -1 label represents noise points that do not belong to any cluster.

The print statements output the estimated number of clusters and the predicted cluster labels.

The next line creates a scatter plot of the X_train data with different colors for each cluster. The c parameter sets the color for each point based on its predicted cluster label, and the cmap parameter specifies the colormap to use for coloring the points.

The next two lines add a title and axis labels to the plot, and the plt.show() command displays the plot.

The next two lines calculate the size of each cluster in the X_train data and create a bar plot of the cluster sizes with the cluster label on the x-axis and the number of points in the cluster on the y-axis.





DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular unsupervised clustering algorithm used in machine learning to identify clusters of data points in a dataset. It works by grouping together points that are closely packed together, while also identifying points that are isolated and do not belong to any clusters.

The algorithm operates by defining a neighborhood around each point in the dataset, and then grouping together points that are within a certain distance of each other. Points that are close together are considered to be part of the same cluster, while points that are far apart from each other are considered to be outliers or noise.

One of the main advantages of DBSCAN is its ability to identify clusters of arbitrary shape, unlike other clustering algorithms such as K-means which assumes that clusters are spherical. Additionally, DBSCAN does not require the user to specify the number of clusters beforehand, as it is able to determine the optimal number of clusters based on the density of the data.

Overall, DBSCAN is a powerful and flexible clustering algorithm that can be used in a wide range of applications, including image segmentation, anomaly detection, and customer segmentation in marketing.
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




