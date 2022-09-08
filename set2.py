#!/usr/bin/env python
# coding: utf-8

# # <center> **Customer Personality Analysis**

# ## Business Question

# ## Import Library

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read Dataset

# In[2]:


customer = pd.read_csv('dataset.csv')
customer.head()


# ## Data Preparation

# In[3]:


customer.info()


# In[4]:


customer.describe() 


# In[5]:


#checking duplicated data
customer.duplicated().sum()


# In[6]:


#checking missing value
customer.isnull().sum()


# In[7]:


missing_values = customer.isna().sum().to_dict()
missing_values_df = pd.DataFrame(list(missing_values.items()), columns=['Column', 'Missing_Values'])

fig = px.bar(missing_values_df,
       x = 'Column',
       y = 'Missing_Values',
       template = 'plotly_white',
       title = 'Missing Values')
fig.show()


# In[8]:


#remove missing value
customer = customer.dropna()


# ## Feature Engineering

# ### Age Customer

# In[9]:


import datetime as dt
customer['Age'] = 2015 - customer.Year_Birth


# ### Months Since Enrollment

# Dari tanggal pendaftaran pelanggan, mari kita hitung berapa bulan pelanggan berafiliasi dengan perusahaan.

# In[10]:


customer['Dt_Customer'] = pd.to_datetime(customer['Dt_Customer'])
customer['Month_Customer'] = 12.0 * (2015 - customer.Dt_Customer.dt.year ) + (1 - customer.Dt_Customer.dt.month)


# ### Total Spendings

# In[11]:


customer['TotalSpendings'] =  customer.MntWines + customer.MntFruits + customer.MntMeatProducts + customer.MntFishProducts + customer.MntSweetProducts + customer.MntGoldProds


# ### Age Groups

# In[12]:


customer.loc[(customer['Age'] >= 13) & (customer['Age'] <= 19), 'AgeGroup'] = 'Teen'
customer.loc[(customer['Age'] >= 20) & (customer['Age']<= 39), 'AgeGroup'] = 'Adult'
customer.loc[(customer['Age'] >= 40) & (customer['Age'] <= 59), 'AgeGroup'] = 'Middle Age Adult'
customer.loc[(customer['Age'] > 60), 'AgeGroup'] = 'Senior Adult'


# ### Number of Children

# In[13]:


customer['Children'] = customer['Kidhome'] + customer['Teenhome']


# ### Marital Status

# In[14]:


customer.Marital_Status = customer.Marital_Status.replace({'Together': 'Partner',
                                                           'Married': 'Partner',
                                                           'Divorced': 'Single',
                                                           'Widow': 'Single', 
                                                           'Alone': 'Single',
                                                           'Absurd': 'Single',
                                                           'YOLO': 'Single'})


# ### Removing Outliers

# In[15]:


#showing outliers "age"
plt.figure(figsize=(3,5))
sns.boxplot(y=customer.Age, color ='red');
plt.ylabel('Age', fontsize=10, labelpad=10);


# In[16]:


#showing outliers "Income"
plt.figure(figsize=(3,5))
sns.boxplot(y=customer.Income, color ='red');
plt.ylabel('Income', fontsize=10, labelpad=10);


# **INSIGHT**
# 

# In[17]:


#removing outliers
customer = customer[customer.Age < 100]
customer = customer[customer.Income < 120000]


# ## Exploratory Data Analysis

# ### Marital Status

# In[18]:


maritalstatus = customer.Marital_Status.value_counts()

fig = px.pie(maritalstatus, 
             values = maritalstatus.values, 
             names = maritalstatus.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 4)))
fig.show()  


# **INSIGHT**
# 
# 

# ### Average Spendings: Marital Status Wise

# In[19]:


maritalspending = customer.groupby('Marital_Status')['TotalSpendings'].mean().sort_values(ascending=False)
maritalspending_df = pd.DataFrame(list(maritalspending.items()), columns=['Marital Status', 'Average Spending'])

plt.figure(figsize=(13,5))
sns.barplot(data = maritalspending_df, x="Average Spending", y="Marital Status", palette='rocket');

plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Average Spending', fontsize=13, labelpad=13)
plt.ylabel('Marital Status', fontsize=13, labelpad=13);


# In[20]:


sns.boxplot(x="Marital_Status", y="TotalSpendings", data=customer, palette='rocket')


# **INSIGHT**
# 
# Meskipun minoritas, para lajang rata-rata menghabiskan lebih banyak uang dibandingkan dengan pelanggan yang memiliki mitra.

# ### Education Level

# In[21]:


education = customer.Education.value_counts()

fig = px.pie(education, 
             values = education.values, 
             names = education.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# **INSIGHT**
# - Setengah dari pelanggan adalah lulusan Universitas
# - Ada lebih banyak pelanggan yang memiliki gelar PhD daripada pelanggan yang memiliki gelar Master

# ### Child Status

# In[22]:


children = customer.Children.value_counts()

fig = px.pie(children, 
             values = children.values, 
             names = children.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# **INSIGHT**
# 

# In[23]:


fig = px.sunburst(customer, path=['Marital_Status','Education', 'Children'], values='TotalSpendings', color='Education')
fig.show()


# ### Average Spendings: Child Status Wise

# In[24]:


childrenspending = customer.groupby('Children')['TotalSpendings'].mean().sort_values(ascending=False)
childrenspending_df = pd.DataFrame(list(childrenspending.items()), columns=['No. of Children', 'Average Spending'])

plt.figure(figsize=(10,5))

sns.barplot(data=childrenspending_df,  x="No. of Children", y="Average Spending", palette='rocket_r');
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('No. of Children', fontsize=13, labelpad=13)
plt.ylabel('Average Spending', fontsize=13, labelpad=13);


# **INSIGHT**
# 

# ### Age Distribution of Customers

# In[25]:


plt.figure(figsize=(10,5))
ax = sns.histplot(data = customer.Age, color='salmon')
ax.set(title = "Age Distribution of Customers");
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Age ', fontsize=13, labelpad=13)
plt.ylabel('Counts', fontsize=13, labelpad=13);


# **INSIGHT**
# 
# Usia pelanggan hampir terdistribusi normal, dengan sebagian besar pelanggan berusia antara 40 dan 60 tahun.

# ### Relationship: Age vs Spendings

# In[26]:


plt.figure(figsize=(20,10))
sns.scatterplot(x=customer.Age, y=customer.TotalSpendings, s=100, color ='black');

plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Age', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=20, labelpad=20);


# **INSIGHT**
# 
# Tampaknya tidak ada hubungan yang jelas antara usia pelanggan dan kebiasaan belanja mereka.

# ### Customers Segmentation: Age Group Wise

# In[27]:


agegroup = customer.AgeGroup.value_counts()

fig = px.pie(labels = agegroup.index, values = agegroup.values, names = agegroup.index, width = 550, height = 550)

fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.4, 
                  marker = dict(colors = ['#3D0C02', '#800000'  , '#C11B17','#C0C0C0'], 
                                line = dict(color = 'white', width = 2)))

fig.update_layout(annotations = [dict(text = 'Age Groups', 
                                      x = 0.5, y = 0.5, font_size = 20, showarrow = False,                                       
                                      font_color = 'black')],
                  showlegend = False)

fig.show()


# **INSIGHT**
# 

# ### Average Spendings: Age Group Wise

# In[28]:


agegroupspending = customer.groupby('AgeGroup')['TotalSpendings'].mean().sort_values(ascending=False)
agegroupspending_df = pd.DataFrame(list(agegroup.items()), columns=['Age Group', 'Average Spending'])

plt.figure(figsize=(20,10))

sns.barplot(data = agegroupspending_df, x="Average Spending", y='Age Group', palette='rocket_r');
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Average Spending', fontsize=20, labelpad=20)
plt.ylabel('Age Group', fontsize=20, labelpad=20);


# **INSIGHT**
# 

# ### Income Distribution of Customers

# In[29]:


plt.figure(figsize=(10,5))
ax = sns.histplot(data = customer.Income, color = "indianred")
ax.set(title = "Income Distribution of Customers");

plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Income', fontsize=13, labelpad=13)
plt.ylabel('Counts', fontsize=13, labelpad=13);


# ### Relationship: Income vs Spendings

# In[30]:


plt.figure(figsize=(20,10))


sns.scatterplot(x=customer.Income, y=customer.TotalSpendings, s=100, color='black');

plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Income', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=20, labelpad=20);


# **INSIGHT**
# 

# ### Most Bought Products

# In[31]:


products = customer[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
product_means = products.mean(axis=0).sort_values(ascending=False)
product_means_df = pd.DataFrame(list(product_means.items()), columns=['Product', 'Average Spending'])

plt.figure(figsize=(15,10))
plt.title('Average Spending on Products')
sns.barplot(data=product_means_df, x='Product', y='Average Spending', palette='rocket_r');
plt.xlabel('Product', fontsize=20, labelpad=20)
plt.ylabel('Average Spending', fontsize=20, labelpad=20);


# **INSIGHT**
# 

# ## Machine Learning Model

# In[32]:


X = customer.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits','MntMeatProducts',
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','Dt_Customer', 'Z_CostContact',
                          'Z_Revenue', 'Recency', 'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases',
                          'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                          'AcceptedCmp1', 'AcceptedCmp2', 'Complain',  'Response', 'AgeGroup'], axis=1)


# In[33]:


X.info()


# ### Optimum Clusters Using Elbow Method

# In[34]:


from sklearn.cluster import KMeans

options = range(2,9)
inertias = []

for n_clusters in options:
    model = KMeans(n_clusters, random_state=42).fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(20,10))    
plt.title("No. of clusters vs. Inertia")
plt.plot(options, inertias, '-o', color = 'black')
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('No. of Clusters (K)', fontsize=20, labelpad=20)
plt.ylabel('Inertia', fontsize=20, labelpad=20);


# In[63]:


X


# In[62]:


import joblib
model = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(X)


joblib.dump(model,'customer.pkl')

mod = joblib.load('customer.pkl')

preds = model.predict(X)

customer_kmeans = X.copy()
customer_kmeans['clusters'] = preds


# ### Clusters Identification

# In[36]:


#Income
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Income',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Income', fontsize=50, labelpad=20);


# In[37]:


#Total Spending
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'TotalSpendings',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=50, labelpad=20);


# In[38]:


#Month Since Customer
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Month_Customer',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Month Since Customer', fontsize=50, labelpad=20);


# In[39]:


plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Age',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Age', fontsize=50, labelpad=20);


# In[40]:


plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Children',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('No. of Children', fontsize=50, labelpad=20);


# ### Data Exploration: Clusters Based

# In[41]:


customer_kmeans.clusters = customer_kmeans.clusters.replace({1: 'Group 2',
                                                             2: 'Group 3',
                                                             3: 'Group 4',
                                                             0: 'Group 1'})

customer['clusters'] = customer_kmeans.clusters


# ### Customers Distribution

# In[42]:


cluster_counts = customer.clusters.value_counts()

fig = px.pie(cluster_counts, 
             values = cluster_counts.values, 
             names = cluster_counts.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# **INSIGHT**
# - Sebagian besar pelanggan berada dalam kategori Perak dan Emas, masing-masing sekitar 29% dan 28%
# - Platinum adalah kategori pelanggan terkenal ke-3 dengan 23% sementara hanya 20% yang menempati kategori perunggu

# ### Relationship: Income vs. Spendings

# In[43]:


plt.figure(figsize=(20,10))
sns.scatterplot(data=customer, x='Income', y='TotalSpendings', hue='clusters', palette='rocket_r');
plt.xlabel('Income', fontsize=20, labelpad=20)
plt.ylabel('Total Spendings', fontsize=20, labelpad=20);


# ### Spending Habits by Clusters

# In[44]:


cluster_spendings = customer.groupby('clusters')[['MntWines', 'MntFruits','MntMeatProducts', 
                                                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()

cluster_spendings.plot(kind='bar', stacked=True, figsize=(9,7), color=['#dc4c4c','#e17070','#157394','#589cb4','#bcb4ac','#3c444c'])

plt.title('Spending Habits by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');


# ### Purchasing Habits by Clusters

# In[45]:


cluster_purchases = customer.groupby('clusters')[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                                                  'NumStorePurchases', 'NumWebVisitsMonth']].sum()

cluster_purchases.plot(kind='bar', color=['#dc4c4c','#157394','#589cb4','#bcb4ac','#3c444c'], figsize=(9,7))

plt.title('Purchasing Habits by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Purchases', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');


# ### Promotions Acceptance by Clusters

# In[46]:


cluster_campaign = customer.groupby('clusters')[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 
                                                  'AcceptedCmp5', 'Response']].sum()

plt.figure(figsize=(30,15))
cluster_campaign.plot(kind='bar', color=['#dc4c4c','#e17070','#157394','#589cb4','#bcb4ac','#3c444c'],figsize=(9,7))

plt.title('Promotions Acceptance by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Promotion Counts', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');


# In[59]:


import pickle
pickle_out = open("customer.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[60]:


get_ipython().system('pip install streamlit')


# In[61]:


import streamlit as st
from PIL import Image


# In[58]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




