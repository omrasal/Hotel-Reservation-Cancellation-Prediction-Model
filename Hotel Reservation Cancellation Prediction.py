#!/usr/bin/env python
# coding: utf-8

# ### MIS 637  Final Project

# ## Business Understanding
# 
# The growth of online hotel booking channels has transformed the way customers make reservations, 
# leading to an increase in cancellations and no-shows, which significantly impacts hotel revenue. Despite efforts by hotels to mitigate this issue, cancellations and no-shows remain a persistent challenge, particularly for last-minute cancellations. 
# 
# We will provide a end-end machine learning solution to help hotels maintain high occupancy rates and improve their revenue management strategies while still providing a satisfactory customer experience

# ## Data Understanding
# 
# The file contains the different attributes of customers' reservation details. The detailed data dictionary is given below
# 
# Booking_ID: unique identifier of each booking
# No of adults: Number of adults
# 
# No of children: Number of Children
# 
# noofweekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
# 
# noofweek_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
# 
# typeofmeal_plan: Type of meal plan booked by the customer:
# 
# Not Selected – No meal plan selected
# Meal Plan 1 – Breakfast
# Meal Plan 2 – Half board (breakfast and one other meal)
# Meal Plan 3 – Full board (breakfast, lunch, and dinner)
# 
# requiredcarparking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
# 
# roomtypereserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
# 
# lead_time: Number of days between the date of booking and the arrival date
# 
# arrival_year: Year of arrival date
# 
# arrival_month: Month of arrival date
# 
# arrival_date: Date of the month
# 
# Market segment type: Market segment designation.
# 
# repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
# 
# noofprevious_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
# 
# noofpreviousbookingsnot_canceled: Number of previous bookings not canceled by the customer prior to the current booking
# 
# avgpriceper_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
# 
# no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
# 
# booking_status: Flag indicating if the booking was canceled or not.

# In[3]:


get_ipython().system('pip install xgboost')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_curve,roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import graphviz


# In[5]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[6]:


#checking null values and data type
train.info()


# In[7]:


#check the duplicated value
train.drop(['booking_status', 'id'], axis =1).duplicated().sum()


# - There are 562 duplicated values in dataset after removing booking status and id
# - After further investigation, we found that each pair of duplicated value has a oppsite booking status. (example below) 
# - We decided to remove those 562 pair of duplicated value from original dataset.

# In[8]:


train[(train['id'] == 12892) | (train['id'] == 37146)]


# In[9]:


t = train.drop(['booking_status', 'id'], axis =1)
t.drop_duplicates(keep=False, inplace = True)
train = train.loc[t.index]
train.shape


# #### Exploratory Data Analysis

# In[10]:


#check the duplicated value
train.describe().T


# In[11]:


def histogram(feature, figsize=(20, 10)):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
    
    sns.boxplot(data=train, x=feature, ax=ax1)
    ax1.set_title(f"{feature} Boxplot")
    
    sns.histplot(data=train, x=feature, ax=ax2)
    mean_value = train[feature].mean()
    median_value = train[feature].median()
    ax2.axvline(mean_value, color="green", linestyle="--", label=f"Mean: {mean_value:.2f}")
    ax2.axvline(median_value, color="red", linestyle="-", label=f"Median: {median_value:.2f}")
    ax2.set_title(f"{feature} Distribution")
    ax2.legend()
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show()


# In[12]:


histogram('lead_time')


# - On average, people reserve their rooms 100 days in advance.
# - Most people book a room on the day of arrival.
# - A few people book a room 400 days in advance.

# In[13]:


histogram('avg_price_per_room')


# - Average price of all rooms is 104 USD.
# - There are some room prices of 0 USD, we will have further investigation into this.
# - There are a few outliers that exceeds 500 USD, which far away from the rest of data points.

# In[14]:


#Intuitively, a price of 0 USD for rooms might be related to the type of market segment.
train.loc[train['avg_price_per_room'] == 0,'market_segment_type'].value_counts()


# - Most of free room are type 4 in market segement and some of free room are type 1.

# In[15]:


def bar_chart(feature, binwidth = 0.25, figsize =(10,4)):
    plt.figure(figsize=figsize)
    category_counts = train[feature].value_counts()
    ax = sns.histplot(data=train, x=feature, bins = category_counts.index, binwidth = binwidth)
    ax.set_xticks(category_counts.index)
    ax.set_xticklabels(category_counts.index)
    ax.set_title(f"{feature} plot")


# In[16]:


bar_chart('market_segment_type')


# In[17]:


train['market_segment_type'].value_counts()


# - We can see that rooms with a price of 0 are highly correlated with market segment type 4. However, since there are no annotations for each market segment in this dataset, we can only assume that type 4 represents a free market.
# 

# In[18]:


bar_chart('no_of_adults')


# - most reservation are for 2 adults

# In[19]:


bar_chart('no_of_children')


# In[20]:


train['no_of_children'].value_counts()


# - Over 90% reservation is not for children
# - Less than 1% reservation is for 3+ children

# In[21]:


bar_chart('no_of_weekend_nights')


# In[22]:


bar_chart('no_of_week_nights')


# - Most reservation are for 2 days. 
# - Quite few reservations are over 1 week.

# In[23]:


bar_chart('arrival_month')


# - October was the busiest month between 2017 and 2018.

# #### Correlation Matrix

# In[24]:


corr_matrix = train.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap = 'coolwarm')
plt.show()


# - There is a moderatly positive correlation of 0.37 between lead time and booking status.
# - There is a positive correaltion of 0.59 between number of previous booking not canceled and repeated guest.
# - There is a positive correaltion of 0.45 between number of children and room type.
#     - More kids might need a bigger room

# In[25]:


bar_chart('booking_status')


# - Imblanced dataset

# ## Data Preparation

# In[26]:


X = train.drop(['booking_status','id'], axis=1)
y = train['booking_status']


# In[27]:


#get dummies for categrial variable
dummies_cat1 = pd.get_dummies(X['type_of_meal_plan'], prefix='type_of_meal_plan')
dummies_cat2 = pd.get_dummies(X['room_type_reserved'], prefix='room_type_reserved')
dummies_cat3 = pd.get_dummies(X['market_segment_type'], prefix='market_segment_type')
df_with_dummies = pd.concat([X, dummies_cat1, dummies_cat2, dummies_cat3], axis=1)
df_with_dummies = df_with_dummies.drop(['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], axis=1)


# In[28]:


X = df_with_dummies.copy()
X


# In[29]:


#split the data 
X_train , X_test , y_train , y_test = train_test_split(X, y, random_state = 42 , test_size =0.20, stratify = y)


# ## Model & Evaluation

# #### Logistic Regression

# In[30]:


logreg = LogisticRegression().fit(X_train, y_train)
y_proba = logreg.predict_proba(X_test)[:, 1]
fpr_l, tpr_l, thresholds_l = roc_curve(y_test, y_proba)
roc_auc_l = roc_auc_score(y_test, y_proba)
print("ROC AUC score of this logistic regression:", roc_auc_l)


# #### Decision Tree

# In[33]:


param_grid = {'criterion':['gini', 'entropy'],
               'max_depth': [4,6,8,10,12]
}
tree_cv = DecisionTreeClassifier(random_state=123)

grid_search = GridSearchCV(tree_cv, param_grid = param_grid, cv = 5, scoring = 'roc_auc')

grid_search.fit(X_train, y_train)

grid_search.best_params_


# In[34]:


tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 8, random_state=123)
tree.fit(X_train, y_train)
y_proba= tree.predict_proba(X_test)[:, 1]
fpr_d, tpr_d, thresholds_d = roc_curve(y_test, y_proba)
roc_auc_d = roc_auc_score(y_test, y_proba)
print("ROC AUC score of this decision tree:", roc_auc_d)


# In[35]:


import graphviz 
dot_data = export_graphviz(tree, out_file=None, 
                                feature_names=X.columns,   
                                class_names=['Not Cancel','Cancel'],
                                filled=True, rounded=True,  
                                special_characters=True)  
# visualize tree
graph = graphviz.Source(dot_data) 
graph


# #### XG Boost

# In[45]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_proba = xgb.predict_proba(X_test)[:, 1]
fpr_x, tpr_x, thresholds_x = roc_curve(y_test, y_proba)
roc_auc_x = roc_auc_score(y_test, y_proba)
print("ROC AUC score of this X gradient bossting model:", roc_auc_x)


# In[37]:


# adding a stratified Kfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scores = []

for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    
    y_proba = xgb.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    roc_auc_scores.append(roc_auc)

print("Average ROC AUC score:", np.mean(roc_auc_scores))


# In[62]:


plt.figure(figsize = (10,10))
plt.plot(fpr_d, tpr_d, label='ROC curve for Decision Tree(area = %0.2f)' % roc_auc_d, color ='r')
plt.plot(fpr_l, tpr_l, label ='ROC curve for Logistic Regression(area = %0.2f)' % roc_auc_l, color ='b')
plt.plot(fpr_x, tpr_x, label ='ROC curve for XGboost (area = %0.2f)' % roc_auc_x, color ='g')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# - The best model is the XGBoost model with default hyper-parameters.
# - The ROC-AUC score has reached 0.91, which is quite good.
# 

# In[61]:


importances = xgb.feature_importances_
idx = np.argsort(importances)

sorted_importances = importances[idx]

cumulative_importances = np.cumsum(sorted_importances)

normalized_importances = sorted_importances / cumulative_importances[-1]

fig = plt.figure(figsize=(12, 12))  
plt.barh(range(len(importances)), normalized_importances, color="violet")
plt.yticks(range(len(importances)), X_train.columns[idx], fontsize=10) 
plt.xlabel("Normalized Importance", fontsize=12)  
plt.ylabel("Feature", fontsize=12) 
plt.title("Feature Importances", fontsize=14)  
plt.show()


# - The most important factor for cancellations is when customers book hotels from market segment type 1.
# - Customers with a higher number of special requests are less likely to cancel their reservation.
# - Customers who required car parking space are less likely to cancel their reservation.
# - There is a higher chance of customers cancelling their hotel reservation if they book it too early.

# ## Deployment
