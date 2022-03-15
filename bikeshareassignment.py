#!/usr/bin/env python
# coding: utf-8

# ## Bike Sharing Assignment : Linear Regression

# ### Problem Statement: 
# ### A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have contracted a consulting company to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:
# ### 1. Which variables are significant in predicting the demand for shared bikes.
# ### 2. How well those variables describe the bike demands

# In[782]:


# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')


# In[783]:


# Importing the required libraries
import numpy as np
import pandas as pd
pd.get_option("display.max_columns")
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1. Reading and Understanding the 'Day' Dataset

# In[784]:


# Reading the day.csv dataset
bikeshare_df = pd.read_csv(r"C:\Users\Binoj\Desktop\day.csv")
# Displays first 5 rows by default
bikeshare_df.head()


# In[785]:


# Returns the dataframe size
bikeshare_df.shape


# In[786]:


# Displays information about the dataset
bikeshare_df.info()


# In[787]:


# Returns the number of missing values in the dataset
bikeshare_df.isnull().sum()


# In[788]:


# Displays the statistical summary
bikeshare_df.describe()


# In[789]:


# Dropping the column 'instant' from the dataset as it's a record index
bikeshare_df.drop(['instant'], axis = 1, inplace = True)


# In[790]:


# Dropping the column 'dteday' as we have this representation in yr and mnth columns
bikeshare_df.drop(['dteday'], axis = 1, inplace = True)


# In[791]:


# Dropping columns 'casual' and 'registered' as this data combined is represented by 'cnt' column
bikeshare_df.drop(['casual','registered'], axis = 1, inplace = True)


# In[792]:


# Displaying the dataset after performing drop
bikeshare_df.head()


# In[793]:


# Replacing 'season' column with appropriate values
bikeshare_df['season'].replace({1:"spring",2:"summer",3:"fall",4:"winter"}, inplace = True)

# Display first 5 rows after replacing season with actual values
bikeshare_df.head()


# In[794]:


# Replacing 'weekday' column with appropriate values 
bikeshare_df['weekday'].replace({0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thurs",5:"Fri",6:"Sat"}, inplace = True)

# Display first 5 rows after replacing weekday with appropriate values
bikeshare_df.head()


# In[795]:


# Replacing 'weathersit' column with appropriate values 
bikeshare_df['weathersit'].replace({1:"clear_partcloud",2:"mist_cloud",3:"light_rain_snow_thunderstrm",4:"heavyrain_ice_thunderstrmmist"}, inplace = True)

# Display first 5 rows after replacing weathersit with appropriate values
bikeshare_df.head()


# ## 2. Visualisation of the 'Day' Dataset

# In[796]:


# Pairplot to visualize the numeric variables
sns.pairplot(data = bikeshare_df, vars=['cnt','temp','atemp','hum','windspeed'])
plt.show()


# ## Observation: It's observed numeric variables 'temp' and 'atemp' are highly correlated to the target variable 'cnt' 

# In[797]:


# Visualising categorical Variables through boxplot
plt.figure(figsize=(15, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bikeshare_df)
plt.subplot(3,3,2)
sns.boxplot(x = 'yr', y = 'cnt', data = bikeshare_df)
plt.subplot(3,3,3)
sns.boxplot(x = 'mnth', y = 'cnt', data = bikeshare_df)
plt.subplot(3,3,4)
sns.boxplot(x = 'workingday', y = 'cnt', data = bikeshare_df)
plt.subplot(3,3,5)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bikeshare_df)
plt.subplot(3,3,6)
sns.boxplot(x = 'weekday', y = 'cnt', data = bikeshare_df)
plt.subplot(3,3,7)
sns.boxplot(x = 'holiday', y = 'cnt', data = bikeshare_df)
plt.show()


# ## Observations:
# ### 1. Fall has the highest count of riders whereas spring has the lowest count of riders.
# ### 2. Significant rise in riders from 2018 to 2019 (0: 2018 | 1:2019)
# ### 3. Under weathersit, clear_partcloud weather has the highest riders and light_rain_snow_thunderstrm has the lowest rider count
# ### 4. Month-wise, September has the higest riders count whereas January month accounts of the lowest

# In[798]:


# Checking the correlation
plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="BuPu")
plt.show()


# ## Observations
# #### 1. It's observed that correlation between temp and atemp is 0.99 ~ equals 1

# ## 3. Data preparation for Linear regression

# In[799]:


# Create dummy variables for categorical variables season, weathersit, weekday and mnth
season_dumm = pd.get_dummies(bikeshare_df['season'],drop_first = True)
weathersit_dumm = pd.get_dummies(bikeshare_df['weathersit'],drop_first = True)
weekday_dumm = pd.get_dummies(bikeshare_df['weekday'],drop_first = True)
month_dumm = pd.get_dummies(bikeshare_df['mnth'],drop_first = True)


# In[800]:


# View columns after dummy variables creation
bikeshare_df.columns


# In[801]:


# Perform concatenation for axis 1
bikeshare_df = pd.concat([bikeshare_df,season_dumm,weathersit_dumm,weekday_dumm,month_dumm], axis = 1)


# In[802]:


# Returns information on the dataset
bikeshare_df.info()


# In[803]:


# Dropping columns 'season','mnth','weekday' and 'weathersit'
bikeshare_df = bikeshare_df.drop(['season','mnth','weekday','weathersit'], axis = 1)


# In[804]:


# View the few rows of the dataset after performing drop
bikeshare_df.head()


# ## 4. Divide the data to Train and Test

# In[805]:


# Importing the required libraries
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[806]:


# Split data into train and test sets.
np.random.seed(0)
df_train, df_test = train_test_split(bikeshare_df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[807]:


# Displays the rows of the train set
df_train.head()


# In[808]:


# Displays the rows of the test set
df_test.head()


# In[809]:


# Displays the shape of the train-test sets
print(df_train.shape)
print(df_test.shape)


# In[810]:


# Perform scaling on continuous variables
scaler=MinMaxScaler()
num_vars = ['temp','atemp','hum','windspeed','cnt']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[811]:


df_train.head()


# In[812]:


# Viewing the statistical summary from Train set after scaling
df_train.describe()


# ## 5. Building a linear model

# In[813]:


# Dividing X and y sets for model building
y_train = df_train.pop('cnt')
X_train = df_train


# In[814]:


# Viewing first 5 rows by default
X_train.head()


# In[815]:


# Viewing the first 5 rows by default
y_train.head()


# In[816]:


# Adopting RFE approach for feature selection, limiting to 15 variables using mixed approach to model building.
lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm,15)
rfe.fit(X_train,y_train)


# In[817]:


# Viewing columns selected by RFE and their weights
list(zip(X_train.columns, rfe.support_,rfe.ranking_))


# In[818]:


# Viewing columns selected by RFE and manual elimination to be performed on these columns accordingly
col = X_train.columns[rfe.support_]
col


# In[819]:


# Viewing features not selected by RFE
X_train.columns[~rfe.support_]


# In[820]:


# Model building using statsmodel api calling the add_constant function
X_train_rfe = X_train[col]
X_train_rfe1 = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe1).fit()
lm.summary()


# In[821]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[822]:


# Dropping column 'hum' having high VIF
X_train_rfe = X_train_rfe.drop(['hum'], axis = 1)


# In[823]:


# Viewing the regression summary after dropping column 'hum'
X_train_rfe1 = sm.add_constant(X_train_rfe)
lm1 = sm.OLS(y_train,X_train_rfe1).fit()
lm1.summary()


# In[824]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[825]:


# Dropping column 'temp' due to higher VIF
X_train_rfe = X_train_rfe.drop(['temp'],axis = 1)


# In[826]:


# Viewing the regression summary after dropping column 'hum'
X_train_rfe2 = sm.add_constant(X_train_rfe)
lm2 = sm.OLS(y_train,X_train_rfe2).fit()
lm2.summary()


# In[827]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[828]:


# Dropping column 'windspeed' having high VIF
X_train_rfe = X_train_rfe.drop(['windspeed'],axis = 1)


# In[829]:


# Viewing the regression summary after dropping column 'windspeed'
X_train_rfe3 = sm.add_constant(X_train_rfe)
lm3 = sm.OLS(y_train,X_train_rfe3).fit()
lm3.summary()


# In[830]:


# Dropping column 'winter' with very high p-value
X_train_rfe = X_train_rfe.drop(['winter'], axis = 1)


# In[831]:


# Viewing the regression summary after dropping column 'winter'
X_train_rfe4 = sm.add_constant(X_train_rfe)
lm4 = sm.OLS(y_train,X_train_rfe4).fit()
lm4.summary()


# In[832]:


# Dropping column '4' with high p-value
X_train_rfe = X_train_rfe.drop([4], axis = 1)


# In[833]:


# Viewing the regression summary after dropping column 'winter'
X_train_rfe5 = sm.add_constant(X_train_rfe)
lm4 = sm.OLS(y_train,X_train_rfe5).fit()
lm4.summary()


# In[834]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[835]:


# Adding saturday and check if model improves
X_train_rfe['Sat'] = X_train['Sat']
X_train_rfe.head()


# In[836]:


# Viewing the regression summary after adding 'sat'
X_train_rfe6 = sm.add_constant(X_train_rfe)
lm6 = sm.OLS(y_train,X_train_rfe6).fit()
lm6.summary()


# In[837]:


# Dropping 'sat' as it has very high p-value
X_train_rfe = X_train_rfe.drop(['Sat'], axis=1)


# In[838]:


# Viewing the regression summary
X_train_rfe7 = sm.add_constant(X_train_rfe)
lm7 = sm.OLS(y_train,X_train_rfe7).fit()
lm7.summary()


# In[839]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[840]:


# Adding 'workingday' and check if model improves
X_train_rfe['workingday']=X_train['workingday']
# Displays first 5 rows by default and checks if working day is added
X_train_rfe.head()


# In[841]:


# Viewing the regression summary
X_train_rfe8 = sm.add_constant(X_train_rfe)
lm8 = sm.OLS(y_train,X_train_rfe8).fit()
lm8.summary()


# In[842]:


# Dropping 'workingday' because of high p-value
X_train_rfe = X_train_rfe.drop(['workingday'], axis=1)


# In[843]:


# Viewing the regression summary
X_train_rfe9 = sm.add_constant(X_train_rfe)
lm9 = sm.OLS(y_train,X_train_rfe9).fit()
lm9.summary()


# In[844]:


# Adding Sunday and check if model improves
X_train_rfe['Sun']=X_train['Sun']
# Displays first 5 rows by default and checks if working day is added
X_train_rfe.head()


# In[845]:


# Viewing the regression summary
X_train_rfe10 = sm.add_constant(X_train_rfe)
lm10 = sm.OLS(y_train,X_train_rfe10).fit()
lm10.summary()


# In[846]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[847]:


# Adding month 7 and check if model improves
X_train_rfe[7] = X_train[7]
# Displays first 5 rows and check if month 7 being added
X_train_rfe.head()


# In[848]:


# Viewing the regression summary
X_train_rfe11 = sm.add_constant(X_train_rfe)
lm11 = sm.OLS(y_train,X_train_rfe11).fit()
lm11.summary()


# ### Observation : Month 7 is retained as significant improvement in model

# In[849]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[850]:


# Adding month 10 and check if model improves
X_train_rfe[10] = X_train[10]
# Displays first 5 rows and check if month 10 being added
X_train_rfe.head()


# In[851]:


# Viewing the regression summary
X_train_rfe12 = sm.add_constant(X_train_rfe)
lm12 = sm.OLS(y_train,X_train_rfe12).fit()
lm12.summary()


# ### Observation : Month 10 is retained as it's addition further improves the model

# In[852]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[853]:


# Adding month 11 and check if model improves
X_train_rfe[11] = X_train[11]
# Displays first 5 rows and check if month 11 being added
X_train_rfe.head()


# In[854]:


# Viewing the regression summary
X_train_rfe13 = sm.add_constant(X_train_rfe)
lm13 = sm.OLS(y_train,X_train_rfe13).fit()
lm13.summary()


# In[855]:


# Dropping month 11 due to high p-value
X_train_rfe = X_train_rfe.drop([11], axis = 1)
X_train_rfe.head()


# In[856]:


# Viewing the regression summary
X_train_rfe14 = sm.add_constant(X_train_rfe)
lm14 = sm.OLS(y_train,X_train_rfe14).fit()
lm14.summary()


# In[857]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[858]:


# Adding month 12 and check if model improves
X_train_rfe[12] = X_train[12]
# Displays first 5 rows and check if month 12 being added
X_train_rfe.head()


# In[859]:


# Viewing the regression summary
X_train_rfe15 = sm.add_constant(X_train_rfe)
lm15 = sm.OLS(y_train,X_train_rfe15).fit()
lm15.summary()


# In[860]:


# Dropping month 12 due to high p-value
X_train_rfe = X_train_rfe.drop([12], axis = 1)


# In[861]:


# Adding summer and check if model improves
X_train_rfe['summer'] = X_train['summer']
# Displays first 5 rows and check if summer being added
X_train_rfe.head()


# In[862]:


# Viewing the regression summary
X_train_rfe16 = sm.add_constant(X_train_rfe)
lm16 = sm.OLS(y_train,X_train_rfe16).fit()
lm16.summary()


# In[863]:


# Calculating and viewing VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# ### Overall Observation:
# ### It's observed that model lm12 provides the best results amongst the models built considering different variables. Hence, we choose model lm12

# ## 6. Model Evaluation

# In[864]:


# Calculating the residuals
y_train_cnt = lm12.predict(X_train_rfe14)
res = y_train - y_train_cnt
res


# In[865]:


# Plot a histogram for error terms
fig = plt.figure()
sns.distplot((res), bins = 18)
plt.title('Error Terms')
plt.xlabel('Errors')


# ### Observation: Error terms are normally distributed and mean of the error terms is 0 

# In[866]:


# Calculating mean of the residuals
mean_res = round(np.mean(res))
mean_res


# In[867]:


bikeshare_df.head()


# In[868]:


# Scaling the test data
num_vars = ['temp','hum','windspeed','cnt']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[869]:


# Creating x and y test sets
y_test = df_test.pop('cnt')
X_test = df_test


# In[870]:


# Employ our model to make predictions. Create X_test_new dropping variables from X_test
X_train_new = X_train_rfe12.drop(['const'], axis=1)
X_test_new = X_test[X_train_new.columns]
X_test_new = sm.add_constant(X_test_new)


# In[871]:


# Making predictions on the chosen model
y_pred = lm12.predict(X_test_new)
# Checking pred vs test data
fig = plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')


# ## Observation: It's observed actual and predicted cnt i.e demand significantly overlapped, thus indicating that the model is able to explain the change in demand very well.

# In[872]:


# Returns mean squared error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[873]:


# Returns r-square calculation for test
r_sqrt = r2_score(y_test,y_pred)
(r_sqrt)*100


# ### Observation: It's observed that the above r-square output of 80.29 is almost equal to the r-square derived from our chosen model lm12 which is 79.7

# In[874]:


# Plot residuals for observing patterns, if any and check for homoscedasticity and autocorrelation
X_train_p = X_train_new.iloc[:,0]
plt.figure()
plt.scatter(X_train_p,res)
plt.xlabel('Independant Variables')
plt.ylabel('Residuals')


# ### Observation: With time series data (example: year), regression is likely to have autocorrelation as one variable is dependent on the other. Therefore, error terms for different observations will also be correlated to each other.

# In[875]:


# Viewing model lm12 regression summary which we have chosen after our model building
lm12.summary()


# # Overall Observations:
# ### 1. Demand for bikes dependant on variables yr, holiday, spring, light_rain_snow_thunderstrm, mist_cloud, 3,5,6,8,9,7,10 and Sun
# ### 2. Demand increases in months 3,5,6,8,9,7,10 and yr
# ### 3. Demand decreases on holiday, spring, sunday, mist_cloud and light_rainsnow_thunderstorm
