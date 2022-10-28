import pickle
from sklearn import metrics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics


df = pd.read_csv("rainfall in india 1901-2015.csv")
# df.head()


# ## Data Exploration and Pre-Processing

# In[348]:


# df.info()


# In[349]:


# df.describe()


# In[350]:


# df.isnull().sum()


# In[351]:


# df.duplicated().sum()


# In[352]:


# df['SUBDIVISION'].value_counts()


# In[353]:


# df.mean(numeric_only=True)


# In[354]:


# filling na values with mean
df = df.fillna(df.mean(numeric_only=True))


# In[355]:


# df.head(25)


# In[356]:


# df.isnull().any()


# In[357]:


# df.YEAR.unique()


# In[358]:


# df.shape


# ## Data Visualization

# In[359]:


# sns.pairplot(df)


# In[360]:


# plt.figure(figsize=(15,6))
# sns.heatmap(df.corr(),annot=True)
# plt.show()


# `The above heatmap shows the coorelation between different features in the dataset`

# In[361]:


# df[["SUBDIVISION","ANNUAL"]].groupby("SUBDIVISION").sum().sort_values(by='ANNUAL',ascending=False).plot(kind='barh',stacked=True,figsize=(15,10))
# plt.xlabel("Rainfall in MM",size=12)
# plt.ylabel("Sub-Division",size=12)
# plt.title("Annual Rainfall v/s SubDivisions")
# plt.grid(axis="x",linestyle="-.")
# plt.show()


# In[362]:


# plt.figure(figsize=(15,8))
# df.groupby("YEAR").sum()['ANNUAL'].plot(kind="line",color="r",marker=".")
# plt.xlabel("YEARS",size=12)
# plt.ylabel("RAINFALL IN MM",size=12)
# plt.grid(axis="both",linestyle="-.")
# plt.title("Rainfall over Years")
# plt.show()


# In[363]:


# df[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP',
#       'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(kind="line",figsize=(18,8))
# plt.xlabel("Year",size=13)
# plt.ylabel("Rainfall in MM",size=13)
# plt.title("Year v/s Rainfall in each month",size=20)
# plt.show()


# In[364]:


# df[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
#        'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").sum().plot(kind="barh",stacked=True,figsize=(13,8))
# plt.title("Sub-Division v/s Rainfall in each month")
# plt.xlabel("Rainfall in MM",size=12)
# plt.ylabel("Sub-Division",size=12)
# plt.grid(axis="x",linestyle="-.")
# plt.show()


# ### Analysis of rainfall data of VIDARBHA

# In[365]:


# V = df.loc[((df['SUBDIVISION'] == 'VIDARBHA'))]
# V.head(5)


# In[366]:


# plt.figure(figsize=(10,6))
# V[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN','JUL','AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot(kind="bar",width=0.5,linewidth=2)
# plt.title("VIDARBHA Rainfall v/s Months",size=14)
# plt.xlabel("Months",size=14)
# plt.ylabel("Rainfall in MM",size=14)
# plt.grid(axis="both",linestyle="-.")
# plt.show()


# `From the above graph we observe that:-
#  1)Tamil Nadu has good amount of rainfall in JUL and AUG`

# In[367]:


# V.groupby("YEAR").sum()['ANNUAL'].plot(ylim=(50,1800),color='r',marker='o',linestyle='-',linewidth=2,figsize=(20,8));
# plt.xlabel('Year',size=14)
# plt.ylabel('Rainfall in MM',size=14)
# plt.title('VIDARBHA Annual Rainfall from Year 1901 to 2015',size=20)
# plt.grid()
# plt.show()


# `From the Above graph we observe that:-
# (1)The lowest rainfall in VIDARBHA was noted in 1920
# (2)and, The highest Rainfall was noted in 1958`

# ## Modelling

# In[368]:


# df["SUBDIVISION"].nunique()


# In[369]:


# group = df.groupby('SUBDIVISION')['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
# df=group.get_group(('VIDARBHA'))
# df.head()


# In[370]:


df2=df.melt(['YEAR']).reset_index()
# df2.head()


# In[371]:


df2= df2[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
# df2.head()


# In[372]:


# df2.YEAR.unique()


# In[373]:


df2.columns=['Index','Year','Month','Avg_Rainfall']


# In[374]:


# df2.head()


# In[375]:


Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
df2['Month']=df2['Month'].map(Month_map)
# df2.head(12)


# In[376]:


df2.drop(columns="Index",inplace=True)


# In[377]:


# df2.head(2)


# In[378]:


# df2.groupby("Year").sum().plot()
# plt.show()


# In[379]:


X=np.asanyarray(df2[['Year','Month']]).astype('int')
y=np.asanyarray(df2['Avg_Rainfall']).astype('int')


# In[380]:


# X


# In[381]:


# X[:15]


# In[382]:


# y


# In[383]:


# print(X.shape)
# print(y.shape)


# In[428]:


# splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[429]:


# X_train


# In[430]:


# y_train


# ### ❏ Linear Regression Model

# In[431]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)


# In[432]:


# predicting 
# y_train_predict=LR.predict(X_train)
# y_test_predict=LR.predict(X_test)


# In[433]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
# print('Explained Variance Score:', metrics.explained_variance_score(y_test, y_test_predict),2)

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))

# print("\n-----Training Accuracy-------")
# print(round(LR.score(X_train,y_train),3)*100)
# print("-----Testing Accuracy--------")
# print(round(LR.score(X_test,y_test),3)*100)


# In[434]:


# predicted = LR.predict([[2014,2]])


# In[435]:


# predicted


# ### ❏ Random Forest Model

# In[436]:


from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                      min_samples_split=10, n_estimators=800)
random_forest_model.fit(X_train, y_train)


# In[437]:


# y_train_predict=random_forest_model.predict(X_train)
# y_test_predict=random_forest_model.predict(X_test)


# In[438]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# In[439]:


# print("-----------Training Accuracy------------")
# print(round(random_forest_model.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(random_forest_model.score(X_test,y_test),3)*100)


# # In[440]:


# predicted = random_forest_model.predict([[2014,2]])


# # In[441]:


# predicted


# In[442]:


# predicted = random_forest_model.predict([[2001,3]])


# In[443]:


# predicted


# ### ❏ SVM

# In[444]:


from sklearn import svm
svm_regr = svm.SVC(kernel='rbf')
svm_regr.fit(X_train, y_train)


# In[445]:


# y_train_predict=svm_regr.predict(X_train)
# y_test_predict=svm_regr.predict(X_test)


# In[446]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# In[447]:


# print("-----------Training Accuracy------------")
# print(round(svm_regr.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(svm_regr.score(X_test,y_test),3)*100)


# ### ❏ Logistic Regression

# In[448]:


from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression(random_state=0,solver='lbfgs',class_weight='balanced', max_iter=10000)
logreg = LogisticRegression(random_state=0,solver='lbfgs')
logreg.fit(X_train,y_train)


# In[449]:


# # y_train_predict=logreg.predict(X_train)
# # y_test_predict=logreg.predict(X_test)


# # In[450]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# # # In[461]:


# print("-----------Training Accuracy------------")
# print(round(logreg.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(logreg.score(X_test,y_test),3)*100)


# ### ❏ xgboost

# In[462]:


from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)


# In[463]:


# y_train_predict=xgb.predict(X_train)
# y_test_predict=xgb.predict(X_test)


# In[464]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# In[465]:


# print("-----------Training Accuracy------------")
# print(round(xgb.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(xgb.score(X_test,y_test),3)*100)


# ### ❏ Gradient Boosting Regressor

# In[514]:


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=0)
gbr.fit(X_train, y_train)


# In[515]:


# y_train_predict=gbr.predict(X_train)
# y_test_predict=gbr.predict(X_test)


# In[516]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# In[517]:


# print("-----------Training Accuracy------------")
# print(round(gbr.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(gbr.score(X_test,y_test),3)*100)


# # Ensemble Stacking

# ### ❏ Hybrid Model 1

# The stacked model with meta learner = xgboost and the weak learners = Linear Regression, Random Forest and SVM

# In[466]:


from mlxtend.regressor import StackingCVRegressor


# In[467]:


stack = StackingCVRegressor(regressors=(LR, random_forest_model, svm_regr),
                            meta_regressor=xgb, cv=12,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack.fit(X_train, y_train)


# In[468]:


# y_train_predict=stack.predict(X_train)
# y_test_predict=stack.predict(X_test)


# In[469]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# # In[470]:


# print("-----------Training Accuracy------------")
# print(round(stack.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(stack.score(X_test,y_test),3)*100)


# ### ❏ Hybrid Model 2

# The stacked model with meta learner = Linear Regression and the weak learners = Linear Regression, Random Forest and SVM 

# In[471]:


stack2 = StackingCVRegressor(regressors=(LR, random_forest_model,svm_regr),
                            meta_regressor=LR, cv=12,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack2.fit(X_train, y_train)


# In[472]:


# y_train_predict=stack2.predict(X_train)
# # y_test_predict=stack2.predict(X_test)


# # In[473]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# In[474]:


# print("-----------Training Accuracy------------")
# print(round(stack2.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(stack2.score(X_test,y_test),3)*100)


# # In[475]:


# from sklearn.metrics import r2_score


# In[476]:


# score = r2_score(y_test, y_test_predict)


# In[477]:

# 
# sc ore


# ### ❏ Hybrid Model 3

# The stacked model with meta learner = Logistic Regression and the weak learners = Linear Regression, Random Forest and SVM 

# In[502]:


stack3 = StackingCVRegressor(regressors=(LR, random_forest_model,logreg),
                            meta_regressor=LR, cv=12,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack3.fit(X_train, y_train)


# In[503]:


# y_train_predict=stack3.predict(X_train)
# y_test_predict=stack3.predict(X_test)


# In[504]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# In[505]:


# print("-----------Training Accuracy------------")
# print(round(stack3.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(stack3.score(X_test,y_test),3)*100)


# ### ❏ Hybrid Model 4

# In[544]:


stack4 = StackingCVRegressor(regressors=(LR, random_forest_model,gbr),
                            meta_regressor=LR, cv=12,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack4.fit(X_train, y_train)

# predicted = stack4.predict([[2014,2]])
# print(predicted)

# stack4.predict(["month","year""])
# In[545]:


# y_train_predict=stack4.predict(X_train)
# y_test_predict=stack4.predict(X_test)


# In[546]:


# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

# print("\n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train,y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))


# In[547]:


# print("-----------Training Accuracy------------")
# print(round(stack4.score(X_train,y_train),3)*100)
# print("-----------Testing Accuracy------------")
# print(round(stack4.score(X_test,y_test),3)*100)


# In[ ]:


file = open("model.pkl","wb")
pickle.dump(stack4,file)
file.close()