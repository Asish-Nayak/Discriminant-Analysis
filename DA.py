# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 09:13:07 2022

@author: harigaran
"""

import pandas as pd 
df= pd.read_csv(r"C:\Users\ADMIN\Desktop\MBA - BA II\Multivariate analysis lab\4.DA\DAdata.csv")
df.columns = df.columns.str.replace(" ", "_")
df=df.rename(columns = {'Annual_family_income_(000s)':'Annual_family_income'})
#Dropping unnecessary columns
df.drop(['Respondent_Number'],axis = 1, inplace=True)
df.info()
#split the feature and target variable
x = df.drop(['Resort_visit'],axis = 1)
x.info()
y = df['Resort_visit']

#group frequency
count = df.groupby(['Resort_visit']).size()
print(count)
#group mean
class_feature_means = pd.DataFrame(columns=y)
for c, rows in df.groupby('Resort_visit'):
    class_feature_means[c] = rows.mean()
class_feature_means = class_feature_means.drop('Resort_visit')
class_feature_means



from statsmodels.multivariate.manova import MANOVA
fit = MANOVA.from_formula('Annual_family_income + Attitude_towads_travel +\
                          Importance_attached_to_family_skiing_holiday+\
                           Household_size+\
                              Age_of_head_of_household + \
                                  Amount_spent_on_family_skiing ~ Resort_visit', data=df)
print(fit.mv_test())

#LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import numpy as np
lda =LinearDiscriminantAnalysis(n_components = 1)
da =lda.fit(x,y)
y_pred = lda.predict(x)
print(y_pred)

# get Prior probabilities of groups:
da.priors_

#plot
X_new = pd.DataFrame(da.transform(x), columns=["lda1"])
val = 0. # this is the value where you want the data to appear on the y-axis.
X_new['null'] = np.zeros_like(X_new) +val;
sns.scatterplot(data=X_new, x="lda1", y="null", hue=df.Resort_visit.tolist(),palette=["C0", "C1"])



from sklearn import metrics
cm=metrics.confusion_matrix(y,y_pred)
cm
x_axis = [1,2]
y_axis = [1,2]
p=sns.heatmap(cm, annot=True, cmap='BrBG_r',xticklabels=x_axis,yticklabels=y_axis)
p.set_xlabel("Predicted", fontsize = 20)
p.set_ylabel("Actual", fontsize = 20)


#test dataset
df_test= pd.read_csv(r"C:\Users\my pc\Desktop\MBA - BA II\Multivariate analysis lab\4.DA\DAdata_test.csv")
df_test.columns = df_test.columns.str.replace(" ", "_")
df_test=df_test.rename(columns = {'Annual_family_income_(000s)':'Annual_family_income'})
df_test.drop(['Respondent_Number'],axis = 1, inplace=True)
test_pred = lda.predict(df_test)
test_pred 