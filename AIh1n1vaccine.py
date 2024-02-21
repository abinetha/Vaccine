import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

df1=pd.read_csv('C:\\Users\\DELL\\Downloads\\h1n1_vaccine_prediction.csv')
print(df1.shape)
print(df1.info())
print(df1.isnull().sum())
print(df1.describe())

median1=df1['h1n1_worry'].median()
# print(median1)
df1['h1n1_worry']=df1['h1n1_worry'].replace(np.nan,median1)
# print(df1['h1n1_worry'].value_counts())

median2=df1['h1n1_awareness'].median()
# print(median2)
df1['h1n1_awareness']=df1['h1n1_awareness'].replace(np.nan,median2)
# print(df1['h1n1_awareness'].value_counts())

median3=df1['antiviral_medication'].median()
# print(median3)
df1['antiviral_medication']=df1['antiviral_medication'].replace(np.nan,median3)
# print(df1['antiviral_medication'].value_counts())

median4=df1['contact_avoidance'].median()
# print(median4)
df1['contact_avoidance']=df1['contact_avoidance'].replace(np.nan,median4)
# print(df1['contact_avoidance'].value_counts())

median5=df1['bought_face_mask'].median()
# print(median5)
df1['bought_face_mask']=df1['bought_face_mask'].replace(np.nan,median5)
# print(df1['bought_face_mask'].value_counts())

median6=df1['wash_hands_frequently'].median()
# print(median6)
df1['wash_hands_frequently']=df1['wash_hands_frequently'].replace(np.nan,median6)
# print(df1['wash_hands_frequently'].value_counts())

median7=df1['avoid_large_gatherings'].median()
# print(median7)
df1['avoid_large_gatherings']=df1['avoid_large_gatherings'].replace(np.nan,median7)
# print(df1['avoid_large_gatherings'].value_counts())

median8=df1['reduced_outside_home_cont'].median()
# print(median8)
df1['reduced_outside_home_cont']=df1['reduced_outside_home_cont'].replace(np.nan,median8)
# print(df1['reduced_outside_home_cont'].value_counts())

median9=df1['avoid_touch_face'].median()
# print(median9)
df1['avoid_touch_face']=df1['avoid_touch_face'].replace(np.nan,median9)
# print(df1['avoid_touch_face'].value_counts())

median10=df1['dr_recc_h1n1_vacc'].median()
# print(median10)
df1['dr_recc_h1n1_vacc']=df1['dr_recc_h1n1_vacc'].replace(np.nan,median10)
# print(df1['dr_recc_h1n1_vacc'].value_counts())

median11=df1['dr_recc_seasonal_vacc'].median()
# print(median11)
df1['dr_recc_seasonal_vacc']=df1['dr_recc_seasonal_vacc'].replace(np.nan,median11)
# print(df1['dr_recc_seasonal_vacc'].value_counts())

median12=df1['chronic_medic_condition'].median()
# print(median12)
df1['chronic_medic_condition']=df1['chronic_medic_condition'].replace(np.nan,median12)
# print(df1['chronic_medic_condition'].value_counts())

median13=df1['cont_child_undr_6_mnths'].median()
# print(median13)
df1['cont_child_undr_6_mnths']=df1['cont_child_undr_6_mnths'].replace(np.nan,median13)
# print(df1['cont_child_undr_6_mnths'].value_counts())

median14=df1['is_health_worker'].median()
# print(median14)
df1['is_health_worker']=df1['is_health_worker'].replace(np.nan,median14)
# print(df1['is_health_worker'].value_counts())

median15=df1['has_health_insur'].median()
# print(median15)
df1['has_health_insur']=df1['has_health_insur'].replace(np.nan,median15)
# print(df1['has_health_insur'].value_counts())

median16=df1['is_h1n1_vacc_effective'].median()
# print(median16)
df1['is_h1n1_vacc_effective']=df1['is_h1n1_vacc_effective'].replace(np.nan,median16)
# print(df1['is_h1n1_vacc_effective'].value_counts())

median17=df1['is_h1n1_risky'].median()
# print(median17)
df1['is_h1n1_risky']=df1['is_h1n1_risky'].replace(np.nan,median17)
# print(df1['is_h1n1_risky'].value_counts())

median18=df1['sick_from_h1n1_vacc'].median()
# print(median18)
df1['sick_from_h1n1_vacc']=df1['sick_from_h1n1_vacc'].replace(np.nan,median18)
# print(df1['sick_from_h1n1_vacc'].value_counts())

median19=df1['is_seas_vacc_effective'].median()
# print(median19)
df1['is_seas_vacc_effective']=df1['is_seas_vacc_effective'].replace(np.nan,median19)
# print(df1['is_seas_vacc_effective'].value_counts())

median20=df1['is_seas_risky'].median()
# print(median20)
df1['is_seas_risky']=df1['is_seas_risky'].replace(np.nan,median20)
# print(df1['is_seas_risky'].value_counts())

median21=df1['sick_from_seas_vacc'].median()
# print(median21)
df1['sick_from_seas_vacc']=df1['sick_from_seas_vacc'].replace(np.nan,median21)
# print(df1['sick_from_seas_vacc'].value_counts())

median22=df1['no_of_adults'].median()
# print(median22)
df1['no_of_adults']=df1['no_of_adults'].replace(np.nan,median22)
# print(df1['no_of_adults'].value_counts())

median23=df1['no_of_children'].median()
# print(median23)
df1['no_of_children']=df1['no_of_children'].replace(np.nan,median23)
# print(df1['no_of_children'].value_counts())

median24=df1['h1n1_vaccine'].median()
# print(median24)
df1['h1n1_vaccine']=df1['h1n1_vaccine'].replace(np.nan,median24)
# print(df1['h1n1_vaccine'].value_counts())

# Remove outlier
def remove_outlier(col):
    sorted(col)
    q1, q3 = col.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_range = q1 - 1.5 * iqr
    upper_range = q3 + 1.5 * iqr
    return lower_range, upper_range

mode1=df1['age_bracket'].mode().values[0]
# print(mode1)
df1['age_bracket']=df1['age_bracket'].replace(np.nan,mode1)
# print(df1['age_bracket'].value_counts())

mode2=df1['qualification'].mode().values[0]
# print(mode2)
df1['qualification']=df1['qualification'].replace(np.nan,mode2)
# print(df1['qualification'].value_counts())

mode3=df1['race'].mode().values[0]
# print(mode3)
df1['race']=df1['race'].replace(np.nan,mode3)
# print(df1['race'].value_counts())

mode4=df1['sex'].mode().values[0]
# print(mode4)
df1['sex']=df1['sex'].replace(np.nan,mode4)
# print(df1['sex'].value_counts())

mode5=df1['income_level'].mode().values[0]
# print(mode5)
df1['income_level']=df1['income_level'].replace(np.nan,mode5)
# print(df1['income_level'].value_counts())

mode6=df1['marital_status'].mode().values[0]
# print(mode6)
df1['marital_status']=df1['marital_status'].replace(np.nan,mode6)
# print(df1['marital_status'].value_counts())

mode7=df1['housing_status'].mode().values[0]
# print(mode7)
df1['housing_status']=df1['housing_status'].replace(np.nan,mode7)
# print(df1['housing_status'].value_counts())

mode8=df1['employment'].mode().values[0]
# print(mode8)
df1['employment']=df1['employment'].replace(np.nan,mode8)
# print(df1['employment'].value_counts())

mode9=df1['census_msa'].mode().values[0]
# print(mode9)
df1['census_msa']=df1['census_msa'].replace(np.nan,mode9)
# print(df1['census_msa'].value_counts())

print(df1.isnull().sum())

# Dropping Unwanted Columns

df1=df1.drop(['housing_status','race','unique_id'],axis=1)

print(df1.info())

df1=pd.get_dummies(df1,columns=['age_bracket','qualification','sex','income_level','marital_status','employment','census_msa'],dtype=int)

print(df1.isnull().sum())
print(df1.dtypes)

#### Split into x and y
x=df1.drop('h1n1_vaccine',axis=1)
y=df1['h1n1_vaccine']

#### Training and Testing split
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.3,random_state=1)


#### Apply Logistic Regression model
model3=LogisticRegression()
model3.fit(x_train1,y_train1)

print(model3.score(x_train1,y_train1))
print(model3.score(x_test1,y_test1))

predictions = model3.predict(x_test1)
print(accuracy_score(y_test1,predictions))

print(metrics.classification_report(y_test1,predictions))

print(confusion_matrix(y_test1,predictions))

cm=metrics.confusion_matrix(y_test1,predictions,labels=[1,0])

df_cm=pd.DataFrame(cm,index=[i for i in [1,0]], columns=[i for i in ['Predict 1','Predict 0']])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')
plt.show()

model4=DecisionTreeClassifier(max_depth=3)
model4.fit(x_train1,y_train1)
print(model4.score(x_train1,y_train1))
print(model4.score(x_test1,y_test1))

from sklearn.ensemble import BaggingClassifier
model5=BaggingClassifier(n_estimators=90,base_estimator=model4)
model5.fit(x_train1,y_train1)
print(model5.score(x_train1,y_train1))
print(model5.score(x_test1,y_test1))

from sklearn.ensemble import AdaBoostClassifier
model6=AdaBoostClassifier(n_estimators=27)
model6.fit(x_train1,y_train1)
print(model6.score(x_train1,y_train1))
print(model6.score(x_test1,y_test1))

from sklearn.ensemble import GradientBoostingClassifier
model7=GradientBoostingClassifier(n_estimators=5)
model7.fit(x_train1,y_train1)
print(model7.score(x_train1,y_train1))
print(model7.score(x_test1,y_test1))

from sklearn.ensemble import RandomForestClassifier
model8=RandomForestClassifier(n_estimators=100,max_depth=11)
model8.fit(x_train1,y_train1)
print(model8.score(x_train1,y_train1))
print(model8.score(x_test1,y_test1))

from sklearn.svm import SVC
model9=SVC()
model9.fit(x_train1,y_train1)
print(model9.score(x_train1,y_train1))
print(model9.score(x_test1,y_test1))

