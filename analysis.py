# Q1: who spent most
# Q2: who viewed most
# Q3: who completed which offer the most
# Q4: who spent during which offer the most
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split#, GridSearchCV

# read in files
df = pd.read_csv('data/df.csv')

# preprocessing
df['member_year'] = df['member_year'].apply(str)
to_be_str = ['web', 'social', 'mobile']
df[to_be_str] = df[to_be_str].replace(1, 'Yes')
df[to_be_str] = df[to_be_str].replace(0, 'No')
# analysis
# trait classification
person = ['gender', 'age', 'income', 'member_year']
offer = ['reward', 'difficulty', 'duration', 'offer_type', 'web', 'mobile',\
         'social']
response = ['total_spending', 'viewed', 'completed', 'spending']
###############################
#Q1: who spent the most
###############################
df1 = df[person+['total_spending']].dropna()
print(df1.groupby('gender')['total_spending'].mean())
print(df1.groupby('member_year')['total_spending'].mean())
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df1.groupby('gender')['total_spending']\
   .plot(kind='density', ind=np.linspace(0, 2500, 100), ax=axs[0])
axs[0].set_xlabel('total spending')
axs[0].yaxis.set_visible(False)
axs[0].legend(title='gender')
df1.groupby('member_year')['total_spending']\
   .plot(kind='density', ind=np.linspace(0, 2500, 100), ax=axs[1])
axs[1].set_xlabel('total spending')
axs[1].legend(title='member_year')
axs[1].yaxis.set_visible(False)
fig.tight_layout()
print('cor between income and total spending: {}'\
      .format(df[['income', 'total_spending']]\
              .corr(method='pearson').iloc[0,1]))
print('cor between age and total spending: {}'\
      .format(df[['age', 'total_spending']].corr(method='pearson').iloc[0, 1]))
# random forest & feature importance
person_numeric = ['age', 'income']
person_categorical = ['gender', 'member_year']
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(drop='first'))])
preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, person_numeric),
        ('cat', categorical_transformer, person_categorical)])
reg1 = Pipeline([('preprocessor', preprocessor),
                 ('reg', RandomForestRegressor(oob_score=True))])
X1 = df1.drop('total_spending', axis=1)
y1 = df1['total_spending']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)
reg1.fit(X1_train, y1_train)
print('oob score: {:.2f}'.format(reg1.named_steps['reg'].oob_score_))
print('test score: {:.2f}'.format(reg1.score(X1_test, y1_test)))
feature_names = person_numeric+list(reg1.named_steps['preprocessor']\
                                    .transformers_[1][1].named_steps['onehot']\
                                    .get_feature_names(person_categorical))
feature_importances = pd.Series(reg1.named_steps['reg'].feature_importances_,\
                                index=feature_names)
print(feature_importances.sort_values(ascending=False))
###############################
# Q2: who viewed the most
###############################
df2 = df[person+['viewed']].dropna()
print(df2.groupby('gender')['viewed'].mean())
print(df2.groupby('member_year')['viewed'].mean())
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df2.groupby('gender')['viewed'].mean().plot.barh(ax=axs[0])
axs[0].set_xlabel('viewed')
df2.groupby('member_year')['viewed'].mean().plot.barh(ax=axs[1])
axs[1].set_xlabel('viewed')
fig.tight_layout()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df2.groupby('viewed')['age']\
   .plot(kind='density', ind=np.linspace(0, 101, 40), ax=axs[0])
axs[0].set_xlabel('age')
axs[0].yaxis.set_visible(False)
df2.groupby('viewed')['income']\
   .plot(kind='density', ind=np.linspace(0, 180000, 100), ax=axs[1])
axs[1].set_xlabel('income')
axs[1].legend(title='viewed', labels=['Yes', 'No'])
axs[1].yaxis.set_visible(False)
fig.tight_layout()
# svc
clf2 = Pipeline([('scaler', StandardScaler()),
                 ('clf', SVC(kernel='rbf'))])
X2 = df2[['age', 'income']]
y2 = df2['viewed']*1
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2)
#param_grid = {
#    'clf__kernel': ['rbf', 'linear', 'poly']
#}
#grid_search = GridSearchCV(clf2, param_grid, cv=10)
#grid_search.fit(X2_train, y2_train)
#print('best model: {}'.format(grid_search.best_params_))
#print('best model test score: {:.2f}'\
#      .format(grid_search.score(X2_test, y2_test)))
clf2.fit(X2_train, y2_train)
# create a mesh to plot in
X2_age_min, X2_age_max = X2['age'].min(), X2['age'].max()
X2_income_min, X2_income_max = X2['income'].min(), X2['income'].max()
X2_age, X2_income = np.meshgrid(np.linspace(X2_age_min, X2_age_max, 10),
                                np.linspace(X2_income_min, X2_income_max, 10))
Z2 = clf2.predict(np.c_[X2_age.ravel(), X2_income.ravel()])
Z2 = Z2.reshape(X2_age.shape)
# Put the result into a color plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(X2_age, X2_income, Z2, cmap=plt.cm.coolwarm, alpha=0.8)
# Plot also the training points
ax.scatter(X2['age'], X2['income'], c=y2, cmap=plt.cm.coolwarm)
ax.set_xlabel('age')
ax.set_ylabel('income')
ax.set_xlim([X2_age_min, X2_age_max])
ax.set_ylim([X2_income_min, X2_income_max])
###############################
#Q3: who completed which offer the most
###############################
# with person
df31 = df[df['viewed']][person+['completed']].dropna()
print(df31.groupby('gender')['completed'].mean())
print(df31.groupby('member_year')['completed'].mean())
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df31.groupby('gender')['completed'].mean().plot.barh(ax=axs[0])
axs[0].set_xlabel('completed')
df31.groupby('member_year')['completed'].mean().plot.barh(ax=axs[1])
axs[1].set_xlabel('completed')
fig.tight_layout()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df31.groupby('completed')['age']\
    .plot(kind='density', ind=np.linspace(0, 101, 40), ax=axs[0])
axs[0].set_xlabel('age')
axs[0].yaxis.set_visible(False)
df31.groupby('completed')['income']\
    .plot(kind='density', ind=np.linspace(0, 180000, 100), ax=axs[1])
axs[1].set_xlabel('income')
axs[1].legend(title='completed', labels=['Yes', 'No'])
axs[1].yaxis.set_visible(False)
fig.tight_layout()
# with offer
# with offer - random forest & feature importance
df32 = df[(df['viewed']) & (df['offer_type']!='informational')]\
         [offer+['completed']]
df32.fillna(0, inplace=True)
offer_numeric = ['reward', 'difficulty', 'duration', 'web', 'mobile', 'social']
offer_categorical = ['offer_type']
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(drop='first'))])
preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, offer_numeric),
        ('cat', categorical_transformer, offer_categorical)])
clf32 = Pipeline([('preprocessor', preprocessor),
                  ('clf', RandomForestClassifier(oob_score=True))])
X32 = df32.drop('completed', axis=1)
y32 = df32['completed']
X32_train, X32_test, y32_train, y32_test = train_test_split(X32, y32)
clf32.fit(X32_train, y32_train)
print('oob score: {:.2f}'.format(clf32.named_steps['clf'].oob_score_))
print('test score: {:.2f}'.format(clf32.score(X32_test, y32_test)))
feature_names = offer_numeric+list(clf32.named_steps['preprocessor']\
                                   .transformers_[1][1].named_steps['onehot']\
                                   .get_feature_names(offer_categorical))
feature_importances = pd.Series(clf32.named_steps['clf'].feature_importances_,\
                                index=feature_names)
print(feature_importances.sort_values(ascending=False))
# with offer
df33 = df[(df['viewed']) & (df['offer_type']!='informational')]\
         [offer+['completed']]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
i = 0
feature_list = ['reward', 'difficulty', 'duration', 'offer_type']
for row in axs:
    for col in row:
        f = feature_list[i]
        if i < 2:
            df_temp = df33.dropna()
        else:
            df_temp = df33
        df_temp.groupby(f)['completed'].mean().plot.bar(rot=0, ax=col)
        col.set_ylim([0, 1])
        if (i == 0) or (i == 2):
            col.set_ylabel('completed')
        else:
            col.axes.get_yaxis().set_visible(False)
        i += 1
plt.subplots_adjust(hspace=0.1, wspace=0.1)
fig.tight_layout()
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
feature_list = ['reward', 'difficulty', 'duration']
for i in range(len(axs)):
    if i < 2:
        df_temp = df33.dropna()
    else:
        df_temp = df33
    f = feature_list[i]
    df_temp.groupby([f, 'offer_type'])['completed'].mean()\
           .unstack().plot.bar(rot=0, ax=axs[i])
    axs[i].set_ylim([0, 1])
    if i != 0:
        axs[i].axes.get_yaxis().set_visible(False)
        axs[i].get_legend().remove()
    else:
        axs[i].set_ylabel('completed')
fig.tight_layout()
# with person and offer
# with person and offer - random forest & feature importance
df34 = df[(df['viewed']) & (df['offer_type']!='informational')]\
         [person+offer+['completed']].dropna()
per_off_num = ['age', 'income', 'reward', 'difficulty', 'duration',\
               'web', 'mobile', 'social']
per_off_cat = ['gender', 'member_year', 'offer_type']
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(drop='first'))])
preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, per_off_num),
        ('cat', categorical_transformer, per_off_cat)])
clf34 = Pipeline([('preprocessor', preprocessor),
                  ('clf', RandomForestClassifier(oob_score=True))])
X34 = df34.drop('completed', axis=1)
y34 = df34['completed']
X34_train, X34_test, y34_train, y34_test = train_test_split(X34, y34)
clf34.fit(X34_train, y34_train)
print('oob score: {:.2f}'.format(clf34.named_steps['clf'].oob_score_))
print('test score: {:.2f}'.format(clf34.score(X34_test, y34_test)))
feature_names = per_off_num+list(clf34.named_steps['preprocessor']\
                                 .transformers_[1][1].named_steps['onehot']\
                                 .get_feature_names(per_off_cat))
feature_importances = pd.Series(clf34.named_steps['clf'].feature_importances_,\
                                index=feature_names)
print(feature_importances.sort_values(ascending=False))
# with person and offer
df35 = df[(df['viewed']) & (df['offer_type']!='informational')]\
         [person+offer+['completed']].dropna()
person_feature = ['gender', 'member_year']
offer_feature = ['reward', 'difficulty', 'duration', 'offer_type']
fig, axs = plt.subplots(len(offer_feature), len(person_feature),\
                        figsize=(20, 10))
for i, row in enumerate(axs):
    o = offer_feature[i]
    for j, col in enumerate(row):
        p = person_feature[j]
        df35.groupby([p, o])['completed'].mean()\
            .unstack().plot.bar(rot=0, ax=col)
        col.set_ylim([0, 1])
        if j == 0:
            col.set_ylabel('completed')
        else:
            col.axes.get_yaxis().set_visible(False)
        if i == 3:
            if j == 0:
                col.set_ylabel('gender')
            else:
                col.set_ylabel('member_year')
        else:
            col.axes.get_xaxis().set_visible(False)
        if j != 0:
            col.get_legend().remove()
        else:
            col.legend(title=o, loc='top center', ncol=df35[o].nunique())
fig.tight_layout()
###############################
#Q4: who spent during which offer the most
###############################
