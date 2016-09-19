#coding=utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('g:/kaggle/house/train.csv')
print df.head()
print df.describe()

print("Some Statistics of the Housing Price:\n")
print(df['SalePrice'].describe())
print("\nThe median of the Housing Price is: ", df['SalePrice'].median(axis = 0))

sns.distplot(df['SalePrice'], kde = False, color = 'b', hist_kws={'alpha': 0.9})

#Numerical Features
#corr：皮尔逊相关系数
corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
print df.select_dtypes(include = ['float64', 'int64'])#选出所有type为float64 int64的字段
# plt.figure(figsize=(12, 12))
# sns.heatmap(corr, vmax=1, square=True)

cor_dict = corr['SalePrice'].to_dict()
del cor_dict['SalePrice']
print("List the numerical features decendingly by their correlation with Sale Price:\n")
for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*ele))

'''
The housing price correlates strongly with OverallQual, GrLivArea(GarageCars), GargeArea, TotalBsmtSF,
 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd, GargeYrBlt, MasVnrArea and Fireplaces.
But some of those features are highly correlated among each others.
'''

# sns.regplot(x = 'OverallQual', y = 'SalePrice', data = df, color = 'Orange')#x,y 是要选择df里面的字段

price = df.SalePrice.values
def first_fig():
    plt.figure(1)
    f, axarr = plt.subplots(3, 2, figsize=(10, 9))
    price = df.SalePrice.values
    axarr[0, 0].scatter(df.GrLivArea.values, price)
    axarr[0, 0].set_title('GrLiveArea')
    axarr[0, 1].scatter(df.GarageArea.values, price)
    axarr[0, 1].set_title('GarageArea')
    axarr[1, 0].scatter(df.TotalBsmtSF.values, price)
    axarr[1, 0].set_title('TotalBsmtSF')
    axarr[1, 1].scatter(df['1stFlrSF'].values, price)
    axarr[1, 1].set_title('1stFlrSF')
    axarr[2, 0].scatter(df.TotRmsAbvGrd.values, price)
    axarr[2, 0].set_title('TotRmsAbvGrd')
    axarr[2, 1].scatter(df.MasVnrArea.values, price)
    axarr[2, 1].set_title('MasVnrArea')
    f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize = 12)
    plt.tight_layout()

    plt.show()

def second_fig():
    fig = plt.figure(2, figsize=(9, 7))
    plt.subplot(211)
    plt.scatter(df.YearBuilt.values, price)
    plt.title('YearBuilt')

    plt.subplot(212)
    plt.scatter(df.YearRemodAdd.values, price)
    plt.title('YearRemodAdd')

    fig.text(0.01, 0.5, 'Sale Price', va = 'center', rotation = 'vertical', fontsize = 12)#添加sale price 这几个字在图的左边

    plt.tight_layout()

    plt.show()
# second_fig()

#Categorical Features
def third():
    print(df.select_dtypes(include=['object']).columns.values)

    plt.figure(figsize = (12, 6))
    sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df)
    xt = plt.xticks(rotation=45)

    plt.figure(figsize = (12, 6))
    sns.countplot(x = 'Neighborhood', data = df)
    xt = plt.xticks(rotation=45)

    #Could group those Neighborhoods with similar housing price into a same bucket for dimension-reduction.

def fourth():
    #Housing Price vs Sales
    fig, ax = plt.subplots(2, 1, figsize = (10, 6))
    sns.boxplot(x = 'SaleType', y = 'SalePrice', data = df, ax = ax[0])
    sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = df, ax = ax[1])
    plt.tight_layout()

    g = sns.FacetGrid(df, col = 'YrSold', col_wrap = 3)
    g.map(sns.boxplot, 'MoSold', 'SalePrice', palette='Set2', order = range(1, 13)).set(ylim = (0, 500000))
    plt.tight_layout()
    #Sale's timing does not seem to hugely affect the house.

#Housing Style
fig, ax = plt.subplots(2, 1, figsize = (10, 8))
sns.boxplot(x = 'BldgType', y = 'SalePrice', data = df, ax = ax[0])
sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = df, ax = ax[1])

fig, ax = plt.subplots(2, 1, figsize = (10, 8))
sns.boxplot(x = 'Condition1', y = 'SalePrice', data = df, ax = ax[0])
sns.boxplot(x = 'Exterior1st', y = 'SalePrice', data = df, ax = ax[1])
x = plt.xticks(rotation = 45)

fig, ax = plt.subplots(2, 2, figsize = (10, 8))
sns.boxplot('BsmtCond', 'SalePrice', data = df, ax = ax[0, 0])
sns.boxplot('BsmtQual', 'SalePrice', data = df, ax = ax[0, 1])
sns.boxplot('BsmtExposure', 'SalePrice', data = df, ax = ax[1, 0])
sns.boxplot('BsmtFinType1', 'SalePrice', data = df, ax = ax[1, 1])

g = sns.FacetGrid(df, col = 'FireplaceQu', col_wrap = 3, col_order=['Ex', 'Gd', 'TA', 'Fa', 'Po'])
g.map(sns.boxplot, 'Fireplaces', 'SalePrice', order = [1, 2, 3], palette = 'Set2')

#Electrical and Price
fig, ax = plt.subplots(1, 2, figsize = (10, 4))
sns.boxplot('Electrical', 'SalePrice', data = df, ax = ax[0]).set(ylim = (0, 400000))
sns.countplot('Electrical', data = df)
plt.tight_layout()

#Street & Alley Access
fig, ax = plt.subplots(1, 2, figsize = (10, 4))
sns.boxplot(x = 'Street', y = 'SalePrice', data = df, ax = ax[0])
sns.boxplot(x = 'Alley', y = 'SalePrice', data = df, ax = ax[1])
plt.tight_layout()

plt.show()