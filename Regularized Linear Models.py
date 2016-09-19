#coding=utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
# from sklearn.model_selection import cross_val_score #0.18的
from sklearn.cross_validation import cross_val_score


pd.options.display.max_rows = 99

#呵呵，学习学习

train = pd.read_csv("g:/kaggle/house/train.csv")
test = pd.read_csv("g:/kaggle/house/test.csv")

all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],test.loc[:, 'MSSubClass':'SaleCondition']))

#First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal
#Create Dummy variables for the categorical features
#Replace the numeric missing values (NaN's) with the mean of their respective columns

# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
print prices.iloc[:,0]
plt.figure(2, figsize=(12, 7))
plt.subplot(121)
plt.hist(prices.iloc[:,0])


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#skewed distribution  偏态分布
'''
    I log transformed certain features for which the skew was > 0.75.This will make the feature more normally distributed
    and this makes linear regression perform better -since linear regression is sensitive to outliers.Note that if I used a
    tree-based model I wouldn't need to transform the variables.
'''
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)  #shape (2919, 288)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

y = train.SalePrice

'''
    Models
    Now we are going to use regularized linear regression models from the scikit learn module.I'm going to try both l_1(Lasso)
    and l_2(Ridge) regularization.I'll also define a function that returns the cross-validation rmse error so we can evaluate
    our models and pick the best tuning par
'''

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))
    return(rmse)

def ridge():
    model_ridge = Ridge()
    '''
    The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is.
     The higher the regularization the less prone our model will be to overfit.
    However it will also lose flexibility and might not capture all of the signal in the data.
    '''
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

    cv_ridge = pd.Series(cv_ridge, index = alphas)
    print cv_ridge
    # cv_ridge.plot(title = "Validation - Just Do It")
    plt.subplot(122)
    plt.plot(cv_ridge)
    plt.xlabel("alpha")
    plt.ylabel("rmse")

    print cv_ridge.min()

'''
Note the U-ish shaped curve above. When alpha is too small the regularization is too strong and
 the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha large)
 the model begins to overfit. A value of alpha = 10 is about right based on the plot above.
'''


#So for the Ridge regression we get a rmsle of about 0.127
#Let' try out the Lasso model. We will do a slightly different approach here and use the built in Lasso CV to figure out the
#best alpha for us. For some reason the alphas in Lasso CV are really the inverse or the alphas in Ridge.

def lasso():
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
    rmse_cv(model_lasso).mean()
    '''
    Nice! The lasso performs even better so we'll just use this one to predict on the test set.
     Another neat thing about the Lasso is that it does feature selection for you - setting coefficients of features
     it deems unimportant to zero. Let's take a look at the coefficients:
    '''
    coef = pd.Series(model_lasso.coef_, index = X_train.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    #Lasso picked 110 variables and eliminated the other 178 variables

    '''
    Good job Lasso. One thing to note here however is that the features selected are not necessarily
    the "correct" ones - especially since there are a lot of collinear features in this dataset.
    One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is.
    '''

    #We can also take a look directly at what the most important coefficients are:
    imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")

    '''
    The most important positive feature is GrLivArea - the above ground area by area square feet.
    This definitely sense. Then a few other location and quality features contributed positively.
    Some of the negative features make less sense and would be worth looking into more - it seems like they might come from
    unbalanced categorical variables.Also note that unlike the feature importance you'd get from a random forest these are
    actual coefficients in your model -so you can say precisely why the predicted price is what it is. The only issue here is that
    we log_transformed both the target and the numeric features so the actual magnitudes are a bit hard to interpret.
    '''

    '''
    loss minimizers/optimizers usually perform better when the variables have means and std deviations that are not too far apart.
    Doing log() or sqrt() usually helps reduce the scale of the variables so that they are closer to each other.
    '''

    # let's look at the residuals as well:
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

    preds = pd.DataFrame({"preds": model_lasso.predict(X_train), "true": y})
    preds["residuals"] = preds["true"] - preds["preds"]
    print preds['residuals']
    # preds.plot(x = "preds", y = "residuals",kind = "scatter")

    #The residual plot looks pretty good.To wrap it up let's predict on the test set and submit on the leaderboard:
    preds = np.expm1(model_lasso.predict(X_test))
    solution = pd.DataFrame({"id": test.Id, "SalePrice": preds})
    # solution.to_csv("ridge_sol.csv", index=False)
    plt.show()
lasso()