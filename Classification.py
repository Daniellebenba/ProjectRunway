# Lets import some modules for basic computation
import time
import pandas as pd
import numpy as np
import random

import pickle

# Some modules for plotting and visualizing
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# And some Machine Learning modules from scikit-learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#These Classifiers have been commented out because they take too long and do not give more accuracy as the other ones.
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.gaussian_process import GaussianProcessClassifier


#1. The dataset
filename_glass = r'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\data.csv'
df_data = pd.read_csv(filename_glass)

#Take only levels in: levels
levels = [1,2,3,4,5]
between_shuffle = 0
df_data = df_data.loc[df_data['level'].isin(levels)]

print("This dataset has nrows, ncols: {}".format(df_data.shape))
display(df_data.head())
display(df_data.describe())


#Separate data into train and test
#between_shuffle = 1 <=> shuffle with all subjects train and test. between_shuffle = 0 <=> subjects different in train and in test
def get_train_test(df, y_col, x_cols, ratio, between_shuffle=1):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    if between_shuffle == 1:
        mask = np.random.rand(len(df)) < ratio

        df_train = df[mask]
        df_test = df[~mask]
        Y_train = df_train[y_col].values
        Y_test = df_test[y_col].values
        X_train = df_train[x_cols].values
        X_test = df_test[x_cols].values
        return df_train, df_test, X_train, Y_train, X_test, Y_test

    else:
        subjects = df.sub_num.unique()
        random.shuffle(subjects)
        num_subjects = len(subjects)
        #x = num_subjects*ratio
        num_train = int(round(num_subjects*ratio))
        train_sub = subjects[0:num_train]
        test_sub = subjects[num_train:num_subjects]
        df_train = df.loc[df['sub_num'].isin(train_sub)]
        df_test = df.loc[df['sub_num'].isin(test_sub)]
        Y_train = df_train[y_col].values
        Y_test = df_test[y_col].values
        X_train = df_train[x_cols].values
        X_test = df_test[x_cols].values
        return df_train, df_test, X_train, Y_train, X_test, Y_test




dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
    #"AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    #"Gaussian Process": GaussianProcessClassifier()
}


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.

    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train.
    So it is best to train them on a smaller dataset first and
    decide whether you want to comment them out or not based on the test accuracy score.
    """

    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()

        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,
                                        'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models


def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)),
                       columns=['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    display(df_.sort_values(by=sort_by, ascending=False))

###################################################################################
#1.3 Classification
y_cols = 'level'
x_cols = list(df_data.columns.values)
x_cols.remove(y_cols)


#Set data
train_test_ratio = 0.7
df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_data, y_cols, x_cols, train_test_ratio, between_shuffle)

#Training
dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8)

#Display results
display_dict_models(dict_models)

###################################################################################

def display_corr_with_col(df, col):
    correlation_matrix = df.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()

###################################################################################
#2.5 Improving upon the Classifier: hyperparameter optimization

#
GDB_params = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.5, 0.1, 0.01, 0.001],
    'criterion': ['friedman_mse', 'mse', 'mae']
}

df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_data, y_cols, x_cols, 0.6)

for n_est in GDB_params['n_estimators']:
    for lr in GDB_params['learning_rate']:
        for crit in GDB_params['criterion']:
            clf = GradientBoostingClassifier(n_estimators=n_est,
                                             learning_rate = lr,
                                             criterion = crit)

            clf.fit(X_train, Y_train)
            train_score = clf.score(X_train, Y_train)
            test_score = clf.score(X_test, Y_test)
            print("For ({}, {}, {}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(n_est, lr, crit[:4], train_score, test_score))

#3. Understanding complex datasets
#3.1 Correlation Matrix
correlation_matrix = df_data.corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
plt.title('Correlation matrix between the features', fontsize=20)
plt.show()


display_corr_with_col(df_data, 'level')


#3.2 Cumulative Explained Variance
X = df_data[x_cols].values
X_std = StandardScaler().fit_transform(X)

# pca = PCA().fit(X_std)
# var_ratio = pca.explained_variance_ratio_
# components = pca.components_
# print(pca.explained_variance_)
# plt.plot(np.cumsum(var_ratio))
# plt.xlim(0,9,1)
# plt.xlabel('Number of Features', fontsize=16)
# plt.ylabel('Cumulative explained variance', fontsize=16)
# plt.show()

# #3.3 Pairwise relationships between the features
# ax = sns.pairplot(df_data, hue='level')
# plt.title('Pairwise relationships between the features')
# plt.show()
