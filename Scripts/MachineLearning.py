import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.linear_model import SGDClassifier

##############################################################################
def balanceDataset(df):
    labels = df.iloc[:, -1]
    
    df_majority = df[labels == 0]  # class 0 is major
    df_minority = df[labels == 1]  # class 1 is minor

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                    replace=True,     # sample with replacement
                                    # to match majority class
                                    n_samples=max(labels.value_counts()),
                                    random_state=123)  # reproducible result
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

def downBalance(df):
    labels = df.iloc[:, -1]

    # Separate majority and minority classes
    df_majority = df[labels==0]
    df_minority = df[labels==1]
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=min(labels.value_counts()),     # to match minority class
                                    random_state=123) # reproducible results
    
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    return df_downsampled

##############################################################################
# Read the csv file
df_base = pd.read_csv("/home/ec2-user/LabeledNetworkTraffic.csv", header=None)
# Drop all rows that contain at least one nan value
df = df_base.dropna(how='any')

# Split to 60 20 20
# https://www.quora.com/What-are-the-best-ways-to-predict-data-once-you-have-your-input-splitted-into-train-cross_validation-and-test-sets
# train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

# # Separate features from labels from training/validation set
# X = train.iloc[:, :-1]
# y = train.iloc[:, -1]
# validation_features = validate.iloc[:, :-1]
# validation_labels = validate.iloc[:, -1]
# test_features = test.iloc[:, :-1]
# test_labels = test.iloc[:, -1]

# ##############################################################################
# # Baseline model - Always predict majority class
# # Check accuracy on validation set : how many samples => always predict 0 => how many errors
# print(validation_labels.value_counts()[0]*100 / float(validation_labels.value_counts()[0] + validation_labels.value_counts()[1]))


# ##############################################################################
# # Check that dataset is imbalanced - big difference in value counts for each class
# print(y.value_counts())
# # # When dealing with imbalanced datasets,
# # # models ignore the minority class, and always predict the majority
# # # class: https://elitedatascience.com/imbalanced-classes

# ##############################################################################
# # Train LR model on imbalanced dataset
# clf_0 = LogisticRegression().fit(X, y)
# # Predict on validation set
# pred_y_0 = clf_0.predict(validation_features)
# # Pretty good accuracy, but because of favoring the majority class
# print( accuracy_score(pred_y_0, validation_labels) )


# # ##############################################################################
# # # Balance the dataset with upsampling
# df_balanced = balanceDataset(df)
# train_balanced, validate_balanced, test_balanced = np.split(df_balanced.sample(frac=1), [int(.6*len(df_balanced)), int(.8*len(df_balanced))])
# # # Separate features from labels
# X_balanced = train_balanced.iloc[:, :-1]
# y_balanced = train_balanced.iloc[:, -1]
# validation_features_balanced = validate_balanced.iloc[:, :-1]
# validation_labels_balanced = validate_balanced.iloc[:, -1]
# print(y_balanced.value_counts())

# # # Baseline model in balanced dataset predicts 50%
# print(validation_labels_balanced.value_counts()[0]*100 / float(validation_labels_balanced.value_counts()[0] + validation_labels_balanced.value_counts()[1]))


# ##############################################################################
# # Train LR model on balanced dataset
# clf_1 = LogisticRegression().fit(X_balanced, y_balanced)
# # Predict on validation set
# pred_y_1 = clf_1.predict(validation_features_balanced)
# # Now LR predicts worst than before but better than trivial
# print( accuracy_score(pred_y_1, validation_labels_balanced) )

# ##############################################################################
# # Imbalanced classes put "accuracy" out of business. 
# # This is a surprisingly common problem in machine learning 
# # (specifically in classification), occurring in datasets with 
# # a disproportionate ratio of observations in each class.
# # Standard accuracy no longer reliably measures performance, 
# # which makes model training much trickier. 
# # That's why we will change our performance metric to the ROC curve

# # Predict class probabilities
# prob_y_1 = clf_1.predict_proba(validation_features) 
# # Keep only the positive class
# prob_y_1 = [p[1] for p in prob_y_1]
# print( roc_auc_score(validation_labels, prob_y_1) )

# # Compare to the original LR model trained on the imbalanced dataset
# prob_y_0 = clf_0.predict_proba(validation_features)
# # Keep only the positive class
# prob_y_0 = [p[1] for p in prob_y_0] 
# print( roc_auc_score(validation_labels, prob_y_0) )    

# # Note: if you got an AUROC of 0.47, 
# # it just means you need to invert the predictions because Scikit-Learn 
# # is misinterpreting the positive class. AUROC should be >= 0.5.

# ##############################################################################
# # Try to train a random forest on the imbalanced dataset
# # Train RF model
# clf_4 = RandomForestClassifier().fit(X, y)

# # # Predict on validation set
# # pred_y_4 = clf_4.predict(validation_features)
 
# # # How's our accuracy?
# # print( accuracy_score(validation_labels, pred_y_4) )
 
# # What about AUROC?
# prob_y_4 = clf_4.predict_proba(validation_features)
# prob_y_4 = [p[1] for p in prob_y_4]
# print( roc_auc_score(validation_labels, prob_y_4) )


##############################################################################
# Hyperparameter Tuning on RF
# ...
# clf_3 = RandomForestClassifier(class_weight='balanced', max_features='log2', n_estimators=100, n_jobs=-1,
#                        oob_score=True, random_state=123).fit(X, y)
# prob_y_3 = clf_3.predict_proba(validation_features)
# prob_y_3 = [p[1] for p in prob_y_3]
# print( roc_auc_score(validation_labels, prob_y_3) )
# #97%

# clf_3 = RandomForestClassifier(class_weight='balanced', max_features='sqrt', n_estimators=100, n_jobs=-1,
#                        oob_score=True, random_state=123).fit(X, y)
# prob_y_3 = clf_3.predict_proba(validation_features)
# prob_y_3 = [p[1] for p in prob_y_3]
# print( roc_auc_score(validation_labels, prob_y_3) )
#97%


# clf_3 = RandomForestClassifier(bootstrap = True, class_weight='balanced', max_features='auto', max_depth=2, n_estimators=100, n_jobs=-1,
#                        oob_score=True, random_state=123).fit(X, y)
# prob_y_3 = clf_3.predict_proba(validation_features)
# prob_y_3 = [p[1] for p in prob_y_3]
# print( roc_auc_score(validation_labels, prob_y_3) )
# # #92%

# clf_3 = RandomForestClassifier(class_weight='balanced', max_features='auto', n_estimators=100, n_jobs=-1,
#                        oob_score=True, random_state=123).fit(X, y)
# prob_y_3 = clf_3.predict_proba(validation_features)
# prob_y_3 = [p[1] for p in prob_y_3]
# print( roc_auc_score(validation_labels, prob_y_3) )
# # 97%

# clf_3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=2, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#                        oob_score=False, random_state=0, verbose=0, warm_start=False).fit(X, y)
# 90%

# clf_3 = RandomForestClassifier(class_weight='balanced', max_features='sqrt', n_estimators=150, n_jobs=-1,
#                        oob_score=True, random_state=123).fit(X, y)

# prob_y_3 = clf_3.predict_proba(validation_features)
# prob_y_3 = [p[1] for p in prob_y_3]
# print( roc_auc_score(validation_labels, prob_y_3) )
# 97,2%
# clf_3 = RandomForestClassifier(class_weight='balanced',n_estimators=100, n_jobs=-1).fit(X, y)
# 97%

# clf_3 = RandomForestClassifier(class_weight='balanced', max_features='auto', n_estimators=100, n_jobs=-1, random_state=123).fit(X, y)
# #97.1
# prob_y_3 = clf_3.predict_proba(validation_features)
# prob_y_3 = [p[1] for p in prob_y_3]
# print( roc_auc_score(validation_labels, prob_y_3) )
# ##############################################################################
# # Even though we test which model is the best by using the validation set,
# # we have overfitted our model because of that validation set usage.
# # That's why we will use the test data for our final model's prediction score
# prob_y_5 = clf_3.predict_proba(test_features)
# prob_y_5 = [p[1] for p in prob_y_5]
# print( roc_auc_score(test_labels, prob_y_5) )

##############################################################################
# Check if Overfitting with CV on original dataset (not on splits)
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# clf_5 = LogisticRegression().fit(X, y)
# scores = cross_val_score(clf_5, X, y, cv=5)
# print("LR Accuracy (imbalanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# clf_4 = RandomForestClassifier(class_weight='balanced', max_features='auto', n_estimators=100, n_jobs=-1, random_state=123).fit(X, y)
# # scores = cross_val_score(clf_4, X, y, cv=5)
# # print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# indexToDrop = []
# feature_importances = clf_4.feature_importances_.tolist()
# counter = 0
# for f in feature_importances:
#     if f == 0.0:
#         print(feature_importances.index(f))
#     # print(format(f,".5f"))
#     # print(feature_importances.index(f))
#     # print("\n")
#         indexToDrop.append(counter)
#     counter = counter + 1
# # print(feature_importances)
# print(indexToDrop)
# df.drop(indexToDrop)
# df.drop([1, 15, 18, 19, 20, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42])
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

##############################################################################
# Avoid overfitting by using CV - Hyperparameter tuning on LR 

# LR Parameters:
# penalty : str, 'l1' or 'l2', default: 'l2'
# Used to specify the norm used in the penalization. 
# The 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties.

# dual : bool, default: False
# Dual or primal formulation. 
# Dual formulation is only implemented for l2 penalty with liblinear solver. 
# Prefer dual=False when n_samples > n_features.

# class_weight : dict or 'balanced', default: None
# Weights associated with classes in the form {class_label: weight}. 
# If not given, all classes are supposed to have weight one.

# The "balanced" mode uses the values of y to automatically adjust weights 
# inversely proportional to class frequencies in the input data as 
# n_samples / (n_classes * np.bincount(y)).

# random_state : int, RandomState instance or None, optional, default: None
# The seed of the pseudo random number generator to use when shuffling the data. 
# If int, random_state is the seed used by the random number generator; 
# If RandomState instance, random_state is the random number generator; 
# If None, the random number generator is the RandomState instance used by np.random. Used when solver == 'sag' or 'liblinear'.

# solver : str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, 
# default: 'liblinear'.
# Algorithm to use in the optimization problem.

# For small datasets, 'liblinear' is a good choice, 
# whereas 'sag' and 'saga' are faster for large ones.
# For multiclass problems, only 'newton-cg', 'sag', 'saga' 
# and 'lbfgs' handle multinomial loss; 'liblinear' is limited 
# to one-versus-rest schemes.
# 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas 
# 'liblinear' and 'saga' handle L1 penalty.
# Note that 'sag' and 'saga' fast convergence is only guaranteed 
# on features with approximately the same scale.

# max_iter : int, default: 100
# Useful only for the newton-cg, sag and lbfgs solvers. 
# Maximum number of iterations taken for the solvers to converge.

# multi_class : str, {'ovr', 'multinomial', 'auto'}, default: 'ovr'
# If the option chosen is 'ovr', then a binary problem is fit for each label. 

# n_jobs : int or None, optional (default=None)
# Number of CPU cores used when parallelizing over classes if multi_class='ovr'. 
# This parameter is ignored when the solver is set to 'liblinear' regardless of 
# whether 'multi_class' is specified or not. 
# None means 1 unless in a joblib.parallel_backend context. 
# -1 means using all processors. See Glossary for more details.

# clf_6 = LogisticRegression(n_jobs=-1, dual=False, class_weight='balanced', random_state=123, solver='lbfgs', penalty='l2', multi_class='ovr').fit(X, y)
# scores = cross_val_score(clf_6, X, y, cv=5)
# print("LR Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# 70%

# ##############################################################################
# UpSample dataset and check again
df_up = balanceDataset(df)
X = df_up.iloc[:, :-1]
y = df_up.iloc[:, -1]


# clf_6 = LogisticRegression(n_jobs=-1, class_weight=None, solver='lbfgs', penalty='l2', multi_class='multinomial').fit(X, y)
# scores = cross_val_score(clf_6, X, y, cv=5)
# print("LR Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# 80%

# clf_6 = LogisticRegression(n_jobs=-1, solver='lbfgs', multi_class='multinomial').fit(X, y)
# scores = cross_val_score(clf_6, X, y, cv=5)
# print("LR Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# # 80%
# # Keep the simpler

# # clf_6 = LogisticRegression(n_jobs=-1, solver='saga', multi_class='multinomial').fit(X, y)
# # scores = cross_val_score(clf_6, X, y, cv=5)
# # print("LR Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# # didnt converge

# clf_6 = LogisticRegression(n_jobs=-1, solver='saga', multi_class='ovr').fit(X, y)
# scores = cross_val_score(clf_6, X, y, cv=5)
# print("LR Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# didnt converge

# ##############################################################################
# HPT on RF
# clf_3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=2, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
#                        oob_score=False, random_state=0, verbose=0, warm_start=False).fit(X, y)
# scores = cross_val_score(clf_3, X, y, cv=5)
# print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 84%
# clf_3 = RandomForestClassifier(class_weight=None, criterion='gini', max_features='auto', n_estimators=150, n_jobs=-1, random_state=123).fit(X, y)
# scores = cross_val_score(clf_3, X, y, cv=5)
# print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 80%

# clf_3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#                        max_depth=2, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
#                        oob_score=False, random_state=0, verbose=0, warm_start=False).fit(X, y)
# scores = cross_val_score(clf_3, X, y, cv=5)
# print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# # 84%

# => !!! Best and Faster model than the rest
clf_3 = RandomForestClassifier(
        bootstrap=True, class_weight=None, criterion='entropy',
        max_depth=2, max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
        oob_score=True, random_state=0, verbose=0, warm_start=False
        ).fit(X, y)
scores = cross_val_score(clf_3, X, y, cv=5)
print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 84% 

# ##############################################################################
# # Apply a cost sensitive algorithm on downbalanced dataset for faster convergence 
# df_down = downBalance(df)
# df_sample = df_down.sample(frac=0.1, replace=True, random_state=1)
# train, validate, test = np.split(df_sample.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
# X = train.iloc[:, :-1]
# y = train.iloc[:, -1]
# validation_features = validate.iloc[:, :-1]
# validation_labels = validate.iloc[:, -1]

# clf_7 = SVC(kernel='rbf').fit(X,y)
# scores = cross_val_score(clf_7, X, y, cv=5)
# print("SVM Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# 52%

# clf_7 = SVC(kernel='rbf', verbose=True).fit(X,y)
# prob_y_4 = clf_7.predict(validation_features)
# print(prob_y_4)
# print( accuracy_score(prob_y_4, validation_labels) )
# scores = cross_val_score(clf_7, X, y, cv=5)
# print("SVM Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# ??? didnt converge %

# #instead of svm which takes forever, do sgd
# df_down = downBalance(df)
# X = df_down.iloc[:, :-1]
# y = df_down.iloc[:, -1]
# clf_3 = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
#        early_stopping=True, epsilon=0.1, eta0=0.0, fit_intercept=True,
#        l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=1000,
#        n_iter=None, n_iter_no_change=5, n_jobs=-1, penalty='l2',
#        power_t=0.5, random_state=123, shuffle=True, tol=0.001,
#        validation_fraction=0.1, verbose=0, warm_start=False).fit(X, y)
# scores = cross_val_score(clf_3, X, y, cv=5)
# print("SGD Accuracy (balanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# # 65%

# Xx = df.iloc[:, :-1]
# yy = df.iloc[:, -1]
# clf_3 = SGDClassifier(alpha=0.0001, average=False, class_weight='balanced',
#        early_stopping=True, epsilon=0.1, eta0=0.0, fit_intercept=True,
#        l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=1000,
#        n_iter=None, n_iter_no_change=5, n_jobs=-1, penalty='l2',
#        power_t=0.5, random_state=123, shuffle=True, tol=0.001,
#        validation_fraction=0.1, verbose=0, warm_start=False).fit(Xx, yy)
# scores = cross_val_score(clf_3, X, y, cv=5)
# print("SGD Accuracy (imbalanced): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 65%