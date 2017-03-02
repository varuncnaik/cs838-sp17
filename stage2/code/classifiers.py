''' 
Usage: 
    python classifiers.py dev_set.json

'''

import fileinput
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def get_reviews():
    reviews = [json.loads(line) for line in fileinput.input()]
    return reviews
    
def get_fvs():  
    reviews = get_reviews()               
    fvs = []
    labels = []
    for rev in reviews:
        for fv in rev['fvs']:
            fvs.append([fv[feat] for feat in fv if feat != 'is_positive' and feat != 'text'])
            labels.append(fv['is_positive'])
    # 661 positive examples, 1741 negative examples
    # labels.count(0), labels.count(1)
    return np.array(fvs), np.array(labels)

def main():
    fvs, labels = get_fvs()
    classifiers = ['DT', 'RF', 'SVM', 'LogR', 'LinR']
    pr_scores = {c: {'precision': None, 'recall': None}  for c in classifiers}
    folds = 4    

    dt = DecisionTreeClassifier()
    pr_scores['DT']['precision'] = cross_val_score(dt, fvs, labels, cv=folds, scoring='precision')
    pr_scores['DT']['recall'] = cross_val_score(dt, fvs, labels, cv=folds, scoring='recall')
    pr_scores['DT']['f1'] = cross_val_score(dt, fvs, labels, cv=folds, scoring='f1')
    
    rf = RandomForestClassifier()
    pr_scores['RF']['precision'] = cross_val_score(rf, fvs, labels, cv=folds, scoring='precision')
    pr_scores['RF']['recall'] = cross_val_score(rf, fvs, labels, cv=folds, scoring='recall')
    pr_scores['RF']['f1'] = cross_val_score(rf, fvs, labels, cv=folds, scoring='f1')
    
    svmclf = SVC()
    pr_scores['SVM']['precision'] = cross_val_score(svmclf, fvs, labels, cv=folds, scoring='precision')
    pr_scores['SVM']['recall'] = cross_val_score(svmclf, fvs, labels, cv=folds, scoring='recall')
    pr_scores['SVM']['f1'] = cross_val_score(svmclf, fvs, labels, cv=folds, scoring='f1')
    
    logr = linear_model.LogisticRegression()
    pr_scores['LogR']['precision'] = cross_val_score(logr, fvs, labels, cv=folds, scoring='precision')
    pr_scores['LogR']['recall'] = cross_val_score(logr, fvs, labels, cv=folds, scoring='recall')
    pr_scores['LogR']['f1'] = cross_val_score(logr, fvs, labels, cv=folds, scoring='f1')
    
    linr = linear_model.LinearRegression()
    kf = KFold(n_splits=folds, shuffle = True, random_state=42)
    linrp = []
    linrr = []
    linrf = []
    for train_i, test_i in kf.split(fvs):
        train_fvs = [fvs[i] for i in train_i]
        train_labels = [labels[i] for i in train_i]
        test_fvs = [fvs[i] for i in test_i]
        test_labels = [labels[i] for i in test_i]
        linr.fit(train_fvs, train_labels)
        test_predicted = linr.predict(test_fvs)
        test_pred_labels = []
        for v in test_predicted:
            if abs(v - 0) < abs(v - 1):
                test_pred_labels.append(0)
            else:
                test_pred_labels.append(1)
        test_pred_labels = np.array(test_pred_labels)
        linrp.append(precision_score(test_labels, test_pred_labels))
        linrr.append(recall_score(test_labels, test_pred_labels))
        linrf.append(f1_score(test_labels, test_pred_labels))
    pr_scores['LinR']['precision'] = np.array(linrp)
    pr_scores['LinR']['recall'] = np.array(linrr)
    pr_scores['LinR']['f1'] = np.array(linrf)
    
    for c in classifiers:
        print c
        print 'mean 4 fold cv precision:', pr_scores[c]['precision'].mean()
        print 'mean 4 fold cv recall:', pr_scores[c]['recall'].mean()
        print 'mean 4 fold cv f1:', pr_scores[c]['f1'].mean()
        print

if __name__ == "__main__":
    main()