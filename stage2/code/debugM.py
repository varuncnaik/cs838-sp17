''' 
Usage: 
    python debugM.py dev_set.json test_set.json

'''

import json
import numpy as np
import random
from sys import argv
from sklearn.svm import SVC
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



def get_reviews(filename):
    with open(filename) as f:
        reviews = [json.loads(line) for line in f]
    return reviews
    
def split_dev_set(filename):
    '''
    split dev set I into two parts P, Q, each containing 100 documents
    '''
    random.seed(80)
    all_reviews = get_reviews(filename)
    random.shuffle(all_reviews)
    P = all_reviews[:100]
    Q = all_reviews[100:]
    return P, Q
    
def get_fvs(reviews):       
    '''
    from reviews, pull feature vectors, labels, and text
    '''        
    fvs = []
    labels = []
    text = []
    for rev in reviews:
        for fv in rev['fvs']:
            fvs.append([fv[feat] for feat in fv if feat != 'is_positive' and feat != 'text'])
            labels.append(fv['is_positive'])
            text.append(fv['text'])
    return np.array(fvs), np.array(labels), np.array(text)
    
def false_pos_neg(test_fvs, test_labels, test_text, predicted_labels):
    '''
    Find and display false positives and false negatives for debugging
    '''
    falsepositives = []
    falsepositivestext = []
    falsenegatives = []
    falsenegativestext = []
    for i, actual_label in enumerate(test_labels):
        if actual_label == 1 and predicted_labels[i] == 0:
            falsenegatives.append(test_fvs[i])
            falsenegativestext.append(test_text[i])
        elif actual_label == 0 and predicted_labels[i] == 1:
            falsepositives.append(test_fvs[i])
            falsepositivestext.append(test_text[i])
    
    print len(falsepositives), 'false positives (improve precision):'
    for i, fv in enumerate(falsepositives):
        print falsepositivestext[i], list(fv)
    print
    print len(falsenegatives), 'false negatives (improve recall):'
    for i, fv in enumerate(falsenegatives):
        print falsenegativestext[i], list(fv)
        
    
def debug():
    '''
    Split dev set I into sets P, Q, each containing 100 docs
    Find & display false positives and false negatives
    '''
    P, Q = split_dev_set(argv[1])
    fvsP, labelsP, textP = get_fvs(P)
    fvsQ, labelsQ, textQ = get_fvs(Q)
    
    # fit SVM classifier using set P
    svmclf = SVC()
    svmclf.fit(fvsP, labelsP)
    # test classifier trained on set P using set Q
    predicted_before_rule = svmclf.predict(fvsQ)
    
    # display false positives/negatives before rules
    false_pos_neg(fvsQ, labelsQ, textQ, predicted_before_rule)
    
    predicted = []
    # rule: menu items must contain a noun
    for i, pmi in enumerate(textQ):
        tokens = word_tokenize(pmi)
        tags = pos_tag(tokens)
        has_noun = False
        for t in tags:
            if t[-1] == 'NN' or t[-1] == 'NNS':
                has_noun = True
                break
        if predicted_before_rule[i] == 1 and has_noun == False:
            # manually change predicted label
            predicted.append(0)
        else:
            predicted.append(predicted_before_rule[i])
    
    # display false positives/negatives after rules
    print
    false_pos_neg(fvsQ, labelsQ, textQ, predicted)

def apply_rules(test_text, test_pred_labels):
    predicted = []
    stopWords = set(stopwords.words('english'))
    # rule: menu items must contain a noun
    for i, pmi in enumerate(test_text):
        tokens = word_tokenize(pmi)
        tags = pos_tag(tokens)
        has_noun = False
        for t in tags:
            if t[-1] == 'NN' or t[-1] == 'NNS':
                has_noun = True
                break
        if test_pred_labels[i] == 1 and has_noun == False:
            # manually change predicted label
            predicted.append(0)
        elif tokens[0] in stopWords or tokens[-1] in stopWords:
            predicted.append(0)
        else:
            predicted.append(test_pred_labels[i])
    # see what false positives/negatives we have left
    #false_pos_neg(test_fvs, test_labels, test_text, predicted)
    return predicted
    
def train_metrics():   
    ''' 
    Do 10 fold CV using SVM and rules on dev set I
    Calculate P/R/F1 
    '''
    
    all_reviews = get_reviews(argv[1])
    fvs, labels, text = get_fvs(all_reviews)
    
    svmclf = SVC()
    
    # cross-valiation
    folds = 10
    kf = KFold(n_splits=folds, shuffle = True, random_state=36)
    precision = []
    recall = []
    f1 = []

    for train_i, test_i in kf.split(fvs):
        train_fvs = [fvs[i] for i in train_i]
        train_labels = [labels[i] for i in train_i]
        #train_text = [text[i] for i in train_i]
        test_fvs = [fvs[i] for i in test_i]
        test_labels = [labels[i] for i in test_i]
        test_text = [text[i] for i in test_i]
        svmclf.fit(train_fvs, train_labels)
        test_pred_labels = svmclf.predict(test_fvs)
        test_pred_labels = np.array(test_pred_labels)
        
        predicted = apply_rules(test_text, test_pred_labels)
                
        precision.append(precision_score(test_labels, predicted))
        recall.append(recall_score(test_labels, predicted))
        f1.append(f1_score(test_labels, predicted))
    p = np.array(precision)
    r = np.array(recall)
    f1 = np.array(f1)
    
    print 'training set metrics:'
    print 'mean 10 fold cv precision:', p.mean()
    print 'mean 10 fold cv recall:', r.mean()
    print 'mean 10 fold cv f1:', f1.mean()
    
def main():
    ''' 
    Finally, train classifier with dev set, use it on test set,
    apply rules, calculate P/R/F1
    '''
    train_reviews = get_reviews(argv[1])
    train_fvs, train_labels, train_text = get_fvs(train_reviews)
    
    # train classifier with dev set
    clf = SVC()
    clf.fit(train_fvs, train_labels)
    
    # get test set
    test_reviews = get_reviews(argv[2])
    test_fvs, test_labels, test_text = get_fvs(test_reviews)
    
    # use trained classifier on test set
    test_pred_labels = clf.predict(test_fvs)
    test_pred_labels = np.array(test_pred_labels)
    
    predicted = apply_rules(test_text, test_pred_labels)
    
    print 'Test set metrics:'
    print 'Precision:', precision_score(test_labels, predicted)
    print 'Recall:', recall_score(test_labels, predicted)
    print 'F1:', f1_score(test_labels, predicted)

if __name__ == "__main__":
    main()