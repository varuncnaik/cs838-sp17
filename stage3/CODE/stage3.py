# Usage: python stage3.py ./DATA/sample_A.csv ./DATA/sample_B.csv ./DATA/I.csv ./DATA/J.csv

# -*- coding: utf-8 -*-
import py_entitymatching as em
from sys import argv

import re

def get_tables(A_file, B_file):
    '''
    A: songs.csv
    B: tracks.csv
    id is the key for songs.csv and tracks.csv
    '''
    A = em.read_csv_metadata(A_file, key='id')
    B = em.read_csv_metadata(B_file, key='id')
    return A, B

def downsamp(A, B):
    sample_A, sample_B = em.down_sample(A, B, size=5000, y_param=1)
    #sample_A.to_csv('sample_A.csv', index = False, encoding='utf-8')
    #sample_B.to_csv('sample_B.csv', index = False, encoding='utf-8')
    return sample_A, sample_B
    
def match(ltup, rtup):
    '''
    Returns True if (ltup, rtup) should be dropped, or False if (ltup, rtup) is
    a candidate. Called by bb_block().
    '''
    l_song_withstop = str(ltup['title']).lower().split()
    r_song_withstop = str(rtup['song']).lower().split()
    l_artist_withstop = str(ltup['artist_name']).lower().split()
    r_artist_withstop = str(rtup['artists']).lower().replace('+', ' ').split()
    stopwords = ('the', 'a')
    l_song = set(w for w in l_song_withstop if w not in stopwords)
    r_song = set(w for w in r_song_withstop if w not in stopwords)
    l_artist = set(w for w in l_artist_withstop if w not in stopwords)
    r_artist = set(w for w in r_artist_withstop if w not in stopwords)

    # If no overlap among artists or no overlap among songs, then drop
    return l_artist.isdisjoint(r_artist) or l_song.isdisjoint(r_song)

def bb_block(sample_A, sample_B):
    '''
    Returns a DataFrame of candidate pairs by performing black-box blocking.
    '''
    sample_B['artists'] = sample_B['artists'].str.replace('+', ' + ')

    # Create black box blocker
    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(match)
    bbC = bb.block_tables(sample_A, sample_B,
                        l_output_attrs=['id', 'title', 'artist_name', 'year'],
                        r_output_attrs=['id', 'title', 'year', 'episode', 'song', 'artists'],
    )

    return bbC
    
def overlap_block(sample_A, sample_B):
    '''
    Returns a DataFrame of candidate pairs by performing overlap blocking.
    Performs overlap blocking on title, then overlap blocking on artist, then
    black-box blocking to guarantee the same results as bb_block().
    '''
    # many artist names in sample_B are something like 
    # 'budda+wc+ice cube+mack 10+westside connection'
    # change this to 'budda + wc + ice cube + mack 10 + westside connection'
    sample_B['artists'] = sample_B['artists'].str.replace('+', ' + ')
    
    # Create overlap blocker
    # If no overlap among artists or no overlap among songs, then drop
    ob = em.OverlapBlocker()
    ob.stop_words = ['a', 'the']
    ob.regex_punctuation = re.compile(r'')
    obC1 = ob.block_tables(sample_A, sample_B, 'title', 'song',
                        l_output_attrs=['id', 'title', 'artist_name', 'year'],
                        r_output_attrs=['id', 'title', 'year', 'episode', 'song', 'artists'],
                        rem_stop_words=True,
    )
    ob.regex_punctuation = re.compile(r'\+')
    obC2 = ob.block_candset(obC1, 'artist_name', 'artists', rem_stop_words=True)

    # Create black box blocker
    # Necessary because overlap blocker behaves incorrectly when words contain Unicode characters
    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(match)
    bbC = bb.block_candset(obC2)

    return bbC
    
def debug_block(C, sample_A, sample_B):
    return em.debug_blocker(C,sample_A, sample_B, attr_corres=[('title', 'song'),('artist_name', 'artists')])
    
def samp_label_split(C):
    S = em.sample_table(C, 500)
    G = em.label_table(S, label_column_name='gold_labels')
    #G.to_csv('G.csv', index = False, encoding='utf-8')

    train_test = em.split_train_test(G, train_proportion=0.7)
    dev_set = train_test['train']
    test_set = train_test['test']
    #dev_set.to_csv('I.csv', index = False, encoding='utf-8')
    #test_set.to_csv('J.csv', index = False, encoding='utf-8')
    return dev_set, test_set
    
def get_feats(sample_A, sample_B):
    match_t = em.get_tokenizers_for_matching()
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(sample_A)
    atypes2 = em.get_attr_types(sample_B)
    match_c = em.get_attr_corres(sample_A, sample_B)
    match_c['corres'] = [('title', 'song'),('artist_name', 'artists')]
    match_f = em.get_features(sample_A, sample_B, atypes1, atypes2, match_c, match_t, match_s)
    return match_f
    
def train_fvs(dev_set, match_f):
    ''' get feature vectors for train set '''
    H = em.extract_feature_vecs(dev_set, feature_table=match_f, attrs_before = ['_id', 'ltable_id', 'rtable_id'], attrs_after='gold_labels')
    return H

def comp_matchers(H, attrs_from_table, attrs_to_be_excluded):
    # Create set of ML matchers
    dt = em.DTMatcher(name='DecisionTree', random_state=4)
    svm = em.SVMMatcher(name='SVM', random_state=12)
    rf = em.RFMatcher(name='RF', random_state=42)
    lg = em.LogRegMatcher(name='LogReg', random_state=88)
    ln = em.LinRegMatcher(name='LinReg')
    nb = em.NBMatcher(name='NaiveBayes')

    # Select best ML matcher using CV
    # precision
    presult = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H, 
            exclude_attrs=attrs_to_be_excluded, k=10, 
            target_attr='gold_labels', metric='precision', random_state=0)
    # recall
    rresult = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H, 
            exclude_attrs=attrs_to_be_excluded, k=10,
            target_attr='gold_labels', metric='recall', random_state=1)
    # F1
    f1result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H, 
            exclude_attrs=attrs_to_be_excluded, k=10,
            target_attr='gold_labels', metric='f1', random_state=2)
    print 'precision: ', '\n', presult['cv_stats']
    print 'recall: ', '\n', rresult['cv_stats']
    print 'F1: ', '\n', f1result['cv_stats']
    
def debug_rf(H):
    # using GUI    
    rf = em.RFMatcher(name='RF', random_state=42)
    train_test = em.split_train_test(H, 0.7)
    train, test = train_test['train'], train_test['test']
    em.vis_debug_rf(rf, train, test, exclude_attrs=['_id', 'ltable_id', 'rtable_id'], target_attr='gold_labels')

def use_test_set(H, test_set, match_f, attrs_from_table, attrs_to_be_excluded):
    # test set to feature vectors 
    L = em.extract_feature_vecs(test_set, feature_table=match_f,
                                 attrs_before= ['_id', 'ltable_id', 'rtable_id'],
                                 attrs_after='gold_labels')
    
    # Create set of ML matchers
    dt = em.DTMatcher(name='DecisionTree', random_state=89)
    svm = em.SVMMatcher(name='SVM', random_state=99)
    rf = em.RFMatcher(name='RF', random_state=5)
    lg = em.LogRegMatcher(name='LogReg', random_state=75)
    ln = em.LinRegMatcher(name='LinReg')
    nb = em.NBMatcher(name='NaiveBayes')
    
    # Train using feature vectors from dev set 
    dt.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold_labels')
    svm.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold_labels')
    rf.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold_labels')
    lg.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold_labels')
    ln.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold_labels')
    nb.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold_labels')
    
    # Predict on test set L
    dtpred = dt.predict(table=L, exclude_attrs=attrs_to_be_excluded, 
                  append=True, target_attr='predicted', inplace=False)
    svmpred = svm.predict(table=L, exclude_attrs=attrs_to_be_excluded, 
                  append=True, target_attr='predicted', inplace=False)
    rfpred = rf.predict(table=L, exclude_attrs=attrs_to_be_excluded, 
              append=True, target_attr='predicted', inplace=False)
    lgpred = lg.predict(table=L, exclude_attrs=attrs_to_be_excluded, 
                  append=True, target_attr='predicted', inplace=False)
    lnpred = ln.predict(table=L, exclude_attrs=attrs_to_be_excluded, 
              append=True, target_attr='predicted', inplace=False)
    nbpred = nb.predict(table=L, exclude_attrs=attrs_to_be_excluded, 
              append=True, target_attr='predicted', inplace=False)
    
    # Evaluate predictions
    dteval = em.eval_matches(dtpred, 'gold_labels', 'predicted')
    svmeval = em.eval_matches(svmpred, 'gold_labels', 'predicted')
    rfeval = em.eval_matches(rfpred, 'gold_labels', 'predicted')
    lgeval = em.eval_matches(lgpred, 'gold_labels', 'predicted')
    lneval = em.eval_matches(lnpred, 'gold_labels', 'predicted')
    nbeval = em.eval_matches(nbpred, 'gold_labels', 'predicted')
    print 'DT:'
    em.print_eval_summary(dteval)
    print 'RF:'
    em.print_eval_summary(rfeval)
    print 'SVM:'
    em.print_eval_summary(svmeval)
    print 'LinReg:'
    em.print_eval_summary(lneval)
    print 'LogReg:'
    em.print_eval_summary(lgeval)
    print 'NaiveBayes:'
    em.print_eval_summary(nbeval)
    
def main():
    sample_A = em.read_csv_metadata(argv[1], key='id')
    sample_B = em.read_csv_metadata(argv[2], key='id')
    dev_set = em.read_csv_metadata(argv[3], key='_id',ltable=sample_A, rtable=sample_B, fk_ltable='ltable_id', fk_rtable='rtable_id')
    test_set = em.read_csv_metadata(argv[4], key='_id',ltable=sample_A, rtable=sample_B, fk_ltable='ltable_id', fk_rtable='rtable_id')
        
    # get features for feature vectors
    match_f = get_feats(sample_A, sample_B)
    
    # Select attributes to be included in feature vector table
    attrs_from_table = ['ltable_title', 'ltable_artist_name', 'rtable_title', 'rtable_artists']
    # attributes to exclude for training
    attrs_to_be_excluded = ['_id', 'ltable_id', 'rtable_id', 'gold_labels']

    H = train_fvs(dev_set, match_f)
    #em.to_csv_metadata(H, './H.csv')
    print 'Comparing Matchers:'
    comp_matchers(H, attrs_from_table, attrs_to_be_excluded)

    print
    print 'Metrics on Test Set:'
    use_test_set(H, test_set, match_f, attrs_from_table, attrs_to_be_excluded)


if __name__ == "__main__":
    main()
    