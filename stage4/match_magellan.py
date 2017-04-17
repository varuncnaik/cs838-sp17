# -*- coding: utf-8 -*-
import py_entitymatching as em
import re

def get_tables(A_file, B_file):
    '''
    Get Yelp & Zomato csv files
    Extract address_num: for example, 123 from 123 Main St
    '''
    A = em.read_csv_metadata(A_file, key='business_id')
    B = em.read_csv_metadata(B_file, key='id')
    A['address_num'] = A['address'].str.extract(r'(\d+)')
    B['address_num'] = B['address'].str.extract(r'(\d+)')
    #em.to_csv_metadata(A, './A.csv')
    #em.to_csv_metadata(B, './B.csv')
    
def match(ltup, rtup):
    '''
    Returns True if (ltup, rtup) should be dropped, or False if (ltup, rtup) is
    a candidate. Called by bb_block().
    '''
    l_latitude = ltup['latitude']
    r_latitude = rtup['latitude']
    l_longitude = ltup['longitude']
    r_longitude = rtup['longitude']
    # block out missing latitudes/longitudes; 
    # otherwise they cause issues when generating feature vectors due to abs_norm
    if l_longitude == 0 or l_latitude == 0 or r_longitude == 0 or r_latitude == 0:
        return True

    # check for overlap in restaurant name and address
    stopwords = ('&', 'the', 'n', 's', 'e', 'w', 'st', 'st.', 'dr', 'dr.', 'rd', 'rd.', 'ln', 'ln.')
    l_namewithstop = set(str(ltup['name']).lower().split())
    r_namewithstop = set(str(rtup['name']).lower().split())
    l_name = set(w for w in l_namewithstop if w not in stopwords)
    r_name = set(w for w in r_namewithstop if w not in stopwords)
    l_addwithstop = set(str(ltup['address']).lower().split())
    r_addwithstop = set(str(rtup['address']).lower().split())
    l_add = set(w for w in l_addwithstop if w not in stopwords)
    r_add = set(w for w in r_addwithstop if w not in stopwords)

    return l_name.isdisjoint(r_name) or l_add.isdisjoint(r_add)

def bb_block(A, B):
    '''
    Returns a DataFrame of candidate pairs by performing black-box blocking.
    '''
    # Create black box blocker
    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(match)
    bbC = bb.block_tables(A, B)
    return bbC
    
def overlap_block(A, B):
    A['name2'] = A['name'].str.lower()
    B['name2'] = B['name'].str.lower()
    ob = em.OverlapBlocker()
    ob.regex_punctuation = re.compile(r'')
    obC = ob.block_tables(A, B, 'name2', 'name2',
                        l_output_attrs=['business_id', 'name2', 'address', 'address_num', 'postal_code', 'latitude', 'longitude'],
                        r_output_attrs=['id', 'name2', 'address', 'address_num', 'zipcode', 'latitude', 'longitude'],
                        rem_stop_words=True,
    )

    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(match)
    bbC = bb.block_candset(obC)

    return bbC

def debug_block(C, A, B):
    #A['latitude'] = A['latitude'].apply(str)
    #B['latitude'] = B['latitude'].apply(str)
    #A['longitude'] = A['longitude'].apply(str)
    #B['longitude'] = B['longitude'].apply(str)
    return em.debug_blocker(C, A, B, attr_corres=[('name', 'name'), ('address', 'address'), ('latitude','latitude'), ('longitude','longitude')])
    
def samp_label_split(C):
    S = em.sample_table(C, 500)
    G = em.label_table(S, label_column_name='gold_labels')
    #em.to_csv_metadata(G, './G.csv')

    train_test = em.split_train_test(G, train_proportion=0.7)
    dev_set = train_test['train']
    test_set = train_test['test']
    dev_set = dev_set.rename(columns={'ltable_name2': 'ltable_name', 'rtable_name2': 'rtable_name'})
    test_set = test_set.rename(columns={'ltable_name2': 'ltable_name', 'rtable_name2': 'rtable_name'})
    #em.to_csv_metadata(dev_set, './I.csv')
    #em.to_csv_metadata(test_set, './J.csv')
    
def get_feats(A, B):
    match_t = em.get_tokenizers_for_matching()
    match_s = em.get_sim_funs_for_matching()
    atypes1 = em.get_attr_types(A)
    atypes2 = em.get_attr_types(B)
    match_c = em.get_attr_corres(A, B)
    match_c['corres'] = [('name', 'name'), ('address_num','address_num'), ('postal_code', 'zipcode'), ('latitude', 'latitude'), ('longitude', 'longitude')]
    match_f = em.get_features(A, B, atypes1, atypes2, match_c, match_t, match_s)
    return match_f
    
def train_fvs(dev_set, match_f):
    ''' get feature vectors for train set '''
    H = em.extract_feature_vecs(dev_set, feature_table=match_f, attrs_before = ['_id', 'ltable_business_id', 'rtable_id'], attrs_after='gold_labels')
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
    em.vis_debug_rf(rf, train, test, exclude_attrs=['_id', 'ltable_business_id', 'rtable_id'], target_attr='gold_labels')


def use_test_set(H, L, match_f, attrs_from_table, attrs_to_be_excluded):
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
    ''' Using train & test sets to choose best matcher '''
    A = em.read_csv_metadata('A.csv', key='business_id')
    B = em.read_csv_metadata('B.csv', key='id')
    
    #C = overlap_block(A, B)
    #D = debug_block(C, A, B)
    #em.to_csv_metadata(C, './C.csv')
    #samp_label_split(C)

    #I = em.read_csv_metadata('I.csv', key='_id',ltable=A, rtable=B, fk_ltable='ltable_business_id', fk_rtable='rtable_id')
    #J = em.read_csv_metadata('J.csv', key='_id',ltable=A, rtable=B, fk_ltable='ltable_business_id', fk_rtable='rtable_id')

    match_f = get_feats(A, B)
    '''
    H = train_fvs(I, match_f)
    H = H.fillna(1)
    em.to_csv_metadata(H, './H.csv')
    '''
    H = em.read_csv_metadata('H.csv', key='_id',ltable=A, rtable=B, fk_ltable='ltable_business_id', fk_rtable='rtable_id')
    
    # Select attributes to be included in feature vector table
    attrs_from_table = ['ltable_name', 'ltable_latitude', 'ltable_longitude', 'ltable_address_num', 'ltable_postal_code', 'rtable_name', 'rtable_latitude', 'rtable_longitude', 'rtable_address_num', 'rtable_zipcode']
    # attributes to exclude for training
    attrs_to_be_excluded = ['_id', 'ltable_business_id', 'rtable_id', 'gold_labels']

    print 'Comparing Matchers:'
    comp_matchers(H, attrs_from_table, attrs_to_be_excluded)
    
    '''
    # test set to feature vectors 
    L = em.extract_feature_vecs(J, feature_table=match_f,
                                 attrs_before= ['_id', 'ltable_business_id', 'rtable_id'],
                                 attrs_after='gold_labels')
    L = L.fillna(1)
    em.to_csv_metadata(L, './L.csv')
    '''
    L = em.read_csv_metadata('L.csv', key='_id',ltable=A, rtable=B, fk_ltable='ltable_business_id', fk_rtable='rtable_id')
    
    print
    print 'Metrics on Test Set:'
    use_test_set(H, L, match_f, attrs_from_table, attrs_to_be_excluded)
    
    

if __name__ == "__main__":
    main()

    