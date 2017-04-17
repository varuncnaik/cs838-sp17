'''
Usage: python merge_tables.py
'''

import py_entitymatching as em
import match_magellan as mm
import pandas as pd

def get_all_matches():
    ''' based on the train & test set, Logistic Regression is the winner '''
    A = em.read_csv_metadata('A.csv', key='business_id')
    B = em.read_csv_metadata('B.csv', key='id')
    C = em.read_csv_metadata('C.csv', key='_id',ltable=A, rtable=B, fk_ltable='ltable_business_id', fk_rtable='rtable_id')
    match_f = mm.get_feats(A, B)
    K = em.extract_feature_vecs(C, feature_table=match_f,
                                 attrs_before= ['_id', 'ltable_business_id', 'rtable_id'])
    K = K.fillna(1)
    attrs_to_exclude = ['_id', 'ltable_business_id', 'rtable_id']
    G = em.read_csv_metadata('G.csv', key='_id', ltable=A, rtable=B, fk_ltable='ltable_business_id', fk_rtable='rtable_id')
    Gfvs = mm.train_fvs(G, match_f)
    Gfvs = Gfvs.fillna(1)
    
    #em.to_csv_metadata(K, './K.csv')
    K = em.read_csv_metadata('K.csv', key='_id', ltable=A, rtable=B, fk_ltable='ltable_business_id', fk_rtable='rtable_id')

    lg = em.LogRegMatcher(name='LogReg', random_state=77)
    lg.fit(table=Gfvs, exclude_attrs=attrs_to_exclude, target_attr='gold_labels')
    
    predictions = lg.predict(table=K, exclude_attrs=attrs_to_exclude, 
              append=True, target_attr='predicted', inplace=False)
    
    matches = predictions.loc[predictions['predicted'] == 1]
    return matches
    
def matchesfvs_to_orig(matches):
    ''' output of get_all_matches() is in feature vector form
        merge with original csv files
    '''
    Y = em.read_csv_metadata('yelp_restaurants.csv', key='business_id')
    Z = em.read_csv_metadata('zomato_restaurants.csv', key='id')
    matches_y = matches[['ltable_business_id', 'rtable_id']].merge(Y, left_on='ltable_business_id', right_on='business_id')
    matches_y_z = matches_y.merge(Z, left_on='rtable_id', right_on='id')   
    em.to_csv_metadata(matches_y_z, './matches.csv')
    
def cleanup(M):
    ''' 
    Drop repetitive or unnecessary columns.
    For Zomato, replace full addresses with short addresses: 
        the first part of an address before a comma, 
        since we already have city, state, zipcode attributes.
    Take Yelp's city_x over Zomato's similar locality attribute.
    Take Yelp's postal_code over Zomato's (which are missing some zipcodes).
    '''
    # drop repetitive columns
    M.pop('business_id')
    M.pop('id')
    # drop columns which offer no information
    M.pop('state') # are all 'WI'
    M.pop('type') # are all 'business'
    M.pop('R') # same as zomato id
    M.pop('apikey') 
    M.pop('url') 
    M.pop('location') # already extracted to separate columns in jsontocsv.py
    M.pop('switch_to_order_menu') # all 0
    M.pop('currency') # all $
    M.pop('offers') # all []
    M.pop('thumb')
    M.pop('photos_url')
    M.pop('menu_url')
    M.pop('featured_image')
    M.pop('has_online_delivery') # all 0
    M.pop('is_delivering_now') # all 0
    M.pop('deeplink')
    M.pop('has_table_booking') # all 0
    M.pop('events_url')
    M.pop('establishment_types') # all []
    M.pop('city_y') # from Zomato, all Madison
    M.pop('city_id') # from Zomato, all Madison
    M.pop('country_id') # all 216 (probably U.S.)
    # locality_verbose is just locality + 'Madison'
    #M['locality_verbose'] = M['locality_verbose'].str.extract('^(.+?),') 
    #M['locality_verbose'].equals(M['locality'])
    M.pop('locality_verbose')
    M['address_y'] = M['address_y'].str.extract('^(.+?),')
    #M['city_x'].value_counts()
    #M['locality'].value_counts()
    # take yelp's city_x over zomato's more finer locality
    M.pop('locality')
    #M['comp_zips'] = M.postal_code == M.zipcode
    # take yelp's postal_code over zomato's zipcode
    M.pop('zipcode')
    
    # rename for clarity
    M = M.rename(columns={'ltable_business_id': 'yelp_id', 'rtable_id': 'zomato_id', 'postal_code': 'zipcode', 'city_x': 'city'})

    return M
    
'''
None of the compute_* functions should perform side effects. See the docs for
pandas.DataFrame.apply() for more information.
'''

def compute_name(row):
    '''
    Chooses the name with the longer length.
    '''
    if len(row['name_x']) < len(row['name_y']):
        row['name'] = row['name_y']
    else:
        row['name'] = row['name_x']
    return row

def compute_address(row):
    '''
    Chooses the address with the longer length.
    '''
    if len(row['address_x']) < len(row['address_y']):
        row['address'] = row['address_y']
    else:
        row['address'] = row['address_x']
    return row

def compute_latitude(row):
    '''
    Computes the latitude as the average (arithmetic mean) of latitude_x and
    and latitude_y. Assumes that none of the latitudes are 0.
    '''
    row['latitude'] = (row['latitude_x'] + row['latitude_y']) / 2
    return row

def compute_longitude(row):
    '''
    Computes the longitude as the average (arithmetic mean) of longitude_x and
    longitude_y. Assumes that none of the longitudes are 0.
    '''
    row['longitude'] = (row['longitude_x'] + row['longitude_y']) / 2
    return row

def main():
    matchesfvs_to_orig(get_all_matches())
    
    uncleanM = em.read_csv_metadata('./matches_cleaned.csv')
    
    M = cleanup(uncleanM)
    
    M['name'] = pd.Series()
    M = M.apply(compute_name, axis=1)
    M.pop('name_x')
    M.pop('name_y')
    
    M['address'] = pd.Series()
    M = M.apply(compute_address, axis=1)
    M.pop('address_x')
    M.pop('address_y')
    
    M['latitude'] = pd.Series()
    M = M.apply(compute_latitude, axis=1)
    M.pop('latitude_x')
    M.pop('latitude_y')
    
    M['longitude'] = pd.Series()
    M = M.apply(compute_longitude, axis=1)
    M.pop('longitude_x')
    M.pop('longitude_y')
    
    em.to_csv_metadata(M, './E.csv')

if __name__ == '__main__':
    main()
