# Usage: python make_all_baskets.py E.csv yelp_reviews_user_id_business_id
# Example: python make_all_baskets.py E.csv yelp_reviews_user_id_business_id

import pandas as pd
import sys

# Each line in a basket is a comma-separated list of values, as described at:
# https://orange.readthedocs.io/en/latest/reference/rst/Orange.associate.html

def clean_categories(categories):
    '''
    Takes a 'categories' string and returns a line in the categories basket.
    Replaces spaces with underscores in each item because when a rule is
    printed, items are separated by spaces.
    '''
    return ", ".join(s.replace(" ", "_") for s in eval(categories))

def clean_cuisines(cuisines):
    '''
    Takes a 'cuisines' string and returns a line in the cuisines basket.
    Replaces spaces with underscores in each item because when a rule is
    printed, items are separated by spaces.
    '''
    return ", ".join(s.replace(" ", "_") for s in cuisines.split(", "))


def extract_good_for_meal(attributes):
    '''
    Takes an 'attributes' string and returns a line in the good for meal basket.
    '''
    if attributes == "":
        return ""
    good_for_meal_str = None
    for attribute in eval(attributes):
        if attribute.startswith("GoodForMeal: "):
            assert good_for_meal_str is None
            good_for_meal_str = attribute[len("GoodForMeal: "):]
    if good_for_meal_str is None:
        return ""

    good_for_meal_list = []
    for k, v in eval(good_for_meal_str).iteritems():
        assert k in ("dessert", "latenight", "dinner", "lunch", "breakfast", "brunch")
        assert v in (True, False)
        if v:
            good_for_meal_list.append(k)
    good_for_meal_list.sort()
    return ", ".join(good_for_meal_list)

def extract_ambiences(attributes):
    '''
    Takes an 'attributes' string and returns a line in the ambiences basket.
    '''
    if attributes == "":
        return ""

    ambience_str = None
    for attribute in eval(attributes):
        if attribute.startswith("Ambience: "):
            assert ambience_str is None
            ambience_str = attribute[len("Ambience: "):]
    if ambience_str is None:
        return ""

    ambience_list = []
    for k, v in eval(ambience_str).iteritems():
        assert k in ("romantic", "intimate", "classy", "hipster", "divey", "touristy", "trendy", "upscale", "casual")
        assert v in (True, False)
        if v:
            ambience_list.append(k)
    ambience_list.sort()
    return ", ".join(ambience_list)

def write_basket(col, filename):
    '''
    Writes col, an iterable containing the lines of a basket, to the specified
    file.
    '''
    with open(filename, "w") as f:
        for val in col:
            f.write(val + "\n")
            
def combine_baskets(col1, col2, filename):
    '''
    Writes combined col1 and col2 to the specified file.
    '''
    with open(filename, "w") as f:
        for i, val in enumerate(col1):
            if not val:
                f.write(col2[i] + "\n")
            else:
                f.write(val + ", " + col2[i] + "\n")



def join_E_with_user_review(df1, user_review_file):
    '''
    Joins E.csv with yelp_academic_user_reviews.csv
    '''
    df2 = pd.read_csv(user_review_file)
      
    user_review_combined_df = pd.merge(df1, df2, how='inner', left_on=['yelp_id'], right_on=['business_id'])
    
    return user_review_combined_df

def all_user_restaurant_attrs(df1, user_review_file, group_by_attr):
    user_review_combined_df = join_E_with_user_review(df1, user_review_file)
    user_review_combined_df_col = user_review_combined_df.groupby('user_id')[group_by_attr].apply(lambda x: "%s" % ', '.join(x))
    return user_review_combined_df_col
    

def main():
    assert len(sys.argv) > 1, "Wrong number of arguments"

    df = pd.read_csv(sys.argv[1], dtype=str)
    df.fillna("", inplace=True)

    all_cuisines = df["cuisines"].apply(clean_cuisines)
    write_basket(all_cuisines, "cuisines.basket")

    all_categories = df["categories"].apply(clean_categories)
    write_basket(all_categories, "categories.basket")

    all_ambiences = df["attributes"].apply(extract_ambiences)
    write_basket(all_ambiences, "ambiences.basket")
    
    all_good_for_meal = df["attributes"].apply(extract_good_for_meal)
    write_basket(all_good_for_meal, "good_for_meal.basket")

    all_stars = df["stars"]
    
    all_avg_cost_for_two = df["average_cost_for_two"]
    
    if (len(sys.argv) > 2):
        #identify correlation in the categories of restaurants reviewed by users 
        df["categories"] = df["categories"].apply(clean_categories)
        all_user_categories = all_user_restaurant_attrs(df, sys.argv[2], "categories")
        write_basket(all_user_categories, "categories_of_restaurants_reviewed_by_users.basket")
        
        #identify correlation in the cuisines of restaurants reviewed by users 
        df["cuisines"] = df["cuisines"].apply(clean_cuisines)
        all_user_cuisines = all_user_restaurant_attrs(df, sys.argv[2], "cuisines")
        write_basket(all_user_cuisines, "cuisines_of_restaurants_reviewed_by_users.basket")
        
        #identify correlation in the ambience of restaurants reviewed by users 
        df["ambiences"] = df["attributes"].apply(extract_ambiences)
        all_user_ambiences = all_user_restaurant_attrs(df, sys.argv[2], "ambiences")
        write_basket(all_user_ambiences, "ambiences_of_restaurants_reviewed_by_users.basket")

    
    combine_baskets(all_cuisines, all_stars, "cuisines_stars.basket")
    
    combine_baskets(all_categories, all_stars, "categories_stars.basket")
    
    combine_baskets(all_ambiences, all_stars, "ambiences_stars.basket")

    combine_baskets(all_cuisines, all_avg_cost_for_two, "cuisines_avg_cost.basket")
    
    combine_baskets(all_ambiences, all_avg_cost_for_two, "ambiences_avg_cost.basket")
    
    combine_baskets(all_good_for_meal, all_cuisines, "meal_cuisines.basket")
    
    
if __name__ == "__main__":
    main()
