''' 
Usage: 
    python split_reviews_dev_test.py feature_vectors.json dev_set.json test_set.json

Randomize order of the 300 labeled reviews and split this into two sets: 
    Dev set I containing 200 reviews
    Test set J containing the remaining 100 reviews
'''

import random
from sys import argv

def get_reviews(labeled_reviews):
    with open(labeled_reviews, 'r') as f:
        reviews = [line for line in f]
    return reviews

def split_dev_test(labeled_reviews):
    random.seed(8)
    all_reviews = get_reviews(labeled_reviews)
    random.shuffle(all_reviews)
    I = all_reviews[:200]
    J = all_reviews[200:]
    return I, J

def main(labeled_reviews, train_out_filename, test_out_filename):
    train, test = split_dev_test(labeled_reviews)
    with open(train_out_filename, 'w') as train_out:
        train_out.writelines(train)
    with open(test_out_filename, 'w') as test_out:
        test_out.writelines(test)

if __name__ == "__main__":
    main(argv[1], argv[2], argv[3])