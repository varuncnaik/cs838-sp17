#!/bin/python

"""
Usage: python join_reviews.py yelp_restaurants.json yelp_users.json labeled_reviews.json > joined_reviews.json

Prints each review, with additional keys "restaurant" and "user", to stdout.
"""

import json
import os
import sys

from collections import OrderedDict

def read_file(filename, key):
    """
    Converts the file of JSON objects to a Python dictionary, whose keys are the unique attribute of each JSON object.
    filename: the name of the file
    key: the JSON attribute that will become a key of the dictionary
    """
    d = {}
    with open(filename, "r") as f:
        for line in f:
            parsed = json.loads(line, object_pairs_hook=OrderedDict)
            key_value = parsed.pop(key)
            assert key_value not in d
            d[key_value] = parsed
    return d

def main():
    assert len(sys.argv) == 4
    assert os.path.exists(sys.argv[1])
    assert os.path.exists(sys.argv[2])
    assert os.path.exists(sys.argv[3])

    restaurants = read_file(sys.argv[1], "business_id")
    users = read_file(sys.argv[2], "user_id")

    with open(sys.argv[3], "r") as reviews_file:
        for line in reviews_file:
            review = json.loads(line, object_pairs_hook=OrderedDict)
            restaurant_id = review["business_id"]
            user_id = review["user_id"]

            review["restaurant"] = restaurants[restaurant_id]
            review["user"] = users[user_id]

            print json.dumps(review, separators=(",", ":"))

if __name__ == "__main__":
    main()
