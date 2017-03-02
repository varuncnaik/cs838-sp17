#!/bin/python

"""
Usage: python generate_feature_vectors.py < joined_reviews.json > feature_vectors.json

Prints each review, with an additional key "fvs", to stdout.
Note that this can be combined with the pipeline for join_reviews.py.
"""

import json
import fileinput
import random
import re

from collections import OrderedDict

def get_negative_example_bool(rgen):
    """
    Helper function that returns True if a new negative example should be started, or False otherwise.
    rgen: random number generator, which is called exactly once
    """
    return rgen.random() < 0.08

def get_negative_example_length(rgen):
    """
    Helper function that returns the length of the negative example.
    rgen: random number generator, which is called exactly once
    """
    return rgen.randint(1, 6)

def bool_to_int(b):
    """
    Helper function that converts True to 1 and False to 0, and crashes on any other value
    b: the value to be converted
    """
    assert isinstance(b, bool)
    return 1 if b else 0

def build_feature_vector(review, raw_review_sentences, parsed_review_sentences, sentence_idx, example_first_word_idx, example_word_length, is_positive):
    """
    This function is called for each positive or negative example.
    An example is part of a sentence. A sentence is part of the review text.
    Returns a feature vector as a Python dictionary.
    Arguments are ordered from most general to most specific.

    review: dict containing all information about the review
    raw_review_sentences: list of the sentences in the review text
    parsed_review_sentences: list of (word in raw sentence, index of first character in word from start of raw sentence, index of last character in word from start of raw sentence)
    sentence_idx: the index (in raw_review_sentences and parsed_review_sentences) of the sentence containing the example
    example_first_word_idx: index (within the parsed sentence) of the first word of the example
    example_word_length: number of words in the example
    is_positive: True if the example is positive, or False otherwise
    """
    common_cooking_styles = ["baked", "fried", "oven-roasted", "grilled", "roasted"]
    restaurant_name_splits = review["restaurant"]["name"].split(" ")
    common_food_names = ["pizza", "burger", "sauce", "cheese", "bread"]
    common_food_adjectives = ["tasty", "delicious", "spicy", "acidic", "sweet"]

    raw_review_sentence = raw_review_sentences[sentence_idx]
    parsed_review_sentence = parsed_review_sentences[sentence_idx]

    text = " ".join(parsed_review_sentence[i][0] for i in xrange(example_first_word_idx, example_first_word_idx + example_word_length)).replace("<", "").replace(">", "").strip().encode("utf-8")
    raw_text = raw_review_sentence[parsed_review_sentence[example_first_word_idx][1]:parsed_review_sentence[example_first_word_idx + example_word_length - 1][2]]

    #f1
    example_len = len(raw_text)

    #f2
    ends_with_s = parsed_review_sentence[example_first_word_idx + example_word_length - 1][0].endswith("s")

    #f3
    is_title = all(parsed_review_sentence[i][0].istitle() for i in xrange(example_first_word_idx, example_first_word_idx + example_word_length))

    #f4
    has_cooking_style = any(s in raw_text for s in common_cooking_styles)

    #f5
    ends_with_common_food_names = any(s in raw_text for s in common_food_names)

    #f6
    has_restaurant_name = False
    if (len(text.split(' ')) > 0):
        if any(s in raw_text for s in restaurant_name_splits):
            has_restaurant_name = True
        else:
            has_restaurant_name = False
    
    #f7
    relative_position_of_word_in_sentence = float(parsed_review_sentence[example_first_word_idx][1])/len(raw_review_sentence)
    
    #f8
    sentence_contains_adjective_words = any(s in raw_review_sentence for s in common_food_adjectives)

    #f9
    if example_first_word_idx == 0:
        is_prev_word_a_definite_artice = False
    else:
        prev_word = parsed_review_sentence[example_first_word_idx-1][0]
        assert isinstance(prev_word, unicode)
        is_prev_word_a_definite_artice = (prev_word.replace("<", "").replace(">", "").lower() == u"the")

    #f10
    contains_the = any(parsed_review_sentence[i][0].replace("<", "").replace(">", "").lower() == u"the" for i in xrange(example_first_word_idx, example_first_word_idx + example_word_length))

    #f11
    num_commas = raw_review_sentence.count(",")

    #f12
    sentence_word_length = len(parsed_review_sentence)

    #f13: example_word_length
    
    #f14
    capital = text[0].isupper()

    #f15
    word_ends_with_ing = any(parsed_review_sentence[i][0].replace("<", "").replace(">", "").lower().endswith("ing") for i in xrange(example_first_word_idx, example_first_word_idx + example_word_length))

    #f16
    word_ends_with_ed = any(parsed_review_sentence[i][0].replace("<", "").replace(">", "").lower().endswith("ed") for i in xrange(example_first_word_idx, example_first_word_idx + example_word_length))

    #f17
    ends_with_es = parsed_review_sentence[example_first_word_idx + example_word_length - 1][0].endswith("es")

    #f18
    if example_first_word_idx == 0:
        is_prev_word_an_indefinite_artice = False
    else:
        prev_word = parsed_review_sentence[example_first_word_idx-1][0]
        assert isinstance(prev_word, unicode)
        is_prev_word_an_indefinite_artice = (prev_word.replace("<", "").replace(">", "").lower() in (u"a", u"an"))

    #f19
    contains_indefinite_article = any(parsed_review_sentence[i][0].replace("<", "").replace(">", "").lower() in (u"a", u"an") for i in xrange(example_first_word_idx, example_first_word_idx + example_word_length))

    #f20
    word_fraction_capitalized = sum(1 if parsed_review_sentence[i][0].istitle() else 0 for i in xrange(example_first_word_idx, example_first_word_idx + example_word_length)) / example_word_length

    #f21
    word_position = example_first_word_idx / len(parsed_review_sentence)

    #f22
    contains_number = any(str(i) in text for i in xrange(0, 10))
   
    # Construct the feature vector
    fv = OrderedDict([
        ("length", example_len),                                                                #f1
        ("ends_with_s", bool_to_int(ends_with_s)),                                              #f2
        #("all_words_capitalized", bool_to_int(is_title)),                                       #f3
        ("has_cooking_style", bool_to_int(has_cooking_style)),                                  #f4
        ("ends_with_common_food_names", bool_to_int(ends_with_common_food_names)),              #f5
        ("has_restaurant_name", bool_to_int(has_restaurant_name)),                              #f6
        #("relative_position_of_word_in_sentence", relative_position_of_word_in_sentence),       #f7
        ("sentence_contains_adjective_words", bool_to_int(sentence_contains_adjective_words)),  #f8
        ("is_prev_word_a_definite_artice", bool_to_int(is_prev_word_a_definite_artice)),        #f9
        ("contains_the", bool_to_int(contains_the)),                                            #f10
        #("num_commas", num_commas),                                                             #f11
        #("sentence_word_length", sentence_word_length),                                         #f12
        ("example_word_length", example_word_length),                                           #f13
        ("capital", bool_to_int(capital)),                                                      #f14
        #("word_ends_with_ing", bool_to_int(word_ends_with_ing)),                                #f15
        #("word_ends_with_ed", bool_to_int(word_ends_with_ed)),                                  #f16
        #("ends_with_es", bool_to_int(ends_with_es)),                                            #f17
        #("is_prev_word_an_indefinite_artice", bool_to_int(is_prev_word_an_indefinite_artice)),  #f18
        #("contains_indefinite_article", bool_to_int(contains_indefinite_article)),              #f19
        ("word_fraction_capitalized", word_fraction_capitalized),                               #f20
        #("word_position", word_position),                                                       #f21
        #("contains_number", bool_to_int(contains_number)),                                      #f22
        ("is_positive", bool_to_int(is_positive)),  
        ("text", text),                                       
    ])

    # Sanity checks before returning fv
    assert all(isinstance(k, str) for k in fv)
    assert all(type(v) in (int, long, float) for _, v in fv.items()[:-1])
    assert isinstance(text, str)

    return fv

def process_review(review, rgen1, rgen2):
    """
    Returns a Python list of feature vectors.
    review: A parsed JSON review (a Python dictionary)
    rgen*: Random number generators
    """
    # Create a list of feature vectors, because a review may have multiple feature vectors
    fvs = []
    
    # Split the review text into sentences,
    # where each sentence is split into word tuples,
    # where each word tuple is (word, character index of first character, character index of last character)
    raw_review_sentences = []
    parsed_review_sentences = []
    for raw_review_sentence in review["text"].split("."):
        raw_review_sentences.append(raw_review_sentence)
        parsed_review_sentence = []
        # Must call re.finditer(), not re.findall(), because we want all match information
        for word_match in re.finditer(r"[\w<>][\w<>]*", raw_review_sentence, flags=re.UNICODE):
            parsed_review_sentence.append((word_match.group(0), word_match.start(0), word_match.end(0)))
        parsed_review_sentences.append(parsed_review_sentence)
    
    for sentence_idx, parsed_review_sentence in enumerate(parsed_review_sentences):
        positive_example = []
        negative_example = []
        negative_example_length = None
        for word_idx, (word, start_idx, end_idx) in enumerate(parsed_review_sentence):
            assert word != "" and "<" not in word[1:] and ">" not in word[:-1]
     
            # If the current word is part of a positive example
            if len(positive_example) > 0:
                assert word[0] != "<"
                positive_example.append((word, start_idx, end_idx))
            elif word[0] == "<":
                assert len(positive_example) == 0
                positive_example = [(word, start_idx, end_idx)]
            
            # If the current word is part of a negative example
            rand_bool = get_negative_example_bool(rgen1)
            rand_length = get_negative_example_length(rgen2)
            assert rand_length >= 1
            if len(negative_example) > 0:
                assert negative_example_length > 0
                negative_example.append((word, start_idx, end_idx))
                negative_example_length -= 1
            elif rand_bool:
                assert len(negative_example) == 0
                assert negative_example_length is None
                negative_example_length = rand_length - 1
                negative_example = [(word, start_idx, end_idx)]
     
            # If the current word ends a positive sample
            if word[-1] == ">":
                assert len(positive_example) > 0
                assert positive_example[0][0][0] == "<"
                assert positive_example[-1][0][-1] == ">"
                assert all(len(i) == 3 for i in positive_example)
                fvs.append(build_feature_vector(review, raw_review_sentences, parsed_review_sentences, sentence_idx, word_idx - len(positive_example) + 1, len(positive_example), True))
                positive_example = []
     
            # If the current word ends a negative sample
            if negative_example_length == 0:
                assert len(negative_example) > 0
                assert all(len(i) == 3 for i in negative_example)
                if not (negative_example[0][0][0] == "<" and negative_example[-1][0][-1] == ">"):
                    fvs.append(build_feature_vector(review, raw_review_sentences, parsed_review_sentences, sentence_idx, word_idx - len(negative_example) + 1, len(negative_example), False))
                negative_example = []
                negative_example_length = None

        assert len(positive_example) == 0
        assert (len(negative_example) == 0 and negative_example_length is None) or (len(negative_example) > 0 and negative_example_length > 0)
        # negative_example may be non-empty here, but we simply ignore it

    return fvs

def main():
    rgen1 = random.Random()
    rgen2 = random.Random()
    rgen1.seed(314)
    rgen2.seed(42)

    # Read one line from stdin at a time
    for line in fileinput.input():
        review = json.loads(line, object_pairs_hook=OrderedDict)
        fvs = process_review(review, rgen1, rgen2)

        # Probably printing out more information than necessary...
        review["fvs"] = fvs
        print json.dumps(review, separators=(",", ":"))

if __name__ == "__main__":
    main()
