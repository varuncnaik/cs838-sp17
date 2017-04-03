# Stage 0: Form Team

Lokananda Dhage

Mary Feng

[Varun Naik](https://www.linkedin.com/in/varuncnaik)

# Stage 1: Problem Definition

We are planning to analyze information about restaurants in the Madison, WI
area. We obtained data from the [Zomato](https://www.zomato.com/) API and the
[Yelp](https://www.yelp.com/) dataset challenge. Each Yelp review and Zomato
review will be one of our text documents for Stage 2.

* [Report](reports/Stage1_Report.pdf)

* [Yelp restaurants](datasets/yelp_restaurants.json)
* [Yelp checkins](datasets/yelp_checkins.json)
* [Yelp reviews](datasets/yelp_reviews.json)
* [Yelp tips](datasets/yelp_tips.json)
* [Yelp users](datasets/yelp_users.json)

* [Zomato restaurants](datasets/zomato_restaurants.json)
* [Zomato reviews](datasets/zomato_reviews.json)

# Stage 2: Information Extraction

We performed information extraction on 300 randomly selected Yelp reviews.

* [Report](reports/Stage2_Report.pdf)

* [300 Yelp reviews](stage2/documents.md)
* [Development set (I)](stage2/dev_set.md)
* [Test set (J)](stage2/test_set.md)
* [Source code](https://github.com/varuncnaik/cs838-sp17/tree/master/stage2/code)
* [Zipped documents and code](stage2/stage2.zip)

# Stage 3: Entity Matching

Since our Yelp/Zomato dataset had fewer than 3,000 tuples in each table, we
switched to a different dataset for this stage of the project. We performed
entity matching between a Song table with 961,593 tuples, and a Track table
with 734,485 tuples.

* [Report](reports/Stage3_Report.pdf)

* [CODE](stage3/CODE.md)
* [DATA](stage3/DATA.md)

# Stage 4

Coming soon!
