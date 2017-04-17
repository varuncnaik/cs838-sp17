# Usage: 
# yelp: python jsontocsv.py yelp_restaurants.json yelp_restaurants.csv n
# zomato: python jsontocsv.py zomato_restaurants.json zomato_restaurants.csv y

import json
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import OrderedDict

def json_to_csv(in_json, out_csv, iszomato):
    with open(in_json, 'r') as f:
        first_line = json.loads(f.readline(), object_pairs_hook=OrderedDict)
    header = first_line.keys()
    if iszomato == "y":
        header.extend(first_line['location'].keys())
    output_file = open(out_csv, 'w')
    csvwriter = csv.DictWriter(output_file, fieldnames = header)
    csvwriter.writeheader()
    with open(in_json) as f:
        for line in f:
            json_line = json.loads(line, object_pairs_hook=OrderedDict)
            if iszomato == "y":
                location = json_line['location']
                json_line.update(location)
            csvwriter.writerow(json_line)
    output_file.close()
    
def main():
    input_json = sys.argv[1]
    output_csv = sys.argv[2]
    zomato = sys.argv[3]
    json_to_csv(input_json, output_csv, zomato)
    
if __name__ == "__main__":
    main()
