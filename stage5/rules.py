# Usage: python rules.py <.basket file> <supp>
# Example: python rules.py cuisines.basket 0.01

import sys
import Orange

def rules(basket, supp):
    rules = Orange.associate.AssociationRulesSparseInducer(basket, support=supp)
    print "Supp   Conf    Rule"
    for rule in rules:
        print "%4.4f %4.4f  %s" % (rule.support, rule.confidence, rule)

def main():
    assert len(sys.argv) == 3, "Wrong number of arguments"

    data = Orange.data.Table(sys.argv[1])
    rules(data, float(sys.argv[2]))

if __name__ == "__main__":
    main()
