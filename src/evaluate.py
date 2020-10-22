'''
This script takes a gold parse file in CONLL format (10 columns, with the parent 
in column 7, and the relation in column 8), and evaluates a predicted parse file 
in the same format against it.

The metrics produced are LAS, UAS, and label.

There is no exception handling for file with different number of tokens.
The words themselves, their POS tags, and other features are not considered in 
this evaluation.  
'''

import sys

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('Usage: python evaluate.py <gold_file> <predicted_file>')
		sys.exit("Missing input argument.")

	gold_file = sys.argv[1]
	parse_file = sys.argv[2]

	number_of_tokens = 0
	LAS = 0
	UAS = 0
	label = 0

	with open(gold_file) as gold, open(parse_file) as parse:         
		
		for gold_line, parse_line in zip(gold, parse):
			if gold_line.rstrip() == '' and parse_line.rstrip() == '':
				pass

			else:
				number_of_tokens += 1

				gold_line_tokens = gold_line.rstrip().split()
				parse_line_tokens = parse_line.rstrip().split()

				gold_parent = gold_line_tokens[6]
				gold_relation = gold_line_tokens[7]

				parse_parent = parse_line_tokens[6]
				parse_relation = parse_line_tokens[7]

				if gold_parent == parse_parent and gold_relation == parse_relation:
					LAS += 1

				if gold_parent == parse_parent:
					UAS += 1

				if gold_relation == parse_relation:
					label += 1

	print("Number of tokens in the test set = " + str(number_of_tokens))
	print("LAS = {0:.2f}%".format((LAS / number_of_tokens) * 100))
	print("UAS = {0:.2f}%".format((UAS / number_of_tokens) * 100))
	print("Label = {0:.2f}%".format((label / number_of_tokens) * 100))
