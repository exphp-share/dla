#!/usr/bin/env python3

import numpy as np
import json
import sys

if sys.stdout.isatty():
	print("Reading from STDIN...", file=sys.stderr)

symbols,indices = zip(*json.load(sys.stdin))
cartesian = np.array(indices)
print(len(cartesian))
print("blah blah blah") # xyz comment
for symbol,(x,y,z) in zip(symbols, cartesian):
	print("{} {} {} {}".format(symbol, x, y, z))
