#!/usr/bin/env python3

import numpy as np
import json
import sys

if sys.stdout.isatty():
	print("Reading from STDIN...", file=sys.stderr)

PARAM_A = 2.4

symbols,indices = zip(*json.load(sys.stdin))

cartesian = PARAM_A * np.array(indices)
print(len(cartesian))
print("blah blah blah") # xyz comment
for symbol,(x,y,z) in zip(symbols, cartesian):
	print("{} {} {} {}".format(symbol, x, y, z))

print("Note: Minima:", np.min(indices, axis=0), file=sys.stderr)
print("Note: Maxima:", np.max(indices, axis=0), file=sys.stderr)


