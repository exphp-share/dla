#!/usr/bin/env python3

import numpy as np
import json
import sys

if sys.stdout.isatty():
	print("Reading from STDIN...", file=sys.stderr)

PARAM_A = 1.
PARAM_C = 2.
# i and j are axial vectors with 60 degrees between them;
# the coordinates are described here in terms of vectors which
# are 120 degrees apart; but they are centered about the x axis,
# hence the components contain cos(60) and sin(60)
#
#       b
#      /     \vec i = \vec a + \vec c
#  c--o      \vec j = \vec b + \vec c
#      \
#       a
VX = np.cos(2/6*np.pi) + 1
VY = np.sin(2/6*np.pi)

indices = np.array(json.load(sys.stdin))
# row-based
cellmatrix = np.diag([PARAM_A, PARAM_A, PARAM_C]).dot(np.array([
	[VX, -VY, 0],
	[VX,  VY, 0],
	[0, 0, 1],
]))

cartesian = indices.dot(cellmatrix)
print(len(cartesian))
print("blah blah blah") # xyz comment
for (x,y,z) in cartesian:
	print("C {} {} {}".format(x,y,z))


