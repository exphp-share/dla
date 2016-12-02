import numpy as np
import json
import sys

PARAM_A = 1.
PARAM_C = 1.

indices = np.array(json.load(sys.stdin))
# row-based
cellmatrix = np.diag([PARAM_A, PARAM_A, PARAM_C]) * np.array([
	[1, 0, 0],
	[np.cos(2/3*np.pi), np.sin(2/3*np.pi), 0],
	[0, 0, 1],
])

cartesian = indices.dot(cellmatrix)
print(len(cartesian))
print("blah blah blah") # xyz comment
for (x,y,z) in cartesian:
	print("C {} {} {}".format(x,y,z))


