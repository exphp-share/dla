import xyz
import sys
import numpy as np

paths = sys.argv[1:]
arr = []
for i,p in enumerate(paths):
	print("reading file ",i,file=sys.stderr)
	with open(p) as f:
		arr.append(xyz.load_anim(f))


def extend_xyz_data(x, name, new_count):
	missing = new_count - len(x.names)
	x.names.extend([name] * missing)

	pos = x.positions.tolist()
	pos.extend([pos[0]] * missing)
	x.positions = np.array(pos)

max_particle_count = max(len(anim[0].names) for anim in arr)

i=0
arr = arr[::-1]
while arr:
	anim = arr.pop()
	print("processing file ",i,file=sys.stderr)
	for frame in anim:
		extend_xyz_data(frame, 'C', max_particle_count)
	print("writing file ",i,file=sys.stderr)
	xyz.dump_anim(sys.stdout, anim)
	i += 1


