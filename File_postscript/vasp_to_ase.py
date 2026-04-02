import ase
from ase.io import read, write

vasp_read = read('OUTCAR', index=':', format='vasp-out')
# vasp_read = read('POSCAR', index=':', format='vasp')
# "index" controls how frequently structures are read.
# Examples: 1:1000:10, ::10, :
# If you want to convert POSCAR to extxyz format (instead of OUTCAR),
# you must set format="vasp".

for structure in vasp_read:
    write('output.extxyz', structure, format="extxyz", append=True)
    # write('test.extxyz', structure, format="extxyz", append=True)
    # append=True prevents overwriting and keeps appending to the file,
    # similar to Python file mode "a".
