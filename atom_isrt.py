import random
from ase.io import read, write
from ase import Atoms
import numpy as np
from ase.geometry import wrap_positions

# ==== INPUT BLOCK ====
input_file = "POSCAR"  # Input file name
output_file = "POSCAR_00"  # Output file name
insert_element = "F"  # Element to insert
min_distance = 1.8  # Minimum distance between the inserted atom and existing atoms (Å)
num_atoms = 1  # Number of atoms to insert
max_iterations = 1000  # Maximum number of iterations for candidate search
dist_from_slab = 2.0  # Minimum distance from slab
dist_from_max = 3.0  # Minimum distance from electrolyte
# ======================

# Read POSCAR file
structure = read(input_file)
cell = structure.get_cell()  # Periodic boundary condition (PBC) cell information

# Get coordinates of all atoms
all_positions = np.array([atom.position for atom in structure])  # Include all atom positions

# Separate slab and H₂O molecules
slab_positions = np.array([atom.position for atom in structure if atom.symbol not in ["H", "O"]])
h2o_positions = np.array([atom.position for atom in structure if atom.symbol in ["H", "O"]])

if len(slab_positions) == 0:
    raise ValueError("Slab structure not found.")
if len(h2o_positions) == 0:
    raise ValueError("H₂O molecules not found.")

# Calculate the highest z-coordinate for the slab and H₂O molecules
slab_max_z = slab_positions[:, 2].max()
h2o_max_z = h2o_positions[:, 2].max()

# Set the z-coordinate range for the inserted atoms
z_min = slab_max_z + dist_from_slab  # At least 2 Å above the highest point of the slab
z_max = h2o_max_z - dist_from_max  # Below the highest point of H₂O molecules

# Set the x, y coordinate range
x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

# Insert the atoms
added_atoms = []
iterations = 0

while len(added_atoms) < num_atoms and iterations < max_iterations:
    iterations += 1

    # Generate random coordinates
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = random.uniform(z_min, z_max)
    candidate = np.array([x, y, z])

    # Wrap candidate coordinates within the PBC cell
    candidate_wrapped = wrap_positions(candidate[np.newaxis, :], cell=cell)[0]

    # Calculate distances to all existing atoms, considering PBC
    distances = []
    for pos in all_positions:
        diff = candidate_wrapped - pos
        diff -= np.round(diff @ np.linalg.inv(cell), 0) @ cell  # Apply PBC
        distance = np.linalg.norm(diff)
        distances.append(distance)

    # Check the minimum distance
    min_dist = min(distances)
    print(f"Iteration {iterations}: Candidate {candidate_wrapped}, Min distance: {min_dist}")

    # Check if the minimum distance satisfies the condition
    if min_dist >= min_distance:
        # Add atom if conditions are met
        added_atoms.append(candidate_wrapped.tolist())
        all_positions = np.append(all_positions, [candidate_wrapped], axis=0)  # Include the new atom position
        print(f"{insert_element} added at {candidate_wrapped}")
    else:
        print(f"Candidate {candidate_wrapped} is too close to existing atoms.")

# Output results
print(f"Max iterations: {max_iterations}")
print(f"{insert_element} atoms successfully added: {len(added_atoms)}")
if len(added_atoms) < num_atoms:
    print(f"Could not place {num_atoms - len(added_atoms)} {insert_element} atoms due to lack of space.")

# Add inserted atoms to the structure
for atom_position in added_atoms:
    added_atom = Atoms(insert_element, positions=[atom_position])
    structure += added_atom

# Save the modified structure
write(output_file, structure, format='vasp')
print(f"Modified structure saved to {output_file}")

