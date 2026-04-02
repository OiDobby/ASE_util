import numpy as np
from ase import Atoms

# ====================== Input Block ======================
input_parameters = {
    "cell_parameters": [
        [24.9534988404999992, 0.0, 0.0],  # a vector
        [0.0, 23.0512008667999984, 0.0],  # b vector
        [0.0, 0.0, 10.0],  # c vector
    ],
    "density_g_cm3": 0.98,  # density (g/cm^3)
    "min_distance_OO": 2.3,  # minimum O-O distance (Å)
    "output_filename": "POSCAR_H2O",  # final output structure path
}
# ========================================================

def write_poscar(poscar_data, filename):
    """
    Write POSCAR data to a file. Coordinates are saved grouped by element.

    Parameters:
        poscar_data (dict): POSCAR data
        filename (str): Output file name
    """
    cell = poscar_data['cell']
    symbols = poscar_data['elements']
    positions = poscar_data['positions']
    counts = poscar_data['counts']

    # Sort coordinates by element
    sorted_positions = []
    for symbol in symbols:
        sorted_positions.extend(poscar_data['symbol_positions'][symbol])

    # Write POSCAR file
    with open(filename, 'w') as f:
        f.write(f"{poscar_data['comment']}\n")
        f.write(f"{poscar_data['scaling']:.16f}\n")
        for vector in cell:
            f.write("  " + "  ".join(f"{v:.16f}" for v in vector) + "\n")
        f.write("  " + "  ".join(symbols) + "\n")
        f.write("  " + "  ".join(map(str, counts)) + "\n")
        f.write(f"{poscar_data['coordinate_type']}\n")
        for pos in sorted_positions:
            f.write("  " + "  ".join(f"{v:>14.9f}" for v in pos) + "\n")

def pbc_distance(r1, r2, cell):
    """
    Calculate the minimum distance between two coordinates r1 and r2 under PBC.

    Parameters:
        r1, r2 (array): Two Cartesian coordinates
        cell (array): Cell vectors

    Returns:
        float: Minimum distance under PBC
    """
    diff = r1 - r2
    cell_inv = np.linalg.inv(cell.T)  # Inverse matrix of cell vectors
    fractional_diff = np.dot(cell_inv, diff)  # Convert to fractional coordinates
    fractional_diff -= np.round(fractional_diff)  # Apply minimum image convention
    cartesian_diff = np.dot(cell.T, fractional_diff)  # Convert back to Cartesian coordinates
    return np.linalg.norm(cartesian_diff)

def generate_random_positions(cell, num_positions, min_distance, pbc=True):
    """
    Generate random positions in the cell with a minimum-distance constraint (supports PBC).

    Parameters:
        cell (array): Cell vectors (3x3)
        num_positions (int): Number of positions to generate
        min_distance (float): Minimum-distance condition
        pbc (bool): Whether to consider PBC

    Returns:
        np.array: Generated coordinate list
    """
    positions = []
    while len(positions) < num_positions:
        new_pos = np.random.uniform(0, 1, size=(3,)) @ cell  # Random Cartesian coordinate
        if all(
            pbc_distance(new_pos, pos, cell) >= min_distance
            for pos in positions
        ):
            positions.append(new_pos)
    return np.array(positions)

def create_water_structure_with_pbc(cell_parameters, density_g_cm3, min_distance_OO=2.8, output_filename="POSCAR"):
    """
    Generate water molecules using the cell volume and density,
    place them under PBC, and save the result as a POSCAR file.

    Parameters:
        cell_parameters (list of list): 3x3 cell matrix (in Å)
        density_g_cm3 (float): Water density (g/cm^3, e.g. 1 g/cm^3)
        min_distance_OO (float): Minimum O-O distance (Å)
        output_filename (str): Output file path
    """
    # Define the initial geometry of a water molecule (O-H distance: 0.96 Å)
    water = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.32, 0.93, 0]],
    )

    # Calculate cell volume
    cell = np.array(cell_parameters)
    volume = np.abs(np.linalg.det(cell))  # Å^3

    # Convert density from g/cm^3 to g/Å^3
    density_g_ang3 = density_g_cm3 / 1e24  # 1 cm^3 = 10^24 Å^3

    # Estimate the number of molecules from molar mass and density
    molar_mass_h2o = 18.01528  # g/mol
    avogadro_number = 6.02214076e23
    mass_h2o = molar_mass_h2o / avogadro_number  # g
    num_molecules = int(volume * density_g_ang3 / mass_h2o)

    print(f"Cell volume: {volume:.2f} Å^3")
    print(f"Estimated number of water molecules: {num_molecules}")

    # Randomly place oxygen atoms
    oxygen_positions = generate_random_positions(cell, num_molecules, min_distance_OO, pbc=True)

    # Generate, rotate, and merge water molecules
    all_positions = []
    all_symbols = []
    symbol_positions = {symbol: [] for symbol in ["O", "H"]}
    for o_position in oxygen_positions:
        water_molecule = water.copy()

        # Random rotation
        axis = np.random.rand(3)  # random axis
        angle = np.random.uniform(0, 360)  # random angle
        water_molecule.rotate(angle, v=axis, center="COM")  # apply rotation

        water_molecule.translate(o_position)  # move to the oxygen position
        for symbol, pos in zip(water_molecule.get_chemical_symbols(), water_molecule.positions):
            all_positions.append(pos)
            all_symbols.append(symbol)
            symbol_positions[symbol].append(pos)

    # Build POSCAR data
    poscar_data = {
        "comment": "Generated by ASE",
        "scaling": 1.0,
        "cell": cell,
        "elements": ["O", "H"],
        "counts": [all_symbols.count("O"), all_symbols.count("H")],
        "coordinate_type": "Cartesian",
        "positions": all_positions,
        "symbol_positions": symbol_positions,
    }

    # Save the POSCAR file
    write_poscar(poscar_data, output_filename)
    print(f"Generated structure saved to {output_filename}")

# --- New helper: axis-angle rotation matrix (Rodrigues) ---
def _rand_rotation_matrix():
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)
    theta = 2*np.pi*np.random.rand()
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)
    return R

# --- Replacement 1: vectorized PBC distance check (reuse inverse matrix) ---
def generate_random_positions_fast(cell, inv_cell, num_positions, min_distance):
    """
    Randomly generate O atom coordinates in the cell.
    Uses a vectorized O-O minimum-distance check and minimizes Python list usage.
    A maximum number of attempts is set to avoid infinite loops.
    """
    pos = np.empty((num_positions, 3), dtype=np.float64)
    k = 0
    tries = 0
    max_tries = max(1000, 50 * num_positions)  # safeguard depending on density / minimum-distance combination

    while k < num_positions and tries < max_tries:
        tries += 1
        # Uniform sampling in fractional coordinates -> Cartesian
        new_cart = np.random.rand(3) @ cell

        if k == 0:
            pos[0] = new_cart
            k += 1
            continue

        diff = pos[:k] - new_cart                      # (k,3)
        diff_frac = diff @ inv_cell                    # Cartesian -> fractional
        diff_frac -= np.round(diff_frac)               # minimum image convention
        diff_mic = diff_frac @ cell                    # fractional -> Cartesian (MIC)
        if np.all(np.linalg.norm(diff_mic, axis=1) >= min_distance):
            pos[k] = new_cart
            k += 1

    if k < num_positions:
        print(f"[warning] only {k} out of the requested {num_positions} positions were placed. "
              f"Try lowering the density or min_distance.")
        pos = pos[:k].copy()
    return pos

# --- Replacement 2: main generation routine (remove Atoms copy/rotation, rotate around O) ---
def create_water_structure_with_pbc_fast(cell_parameters, density_g_cm3, min_distance_OO=2.8, output_filename="POSCAR"):
    cell = np.array(cell_parameters, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)  # computed only once
    volume = abs(np.linalg.det(cell))  # Å^3

    density_g_ang3 = density_g_cm3 / 1e24
    molar_mass_h2o = 18.01528  # g/mol
    NA = 6.02214076e23
    mass_h2o = molar_mass_h2o / NA
    num_mol = int(volume * density_g_ang3 / mass_h2o)

    print(f"Cell volume: {volume:.2f} Å^3")
    print(f"Estimated number of water molecules: {num_mol}")

    # Generate O coordinates (vectorized check)
    Opos = generate_random_positions_fast(cell, inv_cell, num_mol, min_distance_OO)

    # Water geometry (O at the origin, then rotate and move O to Opos)
    W0 = np.array([[0.0, 0.0, 0.0],
                   [0.96, 0.0, 0.0],
                   [-0.32, 0.93, 0.0]], dtype=np.float64)
    # Rotation is performed around O (i.e. W0[0] is at the origin)
    Hrel = W0[1:] - W0[0]  # (2,3)

    n = len(Opos)
    Hpos = np.empty((2*n, 3), dtype=np.float64)

    # Minimize Python object usage to save memory: pure NumPy
    for i in range(n):
        R = _rand_rotation_matrix()
        Hr = (Hrel @ R.T)  # rotated relative coordinates of H atoms
        Hpos[2*i]   = Opos[i] + Hr[0]
        Hpos[2*i+1] = Opos[i] + Hr[1]

    # Write POSCAR directly (species order: O -> H)
    with open(output_filename, "w") as f:
        f.write("Generated by ASE (fast)\n")
        f.write("1.0\n")
        for v in cell:
            f.write("  " + "  ".join(f"{x:.16f}" for x in v) + "\n")
        f.write("  O  H\n")
        f.write(f"  {n}  {2*n}\n")
        f.write("Cartesian\n")
        # O first
        np.savetxt(f, Opos, fmt="%14.9f %14.9f %14.9f")
        # H next
        np.savetxt(f, Hpos, fmt="%14.9f %14.9f %14.9f")

    print(f"Generated structure saved to {output_filename}")

# Use parameters from the Input Block
create_water_structure_with_pbc_fast(
    cell_parameters=input_parameters["cell_parameters"],
    density_g_cm3=input_parameters["density_g_cm3"],
    min_distance_OO=input_parameters["min_distance_OO"],
    output_filename=input_parameters["output_filename"]
)
