import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms, FixCartesian, FixScaled

# ====================== INPUT BLOCK ======================
INPUT = {
    # slab
    "slab_poscar": "POSCAR_slab_unit",
    "output_file": "551_2Na",

    # output format: "poscar" or "extxyz"
    "output_format": "extxyz",

    # supercell: use (5, 4, 1) to reproduce the 24.9535 x 23.0512 area from the original ase_H2O.py
    "supercell": (5, 5, 1),

    # water region
    "water_density_g_cm3": 0.98,
    "min_distance_OO": 2.3,
    "water_gap_from_slab": 2.5,
    "water_thickness": 10.0,

    # cell / vacuum
    "keep_original_cell": False,          # keep the c lattice parameter identical to the unit cell
    "extra_vacuum_above_water": 18.0,

    # ion insertion option
    "add_ions": True,
    "ion_element": "Na",
    "num_ions": 2,
    "ion_min_distance": 2.0,
    "ion_dist_from_slab": 2.5,
    "ion_dist_from_water_top": 9.5,
    "ion_max_iterations": 5000,

    # sort output
    "sort_output": True,

    # move_mask for extxyz
    "write_move_mask": True,
    "move_mask_style": "int",   # "bool" or "int"

    # random seed
    "seed": None,
}
# ========================================================


def cell_xy_area(cell):
    return np.linalg.norm(np.cross(cell[0], cell[1]))


def build_slab_mask(n_atoms, slab_natoms):
    mask = np.zeros(n_atoms, dtype=bool)
    mask[:slab_natoms] = True
    return mask


def get_slab_top_z(atoms, slab_natoms):
    if slab_natoms <= 0:
        raise ValueError("slab_natoms must be positive.")
    if slab_natoms > len(atoms):
        raise ValueError("slab_natoms is larger than total number of atoms.")

    positions = atoms.get_positions()
    return positions[:slab_natoms, 2].max()


def estimate_num_waters(area_xy, thickness, density_g_cm3):
    volume_ang3 = area_xy * thickness
    density_g_ang3 = density_g_cm3 / 1.0e24

    molar_mass_h2o = 18.01528
    avogadro = 6.02214076e23
    mass_per_h2o = molar_mass_h2o / avogadro

    n_h2o = int(volume_ang3 * density_g_ang3 / mass_per_h2o)
    return max(n_h2o, 1)


def mic_distances_to_many(candidate, positions, cell):
    inv_cell = np.linalg.inv(cell)
    diff = positions - candidate[None, :]
    diff_frac = diff @ inv_cell
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    return np.linalg.norm(diff_cart, axis=1)


def random_rotation_matrix(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(0.0, 2.0 * np.pi)

    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def maybe_extend_cell_c(atoms, target_c_length):
    cell = atoms.get_cell().array.copy()
    old_c = cell[2]
    old_c_len = np.linalg.norm(old_c)

    if target_c_length <= old_c_len:
        return atoms

    c_hat = old_c / old_c_len
    cell[2] = c_hat * target_c_length
    atoms.set_cell(cell, scale_atoms=False)
    return atoms


def generate_oxygen_positions(cell, z_min, z_max, n_ox, min_distance, rng, max_trials=300000):
    if z_max <= z_min:
        raise ValueError("z_max must be larger than z_min.")

    c_len = np.linalg.norm(cell[2])
    positions = []
    trial = 0

    while len(positions) < n_ox and trial < max_trials:
        trial += 1

        u = rng.random()
        v = rng.random()
        z_cart = rng.uniform(z_min, z_max)
        w = z_cart / c_len

        frac = np.array([u, v, w], dtype=float)
        cart = frac @ cell

        if len(positions) == 0:
            positions.append(cart)
            continue

        dists = mic_distances_to_many(cart, np.array(positions), cell)
        if np.all(dists >= min_distance):
            positions.append(cart)

    if len(positions) < n_ox:
        print(f"[warning] requested {n_ox} waters, but only placed {len(positions)} waters.")
    return np.array(positions)


def build_water_atoms(cell, oxygen_positions, rng):
    water_rel = np.array([
        [0.00, 0.00, 0.00],
        [0.96, 0.00, 0.00],
        [-0.32, 0.93, 0.00],
    ], dtype=float)

    all_symbols = []
    all_positions = []

    for o_pos in oxygen_positions:
        R = random_rotation_matrix(rng)
        rotated = water_rel @ R.T
        mol = rotated + o_pos

        all_symbols.extend(["O", "H", "H"])
        all_positions.extend(mol)

    return Atoms(
        symbols=all_symbols,
        positions=np.array(all_positions),
        cell=cell,
        pbc=(True, True, True),
    )


def insert_random_ions(atoms, element, num_atoms, min_distance, z_min, z_max, rng, max_iterations):
    if z_max <= z_min:
        raise ValueError("Invalid ion insertion range: z_max <= z_min")

    cell = atoms.get_cell().array
    c_len = np.linalg.norm(cell[2])
    all_positions = atoms.get_positions().copy()

    inserted = []
    it = 0

    while len(inserted) < num_atoms and it < max_iterations:
        it += 1

        u = rng.random()
        v = rng.random()
        z_cart = rng.uniform(z_min, z_max)
        w = z_cart / c_len

        frac = np.array([u, v, w], dtype=float)
        cand = frac @ cell

        dists = mic_distances_to_many(cand, all_positions, cell)
        if np.min(dists) >= min_distance:
            inserted.append(cand)
            all_positions = np.vstack([all_positions, cand])

    if len(inserted) < num_atoms:
        print(f"[warning] only {len(inserted)} / {num_atoms} ions were inserted.")

    if inserted:
        ion_atoms = Atoms(
            symbols=[element] * len(inserted),
            positions=np.array(inserted),
            cell=cell,
            pbc=(True, True, True),
        )
        atoms += ion_atoms

    return atoms


def sort_atoms_for_output(atoms, slab_natoms):
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()

    slab_mask = build_slab_mask(len(atoms), slab_natoms)
    nonslab_mask = ~slab_mask

    order = []

    # 1. slab: grouped by species + sorted by z within each species
    slab_species_order = []
    for s in symbols[slab_mask]:
        if s not in slab_species_order:
            slab_species_order.append(s)

    for s in slab_species_order:
        idx = np.where((symbols == s) & slab_mask)[0]
        idx = idx[np.argsort(positions[idx, 2], kind="stable")]
        order.extend(idx.tolist())

    # 2. non-slab: O -> H -> remaining species (e.g. F)
    preferred_order = ["O", "H"]
    nonslab_species_present = []
    for s in symbols[nonslab_mask]:
        if s not in nonslab_species_present:
            nonslab_species_present.append(s)

    for s in preferred_order:
        if s in nonslab_species_present:
            idx = np.where((symbols == s) & nonslab_mask)[0]
            order.extend(idx.tolist())

    for s in nonslab_species_present:
        if s not in preferred_order:
            idx = np.where((symbols == s) & nonslab_mask)[0]
            order.extend(idx.tolist())

    order = np.array(order, dtype=int)
    new_atoms = atoms[order]
    return new_atoms, order


def normalize_output_format(fmt):
    fmt = fmt.lower()
    if fmt in ["poscar", "vasp"]:
        return "vasp"
    if fmt in ["extxyz", "xyz"]:
        return "extxyz"
    raise ValueError("output_format must be one of: 'poscar', 'vasp', 'extxyz'")


def build_output_filename(base, fmt):
    if fmt == "vasp":
        return base if base.upper().startswith("POSCAR") or "." in base else base + ".vasp"
    if fmt == "extxyz":
        return base if base.endswith(".extxyz") else base + ".extxyz"
    return base


def sanitize_for_extxyz(atoms, verbose=True):
    n = len(atoms)

    # 1) Convert arrays without dtype (e.g. lists) into np.ndarray
    for k, v in list(atoms.arrays.items()):
        if not hasattr(v, "dtype"):
            try:
                arr = np.asarray(v)
                atoms.set_array(k, arr)
                if verbose:
                    print(f"[sanitize] converted array '{k}' from {type(v)} to np.ndarray")
            except Exception as e:
                if verbose:
                    print(f"[sanitize] removed array '{k}' (conversion failed): {e}")
                del atoms.arrays[k]

    # 2) Remove per-atom arrays whose first dimension does not match the number of atoms
    for k, v in list(atoms.arrays.items()):
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] != n:
            if verbose:
                print(f"[sanitize] removed array '{k}' (shape={v.shape}, natoms={n})")
            del atoms.arrays[k]

    return atoms


def build_move_mask_from_constraints(atoms, style="int"):
    """
    Build move_mask from atoms.constraints.
    movable = 1(True), fixed = 0(False)

    Returned shape: (N, 3)
    """
    n = len(atoms)
    move_mask = np.ones((n, 3), dtype=np.int8)  # default: all atoms movable

    cell = atoms.get_cell()

    for cons in atoms.constraints:
        # 1) Fully fixed atoms
        if isinstance(cons, FixAtoms):
            idx = np.asarray(cons.index, dtype=int)
            move_mask[idx, :] = 0

        # 2) Axis-wise constraints in Cartesian coordinates
        elif isinstance(cons, FixCartesian):
            idx = np.asarray(cons.index, dtype=int)

            # In ASE, cons.mask usually means True = fixed axis
            mask = np.asarray(cons.mask, dtype=bool)

            if mask.ndim == 1:
                move_mask[idx[:, None], np.where(mask)[0]] = 0
            else:
                for i_atom, atom_idx in enumerate(idx):
                    move_mask[atom_idx, mask[i_atom]] = 0

        # 3) Axis-wise constraints in scaled coordinates
        elif isinstance(cons, FixScaled):
            idx = np.atleast_1d(np.asarray(cons.index, dtype=int))
            mask = np.asarray(cons.mask, dtype=bool)

            if mask.ndim == 1:
                move_mask[idx[:, None], np.where(mask)[0]] = 0
            else:
                for i_atom, atom_idx in enumerate(idx):
                    move_mask[atom_idx, mask[i_atom]] = 0

    style = style.lower()
    if style == "int":
        return move_mask.astype(np.int8)
    elif style == "bool":
        return move_mask.astype(bool)
    else:
        raise ValueError("move_mask_style must be 'bool' or 'int'")


def write_structure(atoms, output_file, output_format, move_mask=None):
    fmt = normalize_output_format(output_format)
    filename = build_output_filename(output_file, fmt)

    if fmt == "vasp":
        write(filename, atoms, format="vasp", direct=False, sort=False)

    elif fmt == "extxyz":
        atoms_to_write = atoms.copy()

        # Remove existing move_mask-related arrays
        for k in list(atoms_to_write.arrays.keys()):
            if k == "move_mask" or k.startswith("move_mask"):
                del atoms_to_write.arrays[k]

        if move_mask is not None:
            # If we explicitly write move_mask ourselves,
            # prevent ASE from automatically generating another move_mask from constraints
            atoms_to_write.set_constraint([])

            mm = np.asarray(move_mask)
            if mm.dtype == bool:
                atoms_to_write.set_array("move_mask", mm.astype(bool))
            else:
                atoms_to_write.set_array("move_mask", mm.astype(np.int8))

        atoms_to_write = sanitize_for_extxyz(atoms_to_write, verbose=True)
        write(filename, atoms_to_write, format="extxyz")

    return filename


def main():
    if INPUT["seed"] is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(INPUT["seed"])

    # 1. read slab + build supercell
    slab = read(INPUT["slab_poscar"])
    slab = slab.repeat(INPUT["supercell"])
    slab.set_pbc((True, True, True))
    slab_natoms = len(slab)

    cell = slab.get_cell().array
    area_xy = cell_xy_area(cell)
    slab_top_z = get_slab_top_z(slab, slab_natoms=slab_natoms)

    # 2. define water region
    water_z_min = slab_top_z + INPUT["water_gap_from_slab"]
    water_z_max = water_z_min + INPUT["water_thickness"]

    if not INPUT["keep_original_cell"]:
        target_c = water_z_max + INPUT["extra_vacuum_above_water"]
        slab = maybe_extend_cell_c(slab, target_c)
        cell = slab.get_cell().array

    c_len = np.linalg.norm(cell[2])
    if water_z_max >= c_len:
        raise ValueError(
            f"Water region exceeds cell height: water_z_max={water_z_max:.3f}, c={c_len:.3f}"
        )

    # 3. estimate the number of H2O molecules
    n_h2o = estimate_num_waters(
        area_xy=area_xy,
        thickness=INPUT["water_thickness"],
        density_g_cm3=INPUT["water_density_g_cm3"],
    )

    print("========== SYSTEM INFO ==========")
    print(f"supercell           : {INPUT['supercell']}")
    print(f"cell a,b,c lengths  : {np.linalg.norm(cell[0]):.6f}, {np.linalg.norm(cell[1]):.6f}, {np.linalg.norm(cell[2]):.6f} Å")
    print(f"xy area             : {area_xy:.6f} Å^2")
    print(f"slab top z          : {slab_top_z:.6f} Å")
    print(f"water region        : {water_z_min:.6f} ~ {water_z_max:.6f} Å")
    print(f"estimated H2O count : {n_h2o}")
    print("=================================")

    # 4. generate H2O
    oxygen_positions = generate_oxygen_positions(
        cell=cell,
        z_min=water_z_min,
        z_max=water_z_max,
        n_ox=n_h2o,
        min_distance=INPUT["min_distance_OO"],
        rng=rng,
    )

    water = build_water_atoms(cell, oxygen_positions, rng)

    # 5. merge structures
    full = slab + water
    full.set_cell(cell)
    full.set_pbc((True, True, True))

    # 6. optional ion insertion
    if INPUT["add_ions"]:
        positions = full.get_positions()
        symbols = np.array(full.get_chemical_symbols())
        water_mask = np.isin(symbols, ["H", "O"])

        if not np.any(water_mask):
            raise ValueError("No water atoms found for ion insertion window.")

        h2o_top_z = positions[water_mask, 2].max()
        ion_z_min = slab_top_z + INPUT["ion_dist_from_slab"]
        ion_z_max = h2o_top_z - INPUT["ion_dist_from_water_top"]

        print(f"ion region          : {ion_z_min:.6f} ~ {ion_z_max:.6f} Å")

        full = insert_random_ions(
            atoms=full,
            element=INPUT["ion_element"],
            num_atoms=INPUT["num_ions"],
            min_distance=INPUT["ion_min_distance"],
            z_min=ion_z_min,
            z_max=ion_z_max,
            rng=rng,
            max_iterations=INPUT["ion_max_iterations"],
        )

    # 7. build move_mask (reflecting constraints from the input POSCAR)
    move_mask = None
    if INPUT["output_format"].lower() in ["extxyz", "xyz"] and INPUT["write_move_mask"]:
        move_mask = build_move_mask_from_constraints(
            full,
            style=INPUT["move_mask_style"],
        )

    # 8. sort if needed
    if INPUT["sort_output"]:
        full, order = sort_atoms_for_output(full, slab_natoms=slab_natoms)
        if move_mask is not None:
            move_mask = move_mask[order]

    # 9. write structure
    saved_file = write_structure(
        atoms=full,
        output_file=INPUT["output_file"],
        output_format=INPUT["output_format"],
        move_mask=move_mask,
    )
    print(f"saved: {saved_file}")


if __name__ == "__main__":
    main()
