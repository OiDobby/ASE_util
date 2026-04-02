#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extxyz -> LAMMPS data (atom_style atomic)  [ASE-based + always includes Masses + automatic element handling]
- No argparse. Edit only the INPUT BLOCK below and run:
    python extxyz_to_lammps_data_ase_mass.py

Key points
- For any element (e.g. Li, Na, F, Cl, ...), the script automatically retrieves the mass
  from the ASE periodic table and always fills the Masses section in the LAMMPS data file.
- LAMMPS cannot infer atomic masses from type numbers alone, so the Masses section
  (or explicit mass commands) is required.
- The element order (ELEM) used in SevenNet pair_coeff must be identical to the LAMMPS type order.
  -> This script also writes that order into a typemap file.

Requirements:
- ase (pip install ase)
- numpy
"""

# =========================
# INPUT BLOCK (edit here)
# =========================
EXTXYZ_IN   = "541_H2O.extxyz"                         # input extxyz
FRAME_INDEX = 0                                        # frame index
OUT_DATA    = "541_H2O.data"                           # output LAMMPS data

WRAP_POS    = True                                     # wrap positions (recommended)

# Type order (type 1..N)
# - If None: automatically use sorted unique elements appearing in the extxyz (most convenient)
# - If a list is given: types are assigned in exactly this order (must match pair_coeff element order)
SPEC_ORDER  = ["Ag", "O", "H"]   # example: ["Li","H","C","N","O"] or None

# Write typemap output (recommended): type <-> element, and ELEM string for pair_coeff
WRITE_TYPEMAP = False
OUT_TYPEMAP   = "ref.txt"

# Save grain-label TSV (optional)
WRITE_GRAINMAP = False
OUT_GRAINMAP   = "ref_grainmap.tsv"
# The extxyz must contain the following arrays:
#   grain_num, grain_type, intra_grain_sequence
# =========================


from pathlib import Path
import numpy as np


def cell_to_lammps_triclinic(cell_rows_abc: np.ndarray):
    """
    Convert general cell vectors (rows: a,b,c) to LAMMPS triclinic parameters.
    Returns: xlo,xhi,ylo,yhi,zlo,zhi,xy,xz,yz
    Origin is set to (0,0,0) in the data file.
    """
    a = cell_rows_abc[0].astype(float)
    b = cell_rows_abc[1].astype(float)
    c = cell_rows_abc[2].astype(float)

    ax = np.linalg.norm(a)
    if ax <= 0:
        raise ValueError("Invalid cell: |a|=0")

    a_hat = a / ax
    bx = float(np.dot(b, a_hat))
    b_perp = b - bx * a_hat
    by = np.linalg.norm(b_perp)
    if by <= 0:
        raise ValueError("Invalid cell: b is collinear with a")

    b_hat_perp = b_perp / by
    cx = float(np.dot(c, a_hat))
    cy = float(np.dot(c, b_hat_perp))
    c_perp = c - cx * a_hat - cy * b_hat_perp
    cz = np.linalg.norm(c_perp)
    if cz <= 0:
        raise ValueError("Invalid cell: c lies in the a-b plane")

    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi, yhi, zhi = ax, by, cz
    xy, xz, yz = bx, cx, cy
    return xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz


def write_grainmap_tsv(atoms, out_tsv: str) -> None:
    keys = ["grain_num", "grain_type", "intra_grain_sequence"]
    missing = [k for k in keys if k not in atoms.arrays]
    if missing:
        raise RuntimeError(f"WRITE_GRAINMAP=True but missing arrays: {missing}")

    gnum = atoms.arrays["grain_num"].astype(int)
    gtyp = atoms.arrays["grain_type"].astype(int)
    gseq = atoms.arrays["intra_grain_sequence"].astype(int)

    lines = ["id\tgrain_num\tgrain_type\tintra_grain_sequence\n"]
    for i in range(len(atoms)):  # LAMMPS atom IDs are 1..N
        lines.append(f"{i+1}\t{gnum[i]}\t{gtyp[i]}\t{gseq[i]}\n")

    Path(out_tsv).write_text("".join(lines), encoding="utf-8")


def write_typemap(out_path: str, specorder: list[str], masses: list[float]) -> None:
    elem_str = " ".join(specorder)
    lines = []
    lines.append("# type -> element, mass (amu)\n")
    for i, (sym, m) in enumerate(zip(specorder, masses), start=1):
        lines.append(f"{i}\t{sym}\t{m:.8f}\n")
    lines.append("\n# SevenNet pair_coeff element string (must match type order)\n")
    lines.append(f'ELEM="{elem_str}"\n')
    lines.append("\n# (optional) LAMMPS mass commands (redundant if Masses section exists)\n")
    for i, m in enumerate(masses, start=1):
        lines.append(f"mass {i} {m:.8f}\n")
    Path(out_path).write_text("".join(lines), encoding="utf-8")


def main():
    try:
        import ase.io
        from ase.data import atomic_numbers, atomic_masses
    except Exception as e:
        raise SystemExit(f"ASE is required. Install with: pip install ase\nCause: {e}")

    atoms = ase.io.read(EXTXYZ_IN, index=FRAME_INDEX, format="extxyz")

    if atoms.cell is None or atoms.cell.volume <= 0:
        raise SystemExit("The extxyz header does not contain a valid Lattice/cell.")

    if WRAP_POS:
        atoms.wrap(eps=1e-12)

    present = sorted(set(atoms.get_chemical_symbols()))
    specorder = present if SPEC_ORDER is None else SPEC_ORDER

    missing = [s for s in present if s not in specorder]
    if missing:
        raise SystemExit(
            f"The following elements in the extxyz are missing from SPEC_ORDER: {missing}\n"
            f"present={present}\nSPEC_ORDER={specorder}"
        )

    # type mapping
    type_map = {sym: i + 1 for i, sym in enumerate(specorder)}
    types = np.array([type_map[s] for s in atoms.get_chemical_symbols()], dtype=int)

    # masses from the ASE periodic table
    masses = []
    for sym in specorder:
        Z = atomic_numbers[sym]
        m = float(atomic_masses[Z])
        masses.append(m)

    # positions, cell
    pos = atoms.get_positions()
    cell = atoms.cell.array  # rows are a,b,c in ASE convention

    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = cell_to_lammps_triclinic(cell)

    # Write data file
    n = len(atoms)
    ntypes = len(specorder)

    out = []
    out.append("LAMMPS data file (atom_style atomic) from extxyz (ASE-read)\n\n")
    out.append(f"{n} atoms\n")
    out.append(f"{ntypes} atom types\n\n")
    out.append(f"{xlo:.10f} {xhi:.10f} xlo xhi\n")
    out.append(f"{ylo:.10f} {yhi:.10f} ylo yhi\n")
    out.append(f"{zlo:.10f} {zhi:.10f} zlo zhi\n")
    out.append(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n\n")

    out.append("Masses\n\n")
    for i, (sym, m) in enumerate(zip(specorder, masses), start=1):
        out.append(f"{i} {m:.8f} # {sym}\n")

    out.append("\nAtoms # atomic\n\n")
    for i in range(n):
        x, y, z = pos[i]
        out.append(f"{i+1} {types[i]} {x:.10f} {y:.10f} {z:.10f}\n")

    Path(OUT_DATA).write_text("".join(out), encoding="utf-8")
    print(f"[OK] wrote: {OUT_DATA}")
    print(f"[INFO] specorder(type 1..N): {specorder}")

    if WRITE_TYPEMAP:
        write_typemap(OUT_TYPEMAP, specorder, masses)
        print(f"[OK] wrote: {OUT_TYPEMAP}")

    if WRITE_GRAINMAP:
        write_grainmap_tsv(atoms, OUT_GRAINMAP)
        print(f"[OK] wrote: {OUT_GRAINMAP}")


if __name__ == "__main__":
    main()
