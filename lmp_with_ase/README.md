# LAMMPS / extxyz Conversion Utilities

## Overview
This directory contains Python scripts for converting between LAMMPS trajectory/data formats and `extxyz` structures using ASE.

The scripts are intended for workflows such as:

- converting LAMMPS dump trajectories (`lammpstrj`) to `extxyz`
- converting `extxyz` structures to LAMMPS `data` files
- preserving atom type / element mapping
- optionally exporting force arrays
- optionally handling grain-label metadata

Included scripts:

- `lammps_dump_to_extxyz_ase_v4.py`
- `extxyz_to_lammps_data_ase_mass_v2.py`

---

## Repository Structure

    .
    ├── lammps_dump_to_extxyz_ase_v4.py
    └── extxyz_to_lammps_data_ase_mass_v2.py

---

## Files and Their Roles

### 1. `lammps_dump_to_extxyz_ase_v4.py`
This script converts a LAMMPS dump trajectory (`.lammpstrj`) into an `extxyz` trajectory using ASE.

Main features:

- reads LAMMPS dump frames
- supports both orthorhombic and triclinic boxes
- reads atom IDs and atom types
- converts LAMMPS atom types to chemical species
- writes one or more frames into a single `extxyz` file
- optionally stores force arrays if `fx fy fz` are present
- optionally merges grain metadata from a TSV file

This script is useful when you want to analyze or visualize LAMMPS trajectories in ASE-compatible formats.

---

### 2. `extxyz_to_lammps_data_ase_mass_v2.py`
This script converts an `extxyz` structure into a LAMMPS `data` file with `atom_style atomic`.

Main features:

- reads an `extxyz` structure using ASE
- converts ASE cell vectors into LAMMPS triclinic box format
- assigns LAMMPS atom types from element symbols
- automatically writes the `Masses` section
- optionally writes a type-mapping helper file
- optionally writes grain-label metadata TSV

This script is useful when you want to start a LAMMPS simulation from an `extxyz` structure.

---

## Dependencies
These scripts require Python with the following packages:

- `numpy`
- `ase`

Example installation:

    pip install numpy ase

---

## Required Input Files

### `lammps_dump_to_extxyz_ase_v4.py`
Requires:

- a LAMMPS dump trajectory file, for example:
  
      dump_300K_0GPa.lammpstrj

Optional additional files:

- type map text file
- LAMMPS data file containing `Masses` comments
- grain map TSV file

### `extxyz_to_lammps_data_ase_mass_v2.py`
Requires:

- an `extxyz` file, for example:
  
      541_H2O.extxyz

Optional additional outputs:

- type map text file
- grain map TSV file

---

## Usage

Both scripts use an **INPUT BLOCK** inside the Python file.  
There is no `argparse` interface.

Edit the input block, then run:

### LAMMPS dump to extxyz
    python lammps_dump_to_extxyz_ase_v4.py

### extxyz to LAMMPS data
    python extxyz_to_lammps_data_ase_mass_v2.py

---

## Input Parameters

### `lammps_dump_to_extxyz_ase_v4.py`

Main parameters:

- `DUMP_IN`  
  input LAMMPS dump file

- `OUT_EXTXYZ`  
  output extxyz trajectory file

- `TYPEMAP_TXT`  
  optional type-to-species mapping text file

- `DATA_FILE`  
  optional LAMMPS data file used to read species names from the `Masses` section

- `TYPE_TO_SPECIES`  
  manual Python dictionary for direct type-to-element mapping

- `PREFERRED_POS_COLS`  
  preferred position columns, usually `xu yu zu`

- `WRITE_FORCES`  
  whether to export force arrays if `fx fy fz` exist in the dump

- `USE_GRAIN`  
  whether to merge grain metadata

- `GRAINMAP_TSV`  
  grain metadata TSV file

#### Type-to-species mapping priority
The script determines atom species in the following order:

1. `TYPE_TO_SPECIES`
2. `TYPEMAP_TXT`
3. `DATA_FILE` `Masses` section with `# element` comments

---

### `extxyz_to_lammps_data_ase_mass_v2.py`

Main parameters:

- `EXTXYZ_IN`  
  input extxyz file

- `FRAME_INDEX`  
  frame index to extract

- `OUT_DATA`  
  output LAMMPS data file

- `WRAP_POS`  
  whether to wrap positions into the periodic cell

- `SPEC_ORDER`  
  element order used to assign LAMMPS atom types

- `WRITE_TYPEMAP`  
  whether to write a helper type map text file

- `OUT_TYPEMAP`  
  output filename for type map text

- `WRITE_GRAINMAP`  
  whether to export grain-label metadata TSV

- `OUT_GRAINMAP`  
  output filename for grain metadata TSV

#### Type assignment
- If `SPEC_ORDER = None`, element types are assigned automatically from sorted unique symbols in the extxyz file.
- If `SPEC_ORDER` is explicitly given, that order is used as the LAMMPS type order.
- This is important for force-field settings such as `pair_coeff`, because element order must match LAMMPS type numbering.

---

## Output Files

### `lammps_dump_to_extxyz_ase_v4.py`
Main output:

- `*.extxyz`

Optional stored arrays in extxyz:

- `forces`
- `grain_num`
- `grain_type`
- `intra_grain_sequence`

The script writes all frames into a single extxyz trajectory file.

---

### `extxyz_to_lammps_data_ase_mass_v2.py`
Main output:

- `*.data`

Optional additional files:

- type map text file
- grain metadata TSV file

The LAMMPS data file contains:

- number of atoms
- number of atom types
- triclinic cell parameters
- `Masses` section
- `Atoms # atomic` section

---

## Typical Workflow

### 1. LAMMPS dump -> extxyz
1. Prepare a LAMMPS dump file
2. Define type-to-species mapping
3. Run `lammps_dump_to_extxyz_ase_v4.py`
4. Open the generated extxyz file in ASE, OVITO, or other analysis tools

### 2. extxyz -> LAMMPS data
1. Prepare an extxyz structure
2. Set the desired `SPEC_ORDER`
3. Run `extxyz_to_lammps_data_ase_mass_v2.py`
4. Use the generated `.data` file in LAMMPS

### 3. Round-trip workflow
1. Convert LAMMPS dump to extxyz
2. Analyze or modify the structure
3. Convert extxyz back to a LAMMPS data file

---

## Notes

### About `lammps_dump_to_extxyz_ase_v4.py`
- Supports both orthorhombic and triclinic boxes.
- Supports atom columns such as:
  
      id type xu yu zu
      id type x y z
      id type xu yu zu fx fy fz

- If `WRITE_FORCES=True` but force columns are missing, the script prints a warning and omits forces.
- Frames are sorted by atom ID before writing.
- Grain metadata can be attached if a TSV file is provided.

### About `extxyz_to_lammps_data_ase_mass_v2.py`
- Uses ASE periodic-table data to determine atomic masses automatically.
- Always writes a `Masses` section.
- Converts general ASE cell vectors into LAMMPS triclinic box parameters.
- Can write a helper type map file for use with pair styles such as SevenNet.
- Can also export grain labels if they exist in the extxyz arrays.

---

## Example Use Cases

### Convert LAMMPS trajectory for ASE/OVITO analysis
Use `lammps_dump_to_extxyz_ase_v4.py` when you want to:

- inspect MD trajectories
- visualize trajectories in OVITO or ASE
- preserve forces in extxyz format
- preserve grain labels

### Build a LAMMPS input structure from extxyz
Use `extxyz_to_lammps_data_ase_mass_v2.py` when you want to:

- prepare a LAMMPS `data` file from a structure
- automatically generate the `Masses` section
- control atom type order
- generate helper typemap files

---

## Example Commands

### Dump to extxyz
    python lammps_dump_to_extxyz_ase_v4.py

### extxyz to data
    python extxyz_to_lammps_data_ase_mass_v2.py

---

## Source
This README was written based on the behavior of:

- `lammps_dump_to_extxyz_ase_v4.py`
- `extxyz_to_lammps_data_ase_mass_v2.py`
