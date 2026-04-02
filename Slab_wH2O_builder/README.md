# ASE-based Slab / Water / Ion Structure Builders

## Overview
This directory contains three Python scripts for building slab–water–ion structures using ASE.

The scripts are organized by purpose:

- `ase_H2O.py`  
  Generates packed H2O molecules inside a simulation cell.

- `atom_isrt.py`  
  Inserts ions/atoms into an existing slab + water structure.

- `ase_H2O_merg.py`  
  Integrated workflow that reads a slab, builds a water region above it, optionally inserts ions, and writes the final structure.

---

## Repository Structure

    .
    ├── ase_H2O.py
    ├── atom_isrt.py
    └── ase_H2O_merg.py

---

## Files and Their Roles

### 1. `ase_H2O.py`
This script generates water molecules in a user-defined cell based on:

- cell parameters
- target water density
- minimum O–O distance

Main features:

- estimates the number of H2O molecules from cell volume and density
- places oxygen atoms randomly with PBC-aware minimum-distance checks
- rotates each water molecule randomly
- writes the generated water-only structure to a POSCAR-style file

This script is useful when you want to create only the water box part.

---

### 2. `atom_isrt.py`
This script inserts atoms or ions into an existing structure.

Main features:

- reads an existing structure file
- separates slab atoms and H2O atoms
- finds the insertion region between slab and water
- randomly inserts a chosen element
- applies a minimum-distance condition with PBC
- writes the modified structure to a new POSCAR file

This script is useful when the slab + water structure already exists and you only want to insert ions afterward.

---

### 3. `ase_H2O_merg.py`
This is the integrated structure-generation script.

Main features:

- reads a slab structure
- builds a supercell
- estimates the number of H2O molecules for a given water region
- places water molecules above the slab
- optionally inserts ions into the slab/water interface region
- supports both POSCAR and extxyz output
- can preserve and export `move_mask` information for extxyz
- can sort atoms before output

This script is useful when you want a full slab + water + ion structure in one step.

---

## Dependencies
These scripts require Python with the following packages:

- `numpy`
- `ase`

Example installation:

    pip install numpy ase

---

## Required Input Files

### `ase_H2O.py`
No external structure file is required.  
The simulation cell is defined directly inside the script.

### `atom_isrt.py`
Requires an input structure file such as:

    POSCAR

The structure should already contain:

- slab atoms
- H2O molecules

### `ase_H2O_merg.py`
Requires a slab structure file, for example:

    POSCAR_slab_unit

This file is read as the starting slab structure.

---

## Usage

### 1. Water packing only
Edit the input block in `ase_H2O.py`, then run:

    python ase_H2O.py

### 2. Ion insertion only
Edit the input block in `atom_isrt.py`, then run:

    python atom_isrt.py

### 3. Integrated slab + water + ion generation
Edit the `INPUT` block in `ase_H2O_merg.py`, then run:

    python ase_H2O_merg.py

---

## Input Parameters

### `ase_H2O.py`
Main input parameters:

- `cell_parameters`  
  simulation cell vectors

- `density_g_cm3`  
  target water density

- `min_distance_OO`  
  minimum O–O distance

- `output_filename`  
  output file name

This script writes a water-only structure file.

---

### `atom_isrt.py`
Main input parameters:

- `input_file`  
  input structure file

- `output_file`  
  output structure file

- `insert_element`  
  atom/ion species to insert

- `min_distance`  
  minimum distance from all existing atoms

- `num_atoms`  
  number of inserted atoms

- `max_iterations`  
  maximum random insertion attempts

- `dist_from_slab`  
  minimum z-distance above the slab

- `dist_from_max`  
  minimum z-distance below the top of the water region

This script is designed for inserting atoms into the slab–water gap region.

---

### `ase_H2O_merg.py`
Main input parameters are grouped in the `INPUT` dictionary.

#### Slab settings
- `slab_poscar`
- `output_file`
- `output_format`
- `supercell`

#### Water settings
- `water_density_g_cm3`
- `min_distance_OO`
- `water_gap_from_slab`
- `water_thickness`

#### Cell / vacuum settings
- `keep_original_cell`
- `extra_vacuum_above_water`

#### Ion insertion settings
- `add_ions`
- `ion_element`
- `num_ions`
- `ion_min_distance`
- `ion_dist_from_slab`
- `ion_dist_from_water_top`
- `ion_max_iterations`

#### Output settings
- `sort_output`
- `write_move_mask`
- `move_mask_style`

#### Random control
- `seed`

---

## Output Files

### `ase_H2O.py`
Writes a water-only POSCAR-style structure file, for example:

    POSCAR_H2O

### `atom_isrt.py`
Writes a modified POSCAR-style structure file containing the inserted atoms, for example:

    POSCAR_00

### `ase_H2O_merg.py`
Writes the final merged structure in either:

- POSCAR / VASP format
- extxyz format

depending on `output_format`.

Examples:

    551_2Na.extxyz
    551_2Na.vasp

---

## Typical Workflow

### Option 1: Step-by-step workflow
1. Use `ase_H2O.py` to create packed water
2. Merge or prepare slab + water structure
3. Use `atom_isrt.py` to insert ions

### Option 2: Integrated workflow
1. Prepare slab input file
2. Use `ase_H2O_merg.py`
3. Generate slab + water + ion structure in one run

---

## Notes

### About `ase_H2O.py`
- Water molecule count is estimated from cell volume and density.
- Oxygen positions are generated with a PBC-aware minimum-distance check.
- Each water molecule is randomly rotated before placement.
- The script contains both an original implementation and a faster implementation, and the fast routine is used at execution time.

### About `atom_isrt.py`
- The script distinguishes slab atoms from water atoms using element symbols:
  - slab: atoms that are not `H` or `O`
  - water: atoms that are `H` or `O`
- The insertion region is chosen between the slab top and water top.
- Random positions are tested until the minimum-distance condition is satisfied or the iteration limit is reached.

### About `ase_H2O_merg.py`
- The slab is read first and expanded to a supercell.
- The water region is defined above the slab top.
- The cell can be extended along `c` automatically.
- Ion insertion is optional.
- If extxyz output is used, `move_mask` can also be written.
- Constraints from the slab can be converted into `move_mask`.

---

## Example Workflows

### Water-only generation
1. Set `cell_parameters`
2. Set `density_g_cm3`
3. Set `min_distance_OO`
4. Run `ase_H2O.py`
5. Check the generated water structure

### Ion insertion into existing slab + water
1. Prepare a POSCAR containing slab + water
2. Set `insert_element`
3. Set insertion distances and minimum distance
4. Run `atom_isrt.py`
5. Check the output structure

### Full integrated generation
1. Prepare slab POSCAR
2. Set supercell size
3. Set water density and thickness
4. Set ion options if needed
5. Choose output format
6. Run `ase_H2O_merg.py`
7. Check the final merged structure

---

## Recommended Use Cases

### Use `ase_H2O.py` when:
- you only need a water box
- you want to test water packing density
- you want a simple water-only POSCAR

### Use `atom_isrt.py` when:
- slab + water is already prepared
- you only want to insert ions afterward
- you want a small and simple insertion script

### Use `ase_H2O_merg.py` when:
- you want a full slab + water + ion workflow
- you need supercell handling
- you want extxyz output with move masks
- you want one integrated script instead of multiple steps

---

## Source
This README was written based on the behavior of:

- `ase_H2O.py`
- `ase_H2O_merg.py`
- `atom_isrt.py`
