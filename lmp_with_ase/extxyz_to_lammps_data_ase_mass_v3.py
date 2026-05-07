#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extxyz -> LAMMPS data (atom_style atomic)  [ASE 기반 + Masses 항상 포함 + 원소 자동]

추가 기능
- move_mask 읽기(선택)
- move_mask = [0,0,0] 인 atom id를 추출
- LAMMPS input에 바로 넣을 수 있는 freeze 명령을 print / 파일 저장

필요:
- ase
- numpy
"""

# =========================
# INPUT BLOCK (edit here)
# =========================
EXTXYZ_IN   = "551_3F.extxyz"
FRAME_INDEX = 0
OUT_DATA    = "551_3F.data"

WRAP_POS    = True

# type 순서(type 1..N)
SPEC_ORDER  = ["Ag", "O", "H", "F"]

# typemap 출력
WRITE_TYPEMAP = True
OUT_TYPEMAP   = "ref_3F.txt"

# grain 라벨 TSV 저장
WRITE_GRAINMAP = False
OUT_GRAINMAP   = "ref_grainmap.tsv"

# move_mask 처리
USE_MOVE_MASK = True
WRITE_FREEZE_INPUT = True
OUT_FREEZE_INPUT   = "freeze_lammps_3F.in"

# move_mask 판단 방식
# [0,0,0] 이면 완전 고정 atom으로 간주
# 예: extxyz에서 0 0 0 / 1 1 1 형태
# =========================


from pathlib import Path
import numpy as np


def cell_to_lammps_triclinic(cell_rows_abc: np.ndarray):
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
        raise ValueError("Invalid cell: c lies in a-b plane")

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
    for i in range(len(atoms)):
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


def get_frozen_atom_ids(atoms):
    """
    1) atoms.arrays['move_mask']가 있으면 그걸 우선 사용
    2) 없으면 atoms.constraints에서 fully frozen atom id를 복원
    반환: LAMMPS용 1-based atom ids
    """
    import numpy as np
    from ase.constraints import FixAtoms, FixCartesian

    # case 1: move_mask array exists
    if "move_mask" in atoms.arrays:
        mm = np.asarray(atoms.arrays["move_mask"])
        if mm.ndim != 2 or mm.shape[1] != 3:
            raise RuntimeError(f"move_mask shape expected (N,3), got {mm.shape}")
        frozen = np.all(mm == 0, axis=1)
        return (np.where(frozen)[0] + 1).tolist()

    # case 2: ASE converted move_mask -> constraints
    frozen = set()

    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            frozen.update((np.array(c.get_indices(), dtype=int) + 1).tolist())

        elif isinstance(c, FixCartesian):
            idx = np.array(c.get_indices(), dtype=int)

            mask = getattr(c, "mask", None)
            if mask is None:
                continue

            mask = np.asarray(mask)

            # single atom or shared mask: [True, True, True]
            if mask.ndim == 1 and mask.shape[0] == 3:
                if np.all(mask):
                    frozen.update((idx + 1).tolist())

            # per-atom mask: (N,3)
            elif mask.ndim == 2 and mask.shape[1] == 3:
                fully = np.all(mask, axis=1)
                frozen.update((idx[fully] + 1).tolist())

    return sorted(frozen)


def compress_ids_to_ranges(ids):
    """
    [1,2,3,7,8,10] -> ['1:3', '7:8', '10']
    """
    if not ids:
        return []

    ids = sorted(ids)
    ranges = []
    start = prev = ids[0]

    for x in ids[1:]:
        if x == prev + 1:
            prev = x
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}:{prev}")
            start = prev = x

    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}:{prev}")

    return ranges


def build_freeze_lammps_input(frozen_ids):
    """
    frozen atom id 목록으로 LAMMPS input snippet 생성
    """
    if not frozen_ids:
        return "# No fully frozen atoms detected from move_mask.\n"

    id_ranges = compress_ids_to_ranges(frozen_ids)
    id_expr = " ".join(id_ranges)

    lines = []
    lines.append("# LAMMPS commands generated from extxyz move_mask\n")
    lines.append(f"group frozen id {id_expr}\n")
    lines.append("group mobile subtract all frozen\n")
    lines.append("velocity frozen set 0.0 0.0 0.0\n")
    lines.append("fix freeze frozen setforce 0.0 0.0 0.0\n")
    lines.append("\n")
    lines.append("# Example:\n")
    lines.append("# fix int mobile nve\n")
    lines.append("# or\n")
    lines.append("# fix int mobile nvt temp 300.0 300.0 100.0\n")
    return "".join(lines)


def main():
    try:
        import ase.io
        from ase.data import atomic_numbers, atomic_masses
    except Exception as e:
        raise SystemExit(f"ASE가 필요합니다. 설치: pip install ase\n원인: {e}")

    atoms = ase.io.read(EXTXYZ_IN, index=FRAME_INDEX, format="extxyz")

    '''
    print("[DEBUG] arrays keys:", list(atoms.arrays.keys()))
    print("[DEBUG] constraints:", atoms.constraints)
    if "move_mask" in atoms.arrays:
        print("[DEBUG] move_mask shape:", np.asarray(atoms.arrays["move_mask"]).shape)
    '''

    if atoms.cell is None or atoms.cell.volume <= 0:
        raise SystemExit("extxyz header에 Lattice가 없거나 cell이 유효하지 않습니다.")

    if WRAP_POS:
        atoms.wrap(eps=1e-12)

    present = sorted(set(atoms.get_chemical_symbols()))
    specorder = present if SPEC_ORDER is None else SPEC_ORDER

    missing = [s for s in present if s not in specorder]
    if missing:
        raise SystemExit(
            f"extxyz에 있는 원소가 SPEC_ORDER에 없습니다: {missing}\n"
            f"present={present}\nSPEC_ORDER={specorder}"
        )

    type_map = {sym: i + 1 for i, sym in enumerate(specorder)}
    types = np.array([type_map[s] for s in atoms.get_chemical_symbols()], dtype=int)

    masses = []
    for sym in specorder:
        Z = atomic_numbers[sym]
        m = float(atomic_masses[Z])
        masses.append(m)

    pos = atoms.get_positions()
    cell = atoms.cell.array

    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = cell_to_lammps_triclinic(cell)

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

    if USE_MOVE_MASK:
        frozen_ids = get_frozen_atom_ids(atoms)
    
        if len(frozen_ids) == 0:
            print("[INFO] No fully frozen atoms detected from move_mask/constraints.")
        else:
            print(f"[INFO] fully frozen atoms = {len(frozen_ids)}")
            print("[INFO] frozen atom ids:")
            print(" ".join(map(str, frozen_ids)))
    
            freeze_text = build_freeze_lammps_input(frozen_ids)
    
            print("\n[INFO] LAMMPS input snippet:")
            print(freeze_text)
    
            if WRITE_FREEZE_INPUT:
                Path(OUT_FREEZE_INPUT).write_text(freeze_text, encoding="utf-8")
                print(f"[OK] wrote: {OUT_FREEZE_INPUT}")


if __name__ == "__main__":
    main()
