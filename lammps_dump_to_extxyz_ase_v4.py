#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAMMPS dump(lammpstrj) -> extxyz  [ASE로 출력, type->원소 자동 로드 지원]

- argparse 없음. 아래 INPUT BLOCK만 수정하고 실행:
    python lammps_dump_to_extxyz_ase.py

지원:
- BOX BOUNDS (orthorhombic)
- BOX BOUNDS xy xz yz (triclinic)
- ATOMS 컬럼: id type (xu yu zu | x y z) (fx fy fz optional)
- grainmap TSV(id -> grain_num/type/sequence) 병합(선택)
- type->species 매핑 자동:
    1) TYPE_TO_SPECIES dict
    2) TYPEMAP_TXT
    3) DATA_FILE의 Masses 섹션(# element 주석)

필요:
- numpy
- ase (pip install ase)
"""

# =========================
# INPUT BLOCK (edit here)
# =========================
DUMP_IN    = "dump_300K_0GPa.lammpstrj"
OUT_EXTXYZ = "traj_H2O_541.extxyz"

# type 매핑
# - TYPE_TO_SPECIES가 있으면 그걸 최우선 사용
# - 없으면 TYPEMAP_TXT
# - 그것도 없으면 DATA_FILE의 Masses에서 읽기
TYPEMAP_TXT = None
DATA_FILE   = "../input_struct/541_1Na.data"

TYPE_TO_SPECIES = None
# 예:
# TYPE_TO_SPECIES = {1: "H", 2: "C", 3: "N", 4: "O"}

# dump에 xu/yu/zu가 있으면 그걸 우선 사용
PREFERRED_POS_COLS = ("xu", "yu", "zu")   # fallback: ("x", "y", "z")

# forces 컬럼이 있다면 extxyz에 forces 배열로 기록
WRITE_FORCES = True

# grain 옵션
# False: dump만 읽고 grain 관련 처리는 전부 생략
# True : grainmap TSV를 읽어서 grain arrays를 extxyz에 저장
USE_GRAIN = False
GRAINMAP_TSV = "../../ref_grainmap.tsv"
# =========================


from pathlib import Path
import numpy as np


def read_grainmap(path: str):
    gm = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"grainmap not found: {path}")

    for ln in p.read_text().splitlines():
        if not ln.strip() or ln.startswith("id"):
            continue
        cols = ln.split()
        if len(cols) < 4:
            continue
        aid = int(cols[0])
        gm[aid] = (int(cols[1]), int(cols[2]), int(cols[3]))
    return gm


def load_type_map_from_typemap_txt(path):
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        return None

    m = {}
    for ln in p.read_text().splitlines():
        if not ln.strip() or ln.startswith("#") or ln.startswith("ELEM=") or ln.startswith("mass "):
            continue
        cols = ln.split()
        # lines like: "1  H  1.00794"
        if len(cols) >= 2 and cols[0].isdigit():
            m[int(cols[0])] = cols[1]
    return m if m else None


def load_type_map_from_data_masses(path):
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        return None

    lines = p.read_text().splitlines()

    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("masses"))
    except StopIteration:
        return None

    i = i0 + 1

    # skip blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    m = {}
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            break

        # expected: "1 12.01100000 # C"
        if "#" in ln:
            left, right = ln.split("#", 1)
            cols = left.split()
            sym = right.strip().split()[0] if right.strip() else None
            if len(cols) >= 2 and cols[0].isdigit() and sym:
                m[int(cols[0])] = sym
        i += 1

    return m if m else None


def get_type_to_species():
    if TYPE_TO_SPECIES is not None:
        return TYPE_TO_SPECIES

    m = load_type_map_from_typemap_txt(TYPEMAP_TXT)
    if m is not None:
        return m

    m = load_type_map_from_data_masses(DATA_FILE)
    if m is not None:
        return m

    raise RuntimeError(
        "type->species 매핑을 만들 수 없습니다.\n"
        "1) TYPE_TO_SPECIES를 dict로 직접 지정하거나,\n"
        "2) TYPEMAP_TXT를 준비하거나,\n"
        "3) DATA_FILE Masses 라인에 '# 원소' 주석이 있어야 합니다."
    )


def lattice_from_box(bounds, triclinic: bool):
    if triclinic:
        (xlo, xhi, xy) = bounds[0]
        (ylo, yhi, xz) = bounds[1]
        (zlo, zhi, yz) = bounds[2]

        a = np.array([xhi - xlo, 0.0, 0.0], dtype=float)
        b = np.array([xy, yhi - ylo, 0.0], dtype=float)
        c = np.array([xz, yz, zhi - zlo], dtype=float)
    else:
        (xlo, xhi) = bounds[0]
        (ylo, yhi) = bounds[1]
        (zlo, zhi) = bounds[2]

        a = np.array([xhi - xlo, 0.0, 0.0], dtype=float)
        b = np.array([0.0, yhi - ylo, 0.0], dtype=float)
        c = np.array([0.0, 0.0, zhi - zlo], dtype=float)

    return np.vstack([a, b, c])


def main():
    try:
        from ase import Atoms
        import ase.io
    except Exception as e:
        raise SystemExit(f"ASE가 필요합니다. 설치: pip install ase\n원인: {e}")

    type_to_species = get_type_to_species()

    if USE_GRAIN:
        gm = read_grainmap(GRAINMAP_TSV)
    else:
        gm = None

    lines = Path(DUMP_IN).read_text().splitlines()
    frames = []

    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        step = int(lines[i + 1].strip())

        if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError("Unexpected dump format: missing NUMBER OF ATOMS")
        n = int(lines[i + 3].strip())

        if not lines[i + 4].startswith("ITEM: BOX BOUNDS"):
            raise ValueError("Unexpected dump format: missing BOX BOUNDS")
        box_hdr = lines[i + 4]
        triclinic = ("xy" in box_hdr and "xz" in box_hdr and "yz" in box_hdr)

        bounds = []
        for k in range(3):
            parts = lines[i + 5 + k].split()
            if triclinic:
                bounds.append((float(parts[0]), float(parts[1]), float(parts[2])))
            else:
                bounds.append((float(parts[0]), float(parts[1])))

        cell = lattice_from_box(bounds, triclinic)

        atoms_hdr = lines[i + 8]
        if not atoms_hdr.startswith("ITEM: ATOMS"):
            raise ValueError("Unexpected dump format: missing ATOMS")
        cols = atoms_hdr.replace("ITEM: ATOMS", "").split()
        col_idx = {name: idx for idx, name in enumerate(cols)}

        if "id" not in col_idx or "type" not in col_idx:
            raise ValueError(f"Dump must contain 'id' and 'type'. Found: {cols}")

        if all(k in col_idx for k in PREFERRED_POS_COLS):
            xk, yk, zk = PREFERRED_POS_COLS
        elif all(k in col_idx for k in ("x", "y", "z")):
            xk, yk, zk = ("x", "y", "z")
        else:
            raise ValueError(f"Dump must contain {PREFERRED_POS_COLS} or x y z. Found: {cols}")

        has_f = all(k in col_idx for k in ("fx", "fy", "fz"))
        if WRITE_FORCES and not has_f:
            print("[WARN] WRITE_FORCES=True but dump has no fx fy fz columns. Forces will be omitted.")

        rec = []
        for k in range(n):
            parts = lines[i + 9 + k].split()

            aid = int(parts[col_idx["id"]])
            atype = int(parts[col_idx["type"]])

            sp = type_to_species.get(atype)
            if sp is None:
                raise ValueError(f"type {atype} not in type_to_species mapping")

            x = float(parts[col_idx[xk]])
            y = float(parts[col_idx[yk]])
            z = float(parts[col_idx[zk]])

            if has_f:
                fx = float(parts[col_idx["fx"]])
                fy = float(parts[col_idx["fy"]])
                fz = float(parts[col_idx["fz"]])
            else:
                fx = fy = fz = 0.0

            if USE_GRAIN:
                gnum, gtyp, gseq = gm.get(aid, (-1, -1, -1))
                rec.append((aid, sp, x, y, z, gnum, gtyp, gseq, fx, fy, fz))
            else:
                rec.append((aid, sp, x, y, z, fx, fy, fz))

        rec.sort(key=lambda t: t[0])

        symbols = [t[1] for t in rec]
        pos = np.array([[t[2], t[3], t[4]] for t in rec], dtype=float)

        at = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
        at.wrap()
        at.info["step"] = step

        if USE_GRAIN:
            at.arrays["grain_num"] = np.array([t[5] for t in rec], dtype=int)
            at.arrays["grain_type"] = np.array([t[6] for t in rec], dtype=int)
            at.arrays["intra_grain_sequence"] = np.array([t[7] for t in rec], dtype=int)

        if WRITE_FORCES and has_f:
            if USE_GRAIN:
                at.arrays["forces"] = np.array([[t[8], t[9], t[10]] for t in rec], dtype=float)
            else:
                at.arrays["forces"] = np.array([[t[5], t[6], t[7]] for t in rec], dtype=float)

        frames.append(at)
        i = i + 9 + n

    ase.io.write(OUT_EXTXYZ, frames, format="extxyz")
    print(f"[OK] wrote: {OUT_EXTXYZ} (frames={len(frames)})")


if __name__ == "__main__":
    main()
