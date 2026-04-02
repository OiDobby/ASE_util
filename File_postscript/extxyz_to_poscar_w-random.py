#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, List
import re
import random
from ase.io import iread, write

# =========================
# ========= INPUT =========
# =========================
INPUT = {
    "input_file": "./1F/traj_1F_551.extxyz",   # extxyz file to convert
    "output_dir": "./for_DFT/poscars_551_1F",  # output directory

    "start_index": 0,       # starting number for POSCAR_step
    "number_padding": 0,    # e.g. 6 -> POSCAR_step000123

    "resume_from_existing": False, # if existing files are found, continue from the next index
    "overwrite": True,             # overwrite on name collision (default False)

    # VASP writing options
    "vasp_direct": True,    # True=Direct (fractional), False=Cartesian
    "vasp_sort": False,     # sort elements
    "vasp5": True,          # VASP5 header

    # POSCAR first-line comment template
    "comment_template": "{src} | frame {frame} -> step {step}",

    # === selection / sampling options ===
    "sample_mode": "uniform",        # "sequential" | "random" | "uniform"
    "skip_initial": 2500,            # number of initial frames to skip
    "sample_count": 3 ,              # number of frames to extract (None = all)
    "random_seed": 42,               # random seed (for reproducibility)
    "sample_extxyz_path": "./for_DFT/sampled_551_1F.extxyz",  # extxyz collecting only selected frames
}
# =========================
# ======= END INPUT =======
# =========================


def find_next_start_index(outdir: Path, start_index: int, resume: bool) -> int:
    if not resume:
        return int(start_index)
    pat = re.compile(r"^POSCAR_step(\d+)$")
    max_seen: Optional[int] = None
    for p in outdir.glob("POSCAR_step*"):
        m = pat.match(p.name)
        if m:
            try:
                idx = int(m.group(1))
                max_seen = idx if max_seen is None else max(max_seen, idx)
            except ValueError:
                pass
    if max_seen is None:
        return int(start_index)
    return max(max_seen + 1, int(start_index))


def format_step_name(step: int, pad: int) -> str:
    return f"POSCAR_step{step:0{pad}d}" if pad and pad > 0 else f"POSCAR_step{step}"


def _sanitize_comment(s: str) -> str:
    return " ".join(str(s).splitlines()).strip()


def write_poscar_compat(out_path: Path, atoms, *, direct: bool, sort: bool, vasp5: bool, comment: str):
    comment = _sanitize_comment(comment)
    if 'momenta' in atoms.arrays:
        del atoms.arrays['momenta']
    try:
        write(
            filename=str(out_path),
            images=atoms,
            format="vasp",
            direct=direct,
            sort=sort,
            vasp5=vasp5,
            label=comment,
        )
        return
    except TypeError as e:
        if "label" not in str(e):
            raise

    write(
        filename=str(out_path),
        images=atoms,
        format="vasp",
        direct=direct,
        sort=sort,
        vasp5=vasp5,
    )
    try:
        txt = out_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        txt = out_path.read_text(encoding="latin-1").splitlines()
    if txt:
        txt[0] = comment
        out_path.write_text("\n".join(txt) + "\n", encoding="utf-8")


def count_frames(extxyz_path: Path) -> int:
    n = 0
    for _ in iread(str(extxyz_path), format="extxyz"):
        n += 1
    return n


def compute_selected_indices(
    total_frames: int,
    skip_initial: int,
    sample_count: Optional[int],
    sample_mode: str,
    random_seed: int,
) -> List[int]:
    # Available frame range
    start = max(0, int(skip_initial))
    if start >= total_frames:
        return []

    available = total_frames - start
    if sample_count is None or sample_count >= available:
        return list(range(start, total_frames))

    k = int(sample_count)

    if sample_mode == "sequential":
        return list(range(start, start + k))
    if sample_mode == "random":
        random.seed(int(random_seed))
        pool = list(range(start, total_frames))
        sel = random.sample(pool, k)
        sel.sort()
        return sel
    if sample_mode == "uniform":
        step = available / k
        return [start + int(i * step) for i in range(k)]

    raise ValueError(f"Unknown sample_mode: {sample_mode}")


def main():
    cfg = INPUT

    infile = Path(cfg.get("input_file", "")).expanduser().resolve()
    if not infile.exists():
        raise SystemExit(f"[ERROR] Input file not found: {infile}")

    outdir = Path(cfg.get("output_dir", "./poscar_out")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    step = find_next_start_index(
        outdir,
        int(cfg.get("start_index", 0)),
        bool(cfg.get("resume_from_existing", True)),
    )

    pad = int(cfg.get("number_padding", 0))
    overwrite = bool(cfg.get("overwrite", False))
    direct = bool(cfg.get("vasp_direct", True))
    sort = bool(cfg.get("vasp_sort", False))
    vasp5 = bool(cfg.get("vasp5", True))
    cmt_tpl = str(cfg.get("comment_template", "{src} | frame {frame} -> step {step}"))

    skip_initial = int(cfg.get("skip_initial", 0))
    sample_count = cfg.get("sample_count", None)
    sample_count = None if sample_count in (None, "None", "") else int(sample_count)
    sample_mode = cfg.get("sample_mode", "sequential")
    random_seed = int(cfg.get("random_seed", 42))
    sample_extxyz_path = cfg.get("sample_extxyz_path", None)
    sample_extxyz_path = str(sample_extxyz_path) if sample_extxyz_path else None

    print("=== EXTXYZ -> POSCAR (with optional sampling) ===")
    print(f"- input: {infile}")
    print(f"- output dir: {outdir}")
    print(f"- mode: {'Direct' if direct else 'Cartesian'}, vasp5={vasp5}, sort={sort}")
    print(f"- skip_initial: {skip_initial}, sample_count: {sample_count}, sample_mode: {sample_mode}")

    # 1) Count total number of frames
    total = count_frames(infile)
    if total == 0:
        print("[INFO] No frames found in the input file.")
        return

    # 2) Determine selected frame indices
    selected_list = compute_selected_indices(
        total_frames=total,
        skip_initial=skip_initial,
        sample_count=sample_count,
        sample_mode=sample_mode,
        random_seed=random_seed,
    )
    selected_set: Set[int] = set(selected_list)

    print(f"- total frames: {total}")
    print(f"- selected frames: {len(selected_list)}"
          f" ({'none' if not selected_list else f'{selected_list[0]}..{selected_list[-1]} (sorted)'})")

    # Prepare sampled extxyz output (overwrite if exists)
    sample_extxyz_file: Optional[Path] = None
    if sample_extxyz_path:
        sample_extxyz_file = Path(sample_extxyz_path).expanduser().resolve()
        if sample_extxyz_file.exists():
            sample_extxyz_file.unlink()  # create a new file

    saved = 0
    # 3) Second pass: process only selected frames
    for i, atoms in enumerate(iread(str(infile), format="extxyz")):
        if i not in selected_set:
            continue

        name = format_step_name(step, pad)
        out_path = outdir / name

        if out_path.exists() and not overwrite:
            while out_path.exists():
                step += 1
                name = format_step_name(step, pad)
                out_path = outdir / name

        comment = cmt_tpl.format(src=str(infile), frame=i, step=step)

        write_poscar_compat(
            out_path,
            atoms,
            direct=direct,
            sort=sort,
            vasp5=vasp5,
            comment=comment,
        )

        # Also save only the selected frames into a separate extxyz file (append mode)
        if sample_extxyz_file is not None:
            write(
                filename=str(sample_extxyz_file),
                images=atoms,
                format="extxyz",
                append=True,
            )

        print(f"[SAVE] {out_path.name}  <- frame {i}")
        step += 1
        saved += 1

    print("=== Done ===")
    print(f"- saved POSCAR files: {saved}")
    if sample_extxyz_file is not None:
        print(f"- sampled extxyz saved: {sample_extxyz_file}")


if __name__ == "__main__":
    main()
