#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
from typing import Optional

import numpy as np

from ase import units
from ase.constraints import FixAtoms, FixCartesian
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

try:
    from nequip.dynamics.nosehoover import NoseHoover
except Exception:
    NoseHoover = None

from sevenn.calculator import SevenNetCalculator, SevenNetD3Calculator


# =========================================================
# ===================== 사용자 옵션 =======================
# 입력/모델
INPUT_XYZ  = "input.extxyz"   # !!Should be confirm!!
MODEL_PATH = "7Net-Omni"
DEVICE     = "cuda"

# SevenNet 계산기 옵션
MODAL     = "mpa"             # !!Should be confirm!!
USE_D3    = True              # !!Should be confirm!!
FILE_TYPE = "checkpoint"

# D3 보정 옵션 (USE_D3=True일 때만 사용)
D3_DAMPING_TYPE    = "damp_zero"
D3_FUNCTIONAL_NAME = "pbe"
D3_VDW_CUTOFF      = 9000
D3_CN_CUTOFF       = 1600

# MD 적분기
INTEGRATOR    = "langevin"  # "nh" or "langevin"
TEMPERATURE_K = 300.0
DT_FS         = 0.5         # Time step !!Should be confirm!!
TAU_FS        = 100.0
GAMMA_FS      = 0.05        # Langevin friction (1/fs); None이면 1/TAU_FS

# 반복 실행 / 물리시간 기준 종료 !!Should be confirm!!
N_RUNS      = 1             # MD run을 몇개나 돌릴지?
MD_TOTAL_PS = 20            # 몇 ps 까지 돌릴지?
WRITE_EVERY = 10            # 저장은 몇 frame 마다?
TAG         = "Omni"
SEED0       = 1234

# 월타임 체크포인트
ENABLE_CHECKPOINT = True
WALL_LIMIT_HOURS  = 48.0
WALL_SAFETY_MIN   = 5.0
CKPT_PREFIX       = f"{TAG}_ckpt"
RESUME_FROM_CKPT  = True

# 런타임 로그
RUNTIME_LOG_PATH = f"{TAG}_runtime.txt"
LOG_EVERY_STEP   = False
LOG_ON_WRITE     = True
LOG_ON_CKPT      = True
# =========================================================
# =========================================================


# ----------------- 제약(Constraints) -----------------
def apply_move_mask(atoms):
    """
    atoms.arrays:
      - move_mask: (N,) or (N,3), 1=자유, 0=고정
      - fix_mask : (N,) or (N,3), 1=고정, 0=자유
    """
    arrs = atoms.arrays
    cons = []

    def _to_bool3(mask_row):
        if np.isscalar(mask_row) or np.ndim(mask_row) == 0:
            v = bool(mask_row)
            return (v, v, v)
        if len(mask_row) == 3:
            return tuple(bool(x) for x in mask_row)
        v = bool(mask_row[0])
        return (v, v, v)

    if "move_mask" in arrs:
        mask = np.asarray(arrs["move_mask"])
        if mask.ndim == 1:
            mask = mask.reshape(-1, 1)

        fixed_idx = [i for i in range(len(atoms)) if np.all(mask[i] == 0)]
        if fixed_idx:
            cons.append(FixAtoms(indices=fixed_idx))

        if mask.shape[1] == 3:
            for i in range(len(atoms)):
                allow_axes = tuple(bool(x) for x in mask[i])
                fix_axes = tuple(not x for x in allow_axes)
                if any(fix_axes) and not all(fix_axes):
                    cons.append(FixCartesian(indices=[i], mask=fix_axes))

    elif "fix_mask" in arrs:
        mask = np.asarray(arrs["fix_mask"])
        if mask.ndim == 1:
            mask = mask.reshape(-1, 1)

        fixed_idx = [i for i in range(len(atoms)) if np.any(mask[i] != 0)]
        if fixed_idx:
            cons.append(FixAtoms(indices=fixed_idx))

        if mask.shape[1] == 3:
            for i in range(len(atoms)):
                fix_axes = _to_bool3(mask[i])
                if any(fix_axes) and not all(fix_axes):
                    cons.append(FixCartesian(indices=[i], mask=fix_axes))

    if cons:
        atoms.set_constraint(cons)


# ----------------- Calculator -----------------
def build_calculator():
    if USE_D3:
        return SevenNetD3Calculator(
            model=MODEL_PATH,
            file_type=FILE_TYPE,
            device=DEVICE,
            damping_type=D3_DAMPING_TYPE,
            functional_name=D3_FUNCTIONAL_NAME,
            vdw_cutoff=D3_VDW_CUTOFF,
            cn_cutoff=D3_CN_CUTOFF,
            modal=MODAL,
        )

    return SevenNetCalculator(
        model=MODEL_PATH,
        file_type=FILE_TYPE,
        device=DEVICE,
        modal=MODAL,
    )


# ----------------- Integrator -----------------
def build_dyn(atoms, integrator: str, dt_fs: float, temperature_K: float,
              tau_fs: float, gamma_fs: Optional[float]):
    if integrator.lower() in ("nh", "nosehoover", "nose-hoover"):
        assert NoseHoover is not None, "NoseHoover integrator unavailable in this environment."
        dof = 3 * len(atoms) + 1
        nvt_q = dof * units.kB * temperature_K * (tau_fs ** 2)
        return NoseHoover(
            atoms=atoms,
            timestep=dt_fs * units.fs,
            temperature=temperature_K,
            nvt_q=nvt_q,
        )

    friction = gamma_fs if gamma_fs is not None else (1.0 / tau_fs)
    return Langevin(
        atoms=atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
    )


# ----------------- Output helpers -----------------
def collect_results(atoms):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    frms = float(np.sqrt((forces ** 2).mean()))

    atoms.arrays["forces"] = np.asarray(forces, dtype=np.float32)
    atoms.info["energy"] = float(energy)

    qmean = None
    try:
        charges = atoms.get_charges()
        if charges is not None:
            charges = np.asarray(charges, dtype=np.float32).reshape(-1)
            atoms.arrays["charge"] = charges
            qmean = float(charges.mean())
    except Exception:
        pass

    return energy, frms, qmean


def write_md_frame(atoms, path: str, append: bool, traj_step: int,
                   segment_id: int, sim_time_fs: float):
    atoms.wrap(eps=1e-12)
    energy, frms, qmean = collect_results(atoms)
    atoms.info["md_traj_step"] = int(traj_step)
    atoms.info["md_segment"] = int(segment_id)
    atoms.info["md_sim_fs"] = float(sim_time_fs)
    write(path, atoms, format="extxyz", append=append, write_results=False)
    return energy, frms, qmean


# ----------------- 로그 -----------------
def init_runtime_log(path: str, overwrite: bool = True):
    mode = "w" if overwrite else "a"
    with open(path, mode) as f:
        f.write("# time_iso\trun\ttraj_step\tof\tseg\tDT_wall_s\tCUM_wall_s\tdt_fs\tSIM_fs\tevent\tE\tFrms\tqmean\n")


def log_runtime(path: str, run_id: int, traj_step: int, total_steps: int,
                segment_id: int, dt_wall_s: float, cum_wall_s: float,
                dt_fs: float, sim_time_fs: float, event: str,
                energy: Optional[float] = None, frms: Optional[float] = None,
                qmean: Optional[float] = None):
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    def fmt(x):
        return "" if x is None else f"{x:.6f}"

    line = (
        f"{timestamp}\t{run_id}\t{traj_step}\t{total_steps}\t{segment_id}\t"
        f"{dt_wall_s:.6f}\t{cum_wall_s:.2f}\t{dt_fs:.4f}\t{sim_time_fs:.1f}\t"
        f"{event}\t{fmt(energy)}\t{fmt(frms)}\t{fmt(qmean)}\n"
    )
    with open(path, "a") as f:
        f.write(line)


# ----------------- 체크포인트 -----------------
def walltime_exceeded(start_monotonic: float) -> bool:
    if not ENABLE_CHECKPOINT:
        return False
    elapsed_h = (time.monotonic() - start_monotonic) / 3600.0
    return elapsed_h >= max(0.0, WALL_LIMIT_HOURS - WALL_SAFETY_MIN / 60.0)


def write_ckpt(atoms, run_id: int, traj_step: int, sim_time_fs: float,
               segment_id: int):
    path = f"{CKPT_PREFIX}.extxyz"
    atoms.info["md_run"] = int(run_id)
    atoms.info["md_traj_step"] = int(traj_step)
    atoms.info["md_sim_fs"] = float(sim_time_fs)
    atoms.info["md_segment"] = int(segment_id)
    write(path, atoms, format="extxyz", append=False, write_results=False)
    return path


def load_ckpt_if_any(atoms_base):
    if not (ENABLE_CHECKPOINT and RESUME_FROM_CKPT):
        return atoms_base, 0, 0.0, 0, False

    path = f"{CKPT_PREFIX}.extxyz"
    if not os.path.exists(path):
        return atoms_base, 0, 0.0, 0, False

    atoms = read(path, format="extxyz")
    traj_step = int(atoms.info.get("md_traj_step", 0))
    sim_time_fs = float(atoms.info.get("md_sim_fs", 0.0))
    segment_id = int(atoms.info.get("md_segment", 0))
    return atoms, traj_step, sim_time_fs, segment_id, True


# ===================== 메인 루프 =====================
def run_one(run_id: int, atoms_base):
    atoms = atoms_base.copy()
    atoms.calc = build_calculator()
    apply_move_mask(atoms)

    atoms, traj_step, sim_time_fs, segment_id, resumed = load_ckpt_if_any(atoms)
    if resumed:
        atoms.calc = build_calculator()
        apply_move_mask(atoms)
    else:
        np.random.seed(SEED0 + run_id)
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
        Stationary(atoms)
        ZeroRotation(atoms)

    target_time_fs = MD_TOTAL_PS * 1000.0
    total_steps = int(np.ceil(target_time_fs / DT_FS))
    dyn = build_dyn(atoms, INTEGRATOR, DT_FS, TEMPERATURE_K, TAU_FS, GAMMA_FS)

    out_file = f"{TAG}_run{run_id:03d}.extxyz"
    if run_id == 1 and traj_step == 0 and sim_time_fs == 0.0:
        init_runtime_log(RUNTIME_LOG_PATH, overwrite=True)

    start_wall = time.monotonic()
    since_last_write = 0

    if traj_step == 0:
        energy, frms, qmean = write_md_frame(
            atoms, out_file, append=False,
            traj_step=traj_step,
            segment_id=segment_id,
            sim_time_fs=sim_time_fs,
        )
        log_runtime(
            RUNTIME_LOG_PATH, run_id, traj_step, total_steps, segment_id,
            0.0, time.monotonic() - start_wall, 0.0, sim_time_fs,
            "start", energy, frms, qmean,
        )
        msg = f"[run {run_id}] start | E={energy:.6f} eV | F_rms={frms:.6f} eV/A"
        if qmean is not None:
            msg += f" | <q>={qmean:.6f} e"
        print(msg)

    while sim_time_fs < target_time_fs:
        if walltime_exceeded(start_wall):
            path = write_ckpt(atoms, run_id, traj_step, sim_time_fs, segment_id)
            if LOG_ON_CKPT:
                log_runtime(
                    RUNTIME_LOG_PATH, run_id, traj_step, total_steps, segment_id,
                    0.0, time.monotonic() - start_wall, 0.0, sim_time_fs,
                    f"ckpt:{os.path.basename(path)}",
                )
            print(f"[run {run_id}] wall-limit checkpoint written: {path}")
            break

        try:
            t0 = time.monotonic()
            dyn.run(1)
            dt_wall = time.monotonic() - t0

            sim_time_fs += DT_FS
            traj_step += 1
            since_last_write += 1

            should_write = (since_last_write >= WRITE_EVERY) or (sim_time_fs >= target_time_fs)
            if should_write or LOG_EVERY_STEP:
                energy, frms, qmean = collect_results(atoms)
            else:
                energy = frms = qmean = None

            if should_write:
                atoms.wrap(eps=1e-12)
                atoms.info["md_traj_step"] = int(traj_step)
                atoms.info["md_segment"] = int(segment_id)
                atoms.info["md_sim_fs"] = float(sim_time_fs)
                write(out_file, atoms, format="extxyz", append=True, write_results=False)

                if LOG_ON_WRITE:
                    log_runtime(
                        RUNTIME_LOG_PATH, run_id, traj_step, total_steps, segment_id,
                        dt_wall, time.monotonic() - start_wall, DT_FS, sim_time_fs,
                        "write", energy, frms, qmean,
                    )

                since_last_write = 0
                msg = (
                    f"[run {run_id}] traj {traj_step:7d} | "
                    f"SIM={sim_time_fs / 1000.0:8.3f} ps | "
                    f"E={energy:.6f} eV | F_rms={frms:.6f} eV/A"
                )
                if qmean is not None:
                    msg += f" | <q>={qmean:.6f} e"
                print(msg)

            elif LOG_EVERY_STEP:
                log_runtime(
                    RUNTIME_LOG_PATH, run_id, traj_step, total_steps, segment_id,
                    dt_wall, time.monotonic() - start_wall, DT_FS, sim_time_fs,
                    "step", energy, frms, qmean,
                )

        except Exception as err:
            print(f"[EXC][run {run_id}] traj={traj_step}: {err}")
            break

    print(f"[run {run_id}] finished.")


def main():
    atoms_base = read(INPUT_XYZ, format="extxyz")
    atoms_base.set_pbc([True, True, True])

    for run_id in range(1, N_RUNS + 1):
        run_one(run_id, atoms_base)

    print("Done.")


if __name__ == "__main__":
    main()
