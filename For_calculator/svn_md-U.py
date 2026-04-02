#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, datetime
from typing import Optional, List, Tuple

import numpy as np
import torch

from ase.io import read, write
from ase import units
from ase.constraints import FixAtoms, FixCartesian
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.langevin import Langevin

# Nose–Hoover (환경에 설치되어 있어야 함)
try:
    from nequip.dynamics.nosehoover import NoseHoover
except Exception:
    NoseHoover = None

from sevenn.calculator import SevenNetCalculator, SevenNetD3Calculator


# =========================================================
# ===================== 사용자 옵션 =======================
# 입력/모델
INPUT_XYZ         = "/home/jwkuha/Ag_H2O/model_tmp/MLP_MD/input_files/541/541_H2O.extxyz"          # 초기 구조 (extxyz; move_mask:I:3 / fix_mask:I:3 권장)
EFQ_MODEL_PATH    = "/home/jwkuha/scr/SevenNet-Omni.pth"    # EFQ(에너지/힘/전하) 모델 (TorchScript)
SPECIES_MAP       = {"Ag": "Ag", "O": "O", "H": "H"}
DEVICE            = "cuda"                         # "cuda" or "cpu"

# U 모델 (저장 시점에만 호출)
USE_U_MODEL       = False
U_MODEL_PATH      = "/home/jwkuha/Ag_H2O/model_tmp/MLP_MD/deploy_U-cuda.pth"      # TorchScript 권장
U_OUT_KEYS        = ("electrode_potential", "U", "u", "potential")
U_R_MAX           = None                           # None → EFQ calc r_max 사용, 없으면 5.0 Å
U_AREA_NORMALIZE  = True                           # 출력이 면적*A^2 가 곱해진 값이면 True
U_AREA_PLANE      = "auto"                         # "auto"|"xy"|"yz"|"zx"
U_SCALE           = 1.0                            # 최종 보정: scale * U + offset
U_OFFSET          = 0.0

# MD 적분기
INTEGRATOR        = "langevin"                     # "nh" or "langevin"
TEMPERATURE_K     = 300.0
DT_FS             = 1.0                            # 기본 시간 스텝(fs)
TAU_FS            = 100.0                          # NH용 특성시간
GAMMA_FS          = 0.05                           # Langevin friction (1/fs); None이면 1/TAU_FS

# 반복 실행 / 물리시간 기준 종료
N_RUNS            = 1
MD_TOTAL_PS       = 20                            # 총 물리시간(ps) 목표 (예: 100 ps)
WRITE_EVERY       = 10                            # 프레임 저장 주기(스텝 카운터 기반; 롤백 시 since_last_write 사용)
TAG               = "431_EFQ_U"
SEED0             = 1234

# 이상치 감지
CHECK_EVERY_STEP      = False
FRMS_MAX              = 5.0                         # eV/Å
FMAX_MAX              = 8.5                         # eV/Å
DENERGY_MAX_PER_ATOM  = 2.00                        # eV/atom
ALLOW_NAN_INF         = False

# 롤백(기본) & 한계
ROLLBACK_STEPS        = 100                         # 기본 되돌림 폭(스텝, traj 기준)
ROLLBACK_MAX_PER_RUN  = 2000                        # 한 런 최대 롤백 횟수

# [추가] Adaptive rollback
ADAPTIVE_ROLLBACK     = True
ROLLBACK_MULTIPLIER   = 2.0                         # bad 반복 시 되돌림 폭 ×2
ROLLBACK_STEPS_MAX    = 5000                        # 적응 증가 상한

# [추가] Snapshot pool
SNAPSHOT_INTERVAL     = 1                           # N step마다 스냅샷 (traj 기준)
SNAPSHOT_POOL_SIZE    = 2000                        # 보관 스냅샷 개수

# [선택] 롤백 직후 진정 단계
ENABLE_COOLDOWN             = True
DT_SCALE_ON_RECOVERY        = 0.5                    # 롤백 직후 임시 dt 배수
COOLDOWN_STEPS_AFTER_ROLLBACK = 50                   # 진정 단계 길이(스텝)
LANGEVIN_GAMMA_BOOST        = 10.0                    # Langevin일 때 마찰 강화 배수

# 월타임 체크포인트
ENABLE_CHECKPOINT     = True
WALL_LIMIT_HOURS      = 48.0
WALL_SAFETY_MIN       = 5.0                          # 종료 직전 안전 마진(분)
CKPT_PREFIX           = f"{TAG}_ckpt"
RESUME_FROM_CKPT      = True

# 런타임 로그
RUNTIME_LOG_PATH      = f"{TAG}_runtime.txt"
LOG_EVERY_STEP        = False
LOG_ON_WRITE          = True
LOG_ON_BAD            = True
LOG_ON_ROLLBACK       = True
LOG_ON_CKPT           = True
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
        if np.isscalar(mask_row) or (np.ndim(mask_row) == 0):
            v = bool(mask_row)
            return (v, v, v)
        if len(mask_row) == 3:
            return tuple(bool(x) for x in mask_row)
        v = bool(mask_row[0])
        return (v, v, v)

    if "move_mask" in arrs:
        m = np.asarray(arrs["move_mask"])
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        fixed_idx = [i for i in range(len(atoms)) if (np.all(m[i] == 0))]
        if fixed_idx:
            cons.append(FixAtoms(indices=fixed_idx))
        if m.shape[1] == 3:
            for i in range(len(atoms)):
                allow = tuple(bool(x) for x in m[i])
                fix_axes = tuple(not a for a in allow)
                if any(fix_axes) and not all(fix_axes):
                    cons.append(FixCartesian(indices=[i], mask=fix_axes))

    elif "fix_mask" in arrs:
        m = np.asarray(arrs["fix_mask"])
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        fixed_idx = [i for i in range(len(atoms)) if (np.any(m[i] != 0))]
        if fixed_idx:
            cons.append(FixAtoms(indices=fixed_idx))
        if m.shape[1] == 3:
            for i in range(len(atoms)):
                fix_axes = _to_bool3(m[i])
                if any(fix_axes) and not all(fix_axes):
                    cons.append(FixCartesian(indices=[i], mask=fix_axes))

    if cons:
        atoms.set_constraint(cons)


# ----------------- Integrator -----------------
def build_dyn(atoms, integrator: str, dt_fs: float, T: float, tau_fs: float, gamma_fs: Optional[float]):
    if integrator.lower() in ("nh", "nosehoover", "nose-hoover"):
        assert NoseHoover is not None, "NoseHoover integrator unavailable in this environment."
        g = 3*len(atoms) + 1
        nvt_q = g * units.kB * T * (tau_fs ** 2)
        return NoseHoover(atoms=atoms, timestep=dt_fs * units.fs, temperature=T, nvt_q=nvt_q)
    else:
        fric = gamma_fs if gamma_fs is not None else (1.0 / tau_fs)
        return Langevin(atoms=atoms, timestep=dt_fs * units.fs, temperature_K=T, friction=fric)


# ----------------- 속도 재스케일 -----------------
def rescale_velocities_to_T(atoms, T_K: float):
    v = atoms.get_velocities()
    if v is None:
        return
    m = atoms.get_masses()[:, None]
    KE = 0.5 * (m * (v*v)).sum()
    dof = 3*len(atoms)
    if dof <= 0:
        return
    T_cur = (2.0 * KE) / (dof * units.kB)
    if not np.isfinite(T_cur) or T_cur <= 1e-14:
        return
    scale = float(T_K / T_cur) ** 0.5
    atoms.set_velocities(v * scale)
    Stationary(atoms); ZeroRotation(atoms)


# ----------------- 로그 -----------------
def init_runtime_log(path: str, overwrite: bool = True):
    mode = "w" if overwrite else "a"
    with open(path, mode) as f:
        f.write("# time_iso\trun\ttraj_step\tof\tseg\tDT_wall_s\tCUM_wall_s\tdt_fs\tSIM_fs\tevent\tE\tFrms\tqmean\n")

def log_runtime(path: str, run_id: int, traj_step: int, total_steps: int,
                seg: int, dt_wall_s: float, cum_wall_s: float, dt_fs: float, sim_fs: float,
                event: str, E: Optional[float]=None, Frms: Optional[float]=None, qmean: Optional[float]=None):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    def fmt(x): return "" if x is None else f"{x:.6f}"
    line = (
        f"{ts}\t{run_id}\t{traj_step}\t{total_steps}\t{seg}\t"
        f"{dt_wall_s:.6f}\t{cum_wall_s:.2f}\t{dt_fs:.4f}\t{sim_fs:.1f}\t"
        f"{event}\t{fmt(E)}\t{fmt(Frms)}\t{fmt(qmean)}\n"
    )
    with open(path, "a") as f:
        f.write(line)


# ----------------- 스냅샷 -----------------
def snapshot(atoms):
    return (atoms.get_positions().copy(),
            atoms.get_cell().copy(),
            None if atoms.get_velocities() is None else atoms.get_velocities().copy())

def restore(atoms, snap):
    pos, cell, vel = snap
    atoms.set_positions(pos)
    atoms.set_cell(cell)
    if vel is not None:
        atoms.set_velocities(vel)

def pick_snapshot_for_rollback(
    snapshot_pool: List[Tuple[int, tuple]],
    current_traj_step: int,
    rollback_span: int,
    min_back: int = None,
):
    """
    반환: (chosen_snap, chosen_traj_step, used_mode)
      - chosen_snap: (pos, cell, vel) 또는 None
      - chosen_traj_step: 선택된 스냅샷의 traj_step (int) 또는 None
      - used_mode: "target" | "recent" | "earliest" | "none"
    폴백 체인:
      1) target = current - rollback_span 이하 중 가장 가까운 스냅샷
      2) 그게 없으면, current - min_back 이하 중 가장 가까운 스냅샷(최소한 약간이라도 뒤로)
      3) 그래도 없으면, 풀의 가장 오래된 스냅샷(earliest) 단, current보다 과거여야 함
      4) 모두 실패하면 None
    """
    if not snapshot_pool:
        return None, None, "none"

    if min_back is None:
        # 최소 폴백 폭은 스냅샷 간격 정도로 설정
        min_back = max(1, SNAPSHOT_INTERVAL)

    target = int(current_traj_step - rollback_span)

    # 1) ≤ target
    cand = [ (st, s) for (st, s) in snapshot_pool if st <= target ]
    if cand:
        st, snap = cand[-1]
        return snap, st, "target"

    # 2) ≤ (current - min_back)
    thresh = int(current_traj_step - min_back)
    cand = [ (st, s) for (st, s) in snapshot_pool if st <= thresh ]
    if cand:
        st, snap = cand[-1]
        return snap, st, "recent"

    # 3) earliest < current
    earliest_st, earliest_snap = min(snapshot_pool, key=lambda x: x[0])
    if earliest_st < current_traj_step:
        return earliest_snap, earliest_st, "earliest"

    # 4) 실패
    return None, None, "none"


def choose_snapshot_target(snapshot_pool: List[Tuple[int, tuple]], target_traj_step: int):
    """pool에서 target_traj_step 이하 중 가장 가까운 스냅샷 반환 (없으면 None)"""
    candidates = [s for (st, s) in snapshot_pool if st <= target_traj_step]
    if not candidates:
        return None
    return candidates[-1]


# ----------------- 이상치 판정 -----------------
def is_bad_state(e_prev: Optional[float], e_now: float, n_atoms: int, f_now: np.ndarray) -> bool:
    if not np.isfinite(e_now) or not np.isfinite(f_now).all():
        return not ALLOW_NAN_INF
    frms = float(np.sqrt((f_now**2).mean()))
    fmax = float(np.abs(f_now).max())
    if frms > FRMS_MAX or fmax > FMAX_MAX:
        return True
    if e_prev is not None and n_atoms > 0:
        dE_pa = abs(e_now - e_prev) / float(n_atoms)
        if dE_pa > DENERGY_MAX_PER_ATOM:
            return True
    return False


# ----------------- 체크포인트 -----------------
def walltime_exceeded(start_monotonic: float) -> bool:
    if not ENABLE_CHECKPOINT:
        return False
    elapsed_h = (time.monotonic() - start_monotonic) / 3600.0
    return elapsed_h >= max(0.0, WALL_LIMIT_HOURS - WALL_SAFETY_MIN / 60.0)

def write_ckpt(atoms, run_id: int, traj_step: int, sim_time_fs: float, segment_id: int):
    path = f"{CKPT_PREFIX}.extxyz"
    atoms.info["md_run"]       = int(run_id)
    atoms.info["md_traj_step"] = int(traj_step)
    atoms.info["md_sim_fs"]    = float(sim_time_fs)
    atoms.info["md_segment"]   = int(segment_id)
    write(path, atoms, format="extxyz", append=False, write_results=False)
    return path

def load_ckpt_if_any(atoms_base, run_id: int):
    if not (ENABLE_CHECKPOINT and RESUME_FROM_CKPT):
        return atoms_base, 0, 0.0, 0, False
    path = f"{CKPT_PREFIX}.extxyz"
    if not os.path.exists(path):
        return atoms_base, 0, 0.0, 0, False
    at = read(path, format="extxyz")
    traj0 = int(at.info.get("md_traj_step", 0))
    simfs0 = float(at.info.get("md_sim_fs", 0.0))
    seg0 = int(at.info.get("md_segment", 0))
    return at, traj0, simfs0, seg0, True


# ===================== 메인 루프 =====================
def run_one(run_id: int, atomsA):
    # 시작 구조
    atoms = atomsA.copy()

    # EFQ 계산기 (build_calc 없이 직접)
    '''
    calc = SevenNetCalculator(
        model=EFQ_MODEL_PATH,
        modal='mpa'
    )
    '''
    calc = SevenNetD3Calculator(
        model=EFQ_MODEL_PATH,
        file_type='checkpoint',
        device='cuda',
        damping_type='damp_zero',     # 또는 'damp_zero'
        functional_name='pbe',      # D3 파라미터용 함수형 (ex: pbe, b3lyp 등)
        vdw_cutoff=9000,            # 기본값 그대로 사용 가능
        cn_cutoff=1600,             # 기본값 그대로 사용 가능
        modal='omat24'
    )
    atoms.calc = calc

    # 제약
    apply_move_mask(atoms)

    # 재시작 체크포인트
    atoms, traj_step, sim_time_fs, segment_id, resumed = load_ckpt_if_any(atoms, run_id)

    # 초기 속도
    if not resumed:
        np.random.seed(SEED0 + run_id)
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
        Stationary(atoms); ZeroRotation(atoms)

    # 목표 물리시간(fs)
    TARGET_TIME_FS = MD_TOTAL_PS * 1000.0

    # 적분기 (초기)
    dyn = build_dyn(atoms, INTEGRATOR, DT_FS, TEMPERATURE_K, TAU_FS, GAMMA_FS)

    # U 모델 로드
    u_model = None
    if USE_U_MODEL and os.path.exists(U_MODEL_PATH):
        u_model = torch.jit.load(U_MODEL_PATH, map_location=DEVICE)
        u_model.eval()

    # 파일/로그
    out_file     = f"{TAG}_run{run_id:03d}.extxyz"
    out_file_bad = f"{TAG}_run{run_id:03d}_bad.extxyz"
    if run_id == 1 and traj_step == 0 and sim_time_fs == 0.0:
        init_runtime_log(RUNTIME_LOG_PATH, overwrite=True)

    start_wall = time.monotonic()

    # 시작 프레임 (필요하면 초기 저장)
    if traj_step == 0:
        atoms.wrap(eps=1e-12)
        e0 = atoms.get_potential_energy()
        f0 = atoms.get_forces()
        fr0 = float(np.sqrt((f0**2).mean()))
        atoms.arrays["forces"] = np.asarray(f0, dtype=np.float32)
        qmean0 = None
        try:
            q0 = atoms.get_charges()
            if q0 is not None:
                atoms.arrays["charge"] = np.asarray(q0, dtype=np.float32).reshape(-1)
                qmean0 = float(np.asarray(q0).mean())
        except Exception:
            pass
        
        atoms.info["energy"] = float(e0)
        atoms.info["md_traj_step"] = int(traj_step)
        atoms.info["md_segment"]   = int(segment_id)
        atoms.info["md_sim_fs"]    = float(sim_time_fs)
        write(out_file, atoms, format="extxyz", append=False, write_results=False)
        log_runtime(RUNTIME_LOG_PATH, run_id, traj_step, int(TARGET_TIME_FS/DT_FS), segment_id,
                    0.0, time.monotonic()-start_wall, 0.0, sim_time_fs, "start", e0, fr0, qmean0)
        print(f"[run {run_id}] start | E={e0:.6f} eV | F_rms={fr0:.6f} eV/Å" + (f" | <q>={qmean0:.6f} e" if qmean0 is not None else ""))

    # 스냅샷 풀 준비
    snapshot_pool: List[Tuple[int, tuple]] = []
    snapshot_pool.append((traj_step, snapshot(atoms)))

    e_prev = None  # 직전 에너지(이상치 판정에 사용)
    since_last_write = 0
    rollback_span = ROLLBACK_STEPS
    cooldown = 0
    rollback_count = 0

    while sim_time_fs < TARGET_TIME_FS:
        # 월타임 체크포인트
        if walltime_exceeded(start_wall):
            path = write_ckpt(atoms, run_id, traj_step, sim_time_fs, segment_id)
            if LOG_ON_CKPT:
                log_runtime(RUNTIME_LOG_PATH, run_id, traj_step, int(TARGET_TIME_FS/DT_FS), segment_id,
                            0.0, time.monotonic()-start_wall, 0.0, sim_time_fs, f"ckpt:{os.path.basename(path)}",
                            e_prev, None, None)
            print(f"[run {run_id}] wall-limit checkpoint written: {path}")
            break

        # 현재 스텝의 실제 dt 결정 (쿨다운이면 축소)
        dt_curr_fs = DT_FS if (not ENABLE_COOLDOWN or cooldown == 0) else (DT_FS * DT_SCALE_ON_RECOVERY)

        try:
            t0 = time.monotonic()
            dyn.run(1)
            dt_wall = time.monotonic() - t0

            # 누적 카운터 갱신
            sim_time_fs += dt_curr_fs
            traj_step   += 1
            since_last_write += 1

            # 스냅샷 저장 (traj 기준)
            if (traj_step % SNAPSHOT_INTERVAL) == 0:
                snapshot_pool.append((traj_step, snapshot(atoms)))
                if len(snapshot_pool) > SNAPSHOT_POOL_SIZE:
                    snapshot_pool.pop(0)

            # E/F/q
            e_now = atoms.get_potential_energy()
            f_now = atoms.get_forces()
            frms_now = float(np.sqrt((f_now**2).mean()))
            qmean_now = None
            try:
                q_now = atoms.get_charges()
                if q_now is not None:
                    qmean_now = float(np.asarray(q_now).mean())
            except Exception:
                pass

            # 쿨다운 종료 복원
            if ENABLE_COOLDOWN and cooldown > 0:
                cooldown -= 1
                if cooldown == 0:
                    dyn = build_dyn(atoms, INTEGRATOR, DT_FS, TEMPERATURE_K, TAU_FS, GAMMA_FS)

            # 이상치 판정
            bad = CHECK_EVERY_STEP and is_bad_state(e_prev, e_now, len(atoms), f_now)
            if bad:
                # bad 프레임 기록
                try:
                    atoms.arrays["forces"] = np.asarray(f_now, dtype=np.float32)
                    if qmean_now is not None:
                        atoms.arrays["charge"] = np.asarray(q_now, dtype=np.float32).reshape(-1)
                    atoms.info["energy"] = float(e_now)
                    atoms.info["md_traj_step"] = int(traj_step)
                    atoms.info["md_segment"]   = int(segment_id)
                    atoms.info["md_sim_fs"]    = float(sim_time_fs)
                    write(out_file_bad, atoms, format="extxyz", append=True, write_results=False)
                except Exception as err:
                    print(f"[bad-write] failed: {err}")
                if LOG_ON_BAD:
                    log_runtime(RUNTIME_LOG_PATH, run_id, traj_step, int(TARGET_TIME_FS/DT_FS), segment_id,
                                dt_wall, time.monotonic()-start_wall, dt_curr_fs, sim_time_fs, "bad", e_now, frms_now, qmean_now)

                # 롤백 한계 검사
                if rollback_count >= ROLLBACK_MAX_PER_RUN:
                    print(f"[run {run_id}] rollback aborted (count={rollback_count} >= max)")
                    break

                # 타겟 스냅샷 결정 (adaptive)
                target_traj = max(0, int(traj_step - rollback_span))
                target_snap, chosen_traj, used_mode = pick_snapshot_for_rollback(
                    snapshot_pool, traj_step, rollback_span, min_back=SNAPSHOT_INTERVAL
                )
                
                if target_snap is None:
                    print(f"[run {run_id}] rollback failed (no snapshot < current); abort run")
                    break
                
                # 롤백 실행
                restore(atoms, target_snap)
                traj_step = int(chosen_traj)   # ← 궤적 스텝도 스냅샷 시점으로 되감기
                rollback_count += 1

                mode_str = f"rollback(~{rollback_span})[{used_mode}]"
                if LOG_ON_ROLLBACK:
                    log_runtime(RUNTIME_LOG_PATH, run_id, traj_step, int(TARGET_TIME_FS/DT_FS),
                                segment_id, 0.0, time.monotonic()-start_wall, 0.0, sim_time_fs,
                                mode_str, e_now, frms_now, qmean_now)
                print(f"[run {run_id}] >>> rollback #{rollback_count} to traj {traj_step} "
                      f"(span={rollback_span}, mode={used_mode})")

                # 롤백 직후 진정 단계
                if ENABLE_COOLDOWN:
                    rescale_velocities_to_T(atoms, TEMPERATURE_K)
                    dt_tmp = DT_FS * DT_SCALE_ON_RECOVERY
                    gamma_tmp = (GAMMA_FS * LANGEVIN_GAMMA_BOOST) if (INTEGRATOR == "langevin" and GAMMA_FS is not None) else GAMMA_FS
                    dyn = build_dyn(atoms, INTEGRATOR, dt_tmp, TEMPERATURE_K, TAU_FS, gamma_tmp)
                    cooldown = COOLDOWN_STEPS_AFTER_ROLLBACK

                # 스냅샷 풀 재설정(복원 지점 기준으로 재시작)
                snapshot_pool.clear()
                snapshot_pool.append((traj_step, snapshot(atoms)))

                # 다음 롤백 폭: adaptive 증가
                if ADAPTIVE_ROLLBACK:
                    rollback_span = min(int(rollback_span * ROLLBACK_MULTIPLIER), ROLLBACK_STEPS_MAX)
                else:
                    rollback_span = ROLLBACK_STEPS

                # 기준 에너지 갱신
                atoms.wrap(eps=1e-12)
                try:
                    e_prev = atoms.get_potential_energy()
                except Exception:
                    e_prev = None

                # 저장 카운터 리셋
                since_last_write = 0
                continue  # 다음 루프

            # 정상 진행
            e_prev = e_now
            rollback_span = ROLLBACK_STEPS  # 정상 시 롤백 폭 리셋

            # 저장 조건: since_last_write OR 물리시간 종료 직전
            if (since_last_write >= WRITE_EVERY) or (sim_time_fs >= TARGET_TIME_FS):
                atoms.wrap(eps=1e-12)
                atoms.arrays["forces"] = np.asarray(f_now, dtype=np.float32)
                if qmean_now is not None:
                    atoms.arrays["charge"] = np.asarray(q_now, dtype=np.float32).reshape(-1)
                atoms.info["energy"] = float(e_now)
                atoms.info["md_traj_step"] = int(traj_step)
                atoms.info["md_segment"]   = int(segment_id)
                atoms.info["md_sim_fs"]    = float(sim_time_fs)
                
                write(out_file, atoms, format="extxyz", append=True, write_results=False)
                if LOG_ON_WRITE:
                    log_runtime(RUNTIME_LOG_PATH, run_id, traj_step, int(TARGET_TIME_FS/DT_FS), segment_id,
                                dt_wall, time.monotonic()-start_wall, dt_curr_fs, sim_time_fs, "write", e_now, frms_now, qmean_now)
                since_last_write = 0
                print(f"[run {run_id}] traj {traj_step:7d} | SIM={sim_time_fs/1000.0:8.3f} ps | E={e_now:.6f} eV | F_rms={frms_now:.6f} eV/Å" +
                      (f" | <q>={qmean_now:.6f} e" if qmean_now is not None else ""))

            if LOG_EVERY_STEP and (since_last_write != 0):
                log_runtime(RUNTIME_LOG_PATH, run_id, traj_step, int(TARGET_TIME_FS/DT_FS), segment_id,
                            dt_wall, time.monotonic()-start_wall, dt_curr_fs, sim_time_fs, "step", e_now, frms_now, qmean_now)

        except Exception as err:
            print(f"[EXC][run {run_id}] traj={traj_step}: {err}")
            break

    print(f"[run {run_id}] finished. total_rollbacks={rollback_count}")


def main():
    atomsA = read(INPUT_XYZ, format="extxyz")
    atomsA.set_pbc([True, True, True])

    for run_id in range(1, N_RUNS + 1):
        run_one(run_id, atomsA)

    print("Done.")


if __name__ == "__main__":
    main()

