#!/usr/bin/env python3
"""
샘플링된 extxyz 파일(멀티 프레임)에 대해
TorchScript U 모델로 모든 프레임의 electrode_potential을 예측하여
새 extxyz에 저장합니다.

- 명령행 인자 없이, 아래 입력 블록만 수정해서 사용
- forces/calculator 없이 그래프 구성 (AtomicData.from_points)
- 모델 출력이 A*U라면 AREA_NORMALIZE=True 로 두어 면적(A)로 나눠 U로 환산
"""
import os
import numpy as np
import torch
from typing import Optional, Sequence
from ase.io import read, write
from estorch.data.AtomicData import AtomicData
from nequip.data import AtomicDataDict

# ============== 사용자 입력 블록 ==============
INPUT_XYZ: str = "./431_EFQ_U_run001.extxyz"   # 샘플링된 extxyz (멀티 프레임)
OUTPUT_XYZ: Optional[str] = "./541-H2O.MLP_U.extxyz"  # None이면 자동 경로 생성
U_MODEL_PATH: str = "../../../deploy_U_w311-cuda.pth" # TorchScript U 모델 경로
DEVICE: str = "cuda"                             # "cuda" or "cpu"

# 모델 출력 키 후보(모델에 맞게 조정)
U_OUT_KEYS = ("electrode_potential", "U", "u", "potential")

# 컷오프: None이면 atoms.calc.r_max → 없으면 FALLBACK_RMAX 사용
U_R_MAX: Optional[float] = None
FALLBACK_RMAX: float = 5.0

# 면적 정규화 및 단위 보정
AREA_NORMALIZE: bool = True       # 예측이 A*U라면 True
AREA_PLANE: str = "auto"          # "auto"|"xy"|"yz"|"zx"
U_SCALE: float = 1.0              # 단위 보정(예: mV->V 1e-3)
U_OFFSET: float = 0.0             # 참조 전극 오프셋 등
# ============================================


def standardize_atoms(atoms):
    """cell을 (3,3)으로 표준화, PBC 보장, float32 캐스팅."""
    cell = np.asarray(atoms.cell.array)
    if cell.ndim == 1:
        if cell.size == 3:
            atoms.set_cell(np.diag(cell))
        elif cell.size == 9:
            atoms.set_cell(cell.reshape(3, 3))
        else:
            raise RuntimeError(f"Invalid cell shape: {cell.shape}")
    elif cell.shape != (3, 3):
        raise RuntimeError(f"Invalid cell shape: {cell.shape}")
    if not np.all(atoms.get_pbc()):
        atoms.set_pbc([True, True, True])
    atoms.set_positions(atoms.get_positions().astype(np.float32))
    atoms.set_cell(np.asarray(atoms.cell.array, dtype=np.float32))
    return atoms


def surface_area_from_cell(cell, pbc=None, plane: str = "auto"):
    """표면 면적(Å^2) 추정. slab 가정에서 세 외적 중 최소를 표면으로 선택."""
    C = np.asarray(cell, float).reshape(3, 3)
    a, b, c = C[0], C[1], C[2]
    areas = {
        "xy": float(np.linalg.norm(np.cross(a, b))),
        "yz": float(np.linalg.norm(np.cross(b, c))),
        "zx": float(np.linalg.norm(np.cross(c, a))),
    }
    if plane != "auto":
        return areas[plane], plane
    if pbc is not None:
        p = np.asarray(pbc, bool)
        if p[0] and p[1] and not p[2]:
            return areas["xy"], "xy"
        if p[1] and p[2] and not p[0]:
            return areas["yz"], "yz"
        if p[2] and p[0] and not p[1]:
            return areas["zx"], "zx"
    k = min(areas, key=areas.get)
    return areas[k], k


def build_atomic_data_from_points(atoms, r_max: float):
    """forces 없이 그래프 생성."""
    Z    = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    pos  = np.asarray(atoms.get_positions(),     dtype=np.float32)
    cell = np.asarray(atoms.cell.array,          dtype=np.float32)
    pbc  = np.asarray(atoms.get_pbc(),           dtype=bool)
    data = AtomicData.from_points(
        pos=pos, atomic_numbers=Z, cell=cell, pbc=pbc, r_max=r_max, add_fields={}
    )
    dd = AtomicData.to_AtomicDataDict(data)
    N = int(Z.shape[0])
    dd["ptr"] = torch.tensor([0, N], dtype=torch.long)  # 단일 그래프 배치 포인터
    return dd


def move_to_device_float32(dd, device: torch.device):
    out = {}
    for k, v in dd.items():
        if torch.is_tensor(v):
            if v.dtype in (torch.float64, torch.float16, torch.bfloat16):
                v = v.to(dtype=torch.float32)
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def parse_u_output(out, key_candidates: Sequence[str]):
    if isinstance(out, dict):
        for k in key_candidates:
            if k in out:
                v = out[k]
                return float(v.detach().reshape(-1)[0].item()) if torch.is_tensor(v) else float(v)
        for v in out.values():
            if torch.is_tensor(v) and v.numel() == 1:
                return float(v.detach().reshape(-1)[0].item())
        raise RuntimeError(f"U keys not found; got: {list(out.keys())}")
    elif torch.is_tensor(out):
        return float(out.detach().reshape(-1)[0].item())
    else:
        return float(out)


def predict_U_once(
    atoms,
    u_model,                              # torch.jit.load(...)로 로드된 모델
    device: torch.device,
    out_keys = U_OUT_KEYS,
    r_max: Optional[float] = None,        # None이면 atoms.calc.r_max -> fallback
    area_normalize: bool = True,
    area_plane: str = "auto",
    scale: float = 1.0,
    offset: float = 0.0,
    fallback_rmax: float = FALLBACK_RMAX,
) -> float:
    """정규화된 U만 반환(원시값·면적은 저장/반환하지 않음)."""
    # r_max 결정: EFQ calc → 지정값 → fallback
    r_m = getattr(getattr(atoms, "calc", None), "r_max", None) or r_max or fallback_rmax

    dd = build_atomic_data_from_points(atoms, r_max=r_m)
    dd = move_to_device_float32(dd, device)

    with torch.no_grad():
        out = u_model(dd)  # dd는 이미 AtomicDataDict 형태
    U = parse_u_output(out, out_keys)

    if area_normalize:
        A, _ = surface_area_from_cell(atoms.cell.array, atoms.get_pbc(), plane=area_plane)
        if A > 1e-8:
            U = U / A
        else:
            print(f"[U] WARNING: tiny area A={A:.3e} Å^2; skip normalization")

    U = scale * U + offset
    return float(U)


def main():
    dev = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={dev}")

    # 입력: 단일/멀티 프레임 모두 허용
    atoms_list = read(INPUT_XYZ, format="extxyz", index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    print(f"[INFO] frames={len(atoms_list)} | input={os.path.abspath(INPUT_XYZ)}")

    # 모델 로드
    u_model = torch.jit.load(U_MODEL_PATH, map_location=dev)
    u_model.eval()
    print(f"[INFO] loaded U model: {os.path.abspath(U_MODEL_PATH)}")

    out_frames = []
    for i, atoms in enumerate(atoms_list, 1):
        atoms = standardize_atoms(atoms)

        try:
            U_val = predict_U_once(
                atoms,
                u_model=u_model,
                device=dev,
                out_keys=U_OUT_KEYS,
                r_max=U_R_MAX,
                area_normalize=AREA_NORMALIZE,
                area_plane=AREA_PLANE,
                scale=U_SCALE,
                offset=U_OFFSET,
            )
        except Exception as e:
            print(f"[WARN] frame {i}: prediction failed: {e}")
            U_val = None

        if U_val is not None:
            atoms.info["electrode_potential"] = float(U_val)
        out_frames.append(atoms)
        print(f"[{i:04d}] U=" + (f"{U_val:.6f}" if U_val is not None else "NaN"))

    out_path = OUTPUT_XYZ or (os.path.splitext(INPUT_XYZ)[0] + ".U.extxyz")
    write(out_path, out_frames, format="extxyz", write_results=False)
    print(f"[SAVE] wrote: {os.path.abspath(out_path)} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()

