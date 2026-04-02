import os
import re
import ase.io
import numpy as np
from ase import Atoms
from typing import List, Tuple, Dict

# ========== INPUT VARIABLES ==========
BASE_DIRECTORY = "./unit/Na_bot/"  # Root directory
OUTPUT_EXTXYZ = "./U/unit-Na_bot.extxyz"
ERROR_LOG_FILE = "error_log.txt"
VACUUM_LEVEL_MODE = "upper"  # Choose from: "avg", "upper", "lower"
CHARGE_TYPE = "bader"   # "bader", "hirshfeld"

# ========== UTILITY FUNCTIONS ==========
def sorted_run_dirs(base_dir: str) -> List[str]:
    """Return a list of run_* directories sorted first by parent path, then run number."""
    run_dirs = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for d in dirs:
            if d.startswith("run_"):
                match = re.search(r"run_(\d+)", d)
                if match:
                    run_number = int(match.group(1))
                    full_path = os.path.join(root, d)
                    run_dirs.append((root, run_number, full_path))
    run_dirs.sort(key=lambda x: (x[0], x[1]))
    return [path for _, _, path in run_dirs]

def parse_acf(file_path: str) -> List[float]:
    """Parse charges from the ACF.dat file."""
    charges = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start = None
        for i, line in enumerate(lines):
            if "-----" in line and start is None:
                start = i + 1
            elif "-----" in line and start is not None:
                end = i
                break
        if start is not None:
            for line in lines[start:end]:
                columns = line.split()
                charges.append(float(columns[4]))  # Charge value is in the 5th column
    return charges

def parse_hirshfeld_block(file_path: str) -> List[float]:
    """Parse Hirshfeld charges from the critic2 output block."""
    charges = []
    inside_block = False
    with open(file_path, 'r') as file:
        for line in file:
            if "* Integrated atomic properties" in line:
                inside_block = True
                continue
            if inside_block:
                if line.strip() == "" or line.startswith("----") or line.startswith("Sum"):
                    break
                if line.strip().startswith("#") or len(line.strip().split()) < 8:
                    continue
                parts = line.strip().split()
                charges.append(float(parts[7]))  # 8번째 column: Pop
    return charges

def parse_outcar_valence(outcar_file: str) -> Dict[str, float]:
    """Extract valence electron (ZVAL) information from the OUTCAR file."""
    valence_dict = {}
    element_list = []
    zval_values = []
    reading_zval = False
    with open(outcar_file, 'r') as file:
        for line in file:
            if "VRHFIN =" in line:
                parts = line.split("=")[1].strip().split(":")
                if len(parts) > 1:
                    element = parts[0].strip()
                    element_list.append(element)
            if reading_zval:
                zval_values = line.replace("ZVAL", "").replace("=", "").strip().split()
                break
            if "Ionic Valenz" in line:
                reading_zval = True
    if len(zval_values) == 0:
        raise ValueError(f"Could not find any ZVAL values in {outcar_file}.")
    if len(element_list) != len(zval_values):
        raise ValueError(f"Mismatch between number of elements ({len(element_list)}) and ZVAL values ({len(zval_values)}) in {outcar_file}.")
    for element, zval in zip(element_list, zval_values):
        valence_dict[element] = float(zval)
    return valence_dict

def parse_outcar_vacuum_and_fermi(outcar_file: str, method: str = "avg") -> Tuple[float, float]:
    """Extract vacuum and Fermi level using the selected method."""
    vacuum_upper, vacuum_lower, fermi_level = None, None, None
    with open(outcar_file, 'r') as file:
        for line in file:
            if "E-fermi" in line:
                fermi_level = float(line.split()[2])
            if "vacuum level on the upper side and lower side of the slab" in line:
                columns = line.split()
                vacuum_upper = float(columns[-2])
                vacuum_lower = float(columns[-1])
    if fermi_level is None:
        raise ValueError("Fermi level not found.")
    if vacuum_upper is None or vacuum_lower is None:
        raise ValueError(f"Could not find vacuum level in {outcar_file}.")
    if method == "upper":
        vacuum_level = vacuum_upper
    elif method == "lower":
        vacuum_level = vacuum_lower
    elif method == "delta":
        vacuum_level = vacuum_lower - vacuum_upper
    else:
        vacuum_level = (vacuum_upper + vacuum_lower) / 2
    return vacuum_level, fermi_level

def adjust_charges(charges: List[float], structure: Atoms, valence_dict: Dict[str, float]) -> List[float]:
    """Adjust charge values by subtracting valence electrons."""
    adjusted = []
    for atom, charge in zip(structure, charges):
        element = atom.symbol
        if element not in valence_dict:
            raise ValueError(f"Missing valence data for {element}.")
        #adjusted.append(charge - valence_dict[element])
        adjusted.append(valence_dict[element] - charge)
    return adjusted

def process_directory(outcar_path: str, charge_path: str, log_file, vacuum_mode: str) -> Atoms:
    """Read one VASP OUTCAR and corresponding charge file, attach charges
    and electrode potential to an ASE Atoms object.

    This is agnostic to the charge type (Bader / Hirshfeld); it just chooses
    the proper parser based on CHARGE_TYPE.
    """
    # 마지막 스냅샷 읽기
    structure = ase.io.read(outcar_path, format="vasp-out", index=-1)

    # OUTCAR/POTCAR 에서 ZVAL 정보 읽기
    valence_dict = parse_outcar_valence(outcar_path)

    # CHARGE_TYPE 에 따라 원시 전자수(population) 읽기
    if CHARGE_TYPE == "bader":
        charges = parse_acf(charge_path)
    elif CHARGE_TYPE == "hirshfeld":
        charges = parse_hirshfeld_block(charge_path)
    else:
        raise ValueError(f"Unknown CHARGE_TYPE: {CHARGE_TYPE}")

    # net charge = ZVAL - population
    adjusted_charges = adjust_charges(charges, structure, valence_dict)
    structure.set_array("charges", np.asarray(adjusted_charges, float))

    # vacuum / Fermi → electrode potential
    vacuum_level, fermi_level = parse_outcar_vacuum_and_fermi(outcar_path, method=vacuum_mode)
    if vacuum_mode == "delta":
        structure.info["electrode_potential"] = vacuum_level
    else:
        structure.info["electrode_potential"] = -1 * (vacuum_level - fermi_level)
    #structure.info["electrode_potential"] = -1.0 * (vacuum_level - fermi_level)

    return structure

def make_clean_atoms(at: Atoms) -> Atoms:
    """Return a minimal Atoms with only safe numeric arrays.

    This avoids ASE/extxyz choking on any list-type or exotic arrays that
    might be attached by the VASP reader or calculators.
    """
    # 기본 기하 정보만 복사
    clean = Atoms(
        numbers=at.get_atomic_numbers(),
        positions=at.get_positions(),
        cell=at.get_cell(),
        pbc=at.get_pbc(),
    )

    # charges 는 반드시 유지
    if "charges" in at.arrays:
        clean.set_array("charges", np.asarray(at.arrays["charges"], float))

    # --- energy / free_energy (info 쪽으로) ---
    energy = None
    free_energy = None
    stress = None
    # 1) calc.results 에서 직접 읽기 (VASP calculator가 붙어 있으면 여기 있음)
    try:
        calc = getattr(at, "calc", None)
        if calc is not None:
            res = getattr(calc, "results", {})
            if "energy" in res:
                energy = float(res["energy"])
            if "free_energy" in res:
                try:
                    free_energy = float(res["free_energy"])
                except Exception:
                    pass
            if "stress" in res:
                try:
                    stress = np.asarray(res["stress"], float)
                except Exception:
                    stress = None
    except Exception:
        pass

    # 2) energy 가 아직 없으면 get_potential_energy() 시도
    if energy is None:
        try:
            energy = float(at.get_potential_energy())
        except Exception:
            # 그래도 없으면 at.info["energy"] 를 마지막으로 시도
            if "energy" in at.info:
                try:
                    energy = float(at.info["energy"])
                except Exception:
                    energy = None

    # 3) free_energy 가 여전히 None 이고, info 에 있으면 거기서 가져오기
    if free_energy is None and "free_energy" in at.info:
        try:
            free_energy = float(at.info["free_energy"])
        except Exception:
            free_energy = None

    # 4) stress 가 없으면 get_stress() 또는 info["stress"] 시도
    if stress is None:
        # get_stress() 시도
        try:
            s = at.get_stress()
            stress = np.asarray(s, float)
        except Exception:
            # info 에 저장된 stress 가 있다면 그대로 사용
            if "stress" in at.info:
                try:
                    stress = np.asarray(at.info["stress"], float)
                except Exception:
                    stress = None

    # ---------- forces ----------
    forces = None
    # 1) calc 가 있으면 get_forces() 시도
    try:
        forces = at.get_forces()
    except Exception:
        # 2) arrays 에 forces 가 있으면 거기서
        if "forces" in at.arrays:
            try:
                forces = np.asarray(at.arrays["forces"], float)
            except Exception:
                forces = None

    if forces is not None:
        clean.set_array("forces", forces)

    # ---------- info 에 값들 저장 ----------
    if energy is not None:
        clean.info["energy"] = energy

    if free_energy is not None:
        clean.info["free_energy"] = free_energy

    # stress는 list로 저장해 두면 extxyz info 필드로 잘 나감
    if stress is not None:
        s_arr = np.asarray(stress, float).reshape(-1)  # 1D array
        # 필요하면 3x3 → Voigt(6) 등 후처리도 여기서 가능
        clean.info["stress"] = s_arr

    # electrode_potential (우리가 process_directory 에서 넣은 값)
    if "electrode_potential" in at.info:
        try:
            clean.info["electrode_potential"] = float(at.info["electrode_potential"])
        except Exception:
            pass

    # --- electrode_potential (우리가 process_directory에서 넣은 값) ---
    if "electrode_potential" in at.info:
        try:
            clean.info["electrode_potential"] = float(at.info["electrode_potential"])
        except Exception:
            pass

    # --- 그 외 간단한 스칼라 info도 필요하면 같이 복사 ---
    for key, val in at.info.items():
        if key in clean.info:
            continue  # 이미 넣은 energy/free_energy/electrode_potential 등은 건너뜀
        if isinstance(val, (int, float, np.number, str)):
            clean.info[key] = val

    return clean

# ========== MAIN ROUTINE ==========
def collect_data_to_extxyz(base_dir: str, output_file: str, log_file_path: str, vacuum_mode: str):
    atoms_list: List[Atoms] = []
    error_count = 0
    with open(log_file_path, "w") as log_file:
        log_file.write("Error Log:\n")
        log_file.write("=" * 40 + "\n")
        run_dirs = sorted_run_dirs(base_dir)
        for run_dir in run_dirs:
            outcar = os.path.join(run_dir, "OUTCAR")

            if CHARGE_TYPE == "bader":
                charge_file = os.path.join(run_dir, "ACF.dat")
            elif CHARGE_TYPE == "hirshfeld":
                charge_file = os.path.join(run_dir, "hirshfeld_block.dat")
            else:
                raise ValueError("Invalid CHARGE_TYPE")

            if not os.path.exists(outcar):
                log_file.write(f"Missing OUTCAR in {run_dir}\n")
                error_count += 1
                continue

            if not os.path.exists(charge_file):
                log_file.write(f"Missing charge file ({CHARGE_TYPE}) in {run_dir}\n")
                error_count += 1
                continue

            try:
                # 1) OUTCAR + charge 파일을 읽어서 charges / electrode_potential 부착
                atoms = process_directory(outcar, charge_file, log_file, vacuum_mode)
                # 2) extxyz 에 안전하게 쓸 수 있는 최소한의 Atoms 로 정리
                clean_atoms = make_clean_atoms(atoms)
                atoms_list.append(clean_atoms)
            except Exception as e:
                log_file.write(f"Skipped {run_dir}: {e}\n")
                error_count += 1

    if atoms_list:
        # list-type array는 clean_atoms 에서 제거했으므로, 여기서 더 이상
        # 'list' object has no attribute 'dtype' 에러가 나면 안 된다.
        ase.io.write(output_file, atoms_list, format="extxyz")
        print(f"✅ Combined extxyz written: {output_file}")
    else:
        print("[!] No valid structures were processed.")

    print(f"[!] {error_count} error(s) logged. See {log_file_path}")

# Run it
collect_data_to_extxyz(BASE_DIRECTORY, OUTPUT_EXTXYZ, ERROR_LOG_FILE, VACUUM_LEVEL_MODE)
