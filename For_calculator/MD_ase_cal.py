import time
import torch
import numpy as np
import gc
import psutil
from ase import units
from ase import Atoms
from ase.md.langevin import Langevin
from ase.io import read
from nequip.dynamics.nequip_calculator import NequIPCalculator
from nequip.data import AtomicDataDict
from torch.cuda.amp import autocast

# === User-defined inputs ===
device = 'cuda'  # Set to 'cuda' to use GPU if available
deployed_model_path = "./deploy_model_model2.pth"  # Path to the NequIP model file
input_structure_path = '../../supercell/temp566.extxyz'  # Path to the structure file for the simulation
output_file_path = "md_566_results.txt"
log_file_path = "md_566.log"  # Path to the output log file

# MD simulation parameters
temperature = 300  # in Kelvin
time_step = 1 * units.fs  # Time interval in femtoseconds
total_steps = 5   # Total number of simulation steps

species_to_type_name = {"Sr": "Sr", "Ti": "Ti", "O": "O"}  # Element and model name mapping
# ===========================

# Function to log memory usage
def log_memory_usage(label):
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 ** 3)  # Resident Set Size (RSS) in GB
    gpu_memory = f"{torch.cuda.memory_allocated() / (1024 ** 3):.4f} GB allocated, {torch.cuda.memory_reserved() / (1024 ** 3):.4f} GB reserved" if torch.cuda.is_available() and device == 'cuda' else "N/A"
    print(f"{label} - CPU Memory: {cpu_memory:.4f} GB, GPU Memory: {gpu_memory}")

# Set up the calculator model
calculator = NequIPCalculator.from_deployed_model(
    model_path=deployed_model_path,
    species_to_type_name=species_to_type_name,
    device=device
)

# Create ASE Atoms object and assign the calculator
atom_pos = read(input_structure_path, format='extxyz')
atom_pos.calc = calculator

# Set up MD simulation
dyn = Langevin(atom_pos, time_step, temperature_K=temperature * units.kB, friction=0.02)

# Start the overall timer
overall_start_time = time.time()

# MD simulation loop
with open(output_file_path, "w") as output_file, open(log_file_path, "w") as log_file:
    for step in range(total_steps):
        log_memory_usage(f"Before Step {step + 1}")
        start_time = time.time()
        dyn.run(1)  # Execute 1 step
        log_memory_usage(f"After MD Step {step + 1}")

        # Calculate energy, forces, stress, and charges
        with autocast():
            energy = atom_pos.get_potential_energy()
            log_memory_usage(f"After Energy Calculation {step + 1}")
            forces = atom_pos.get_forces()
            log_memory_usage(f"After Forces Calculation {step + 1}")
            stress = atom_pos.get_stress() * -1602.1766208  # Convert to kBar
            log_memory_usage(f"After Stress Calculation {step + 1}")

        # Reshape stress to a single line for the log
        stress_formatted = ' '.join(f"{s:.6f}" for s in stress)

        charges_key = getattr(calculator, "charges_key", "charges")  # Default to "charges"
        charges = calculator.results.get(charges_key, None)
        log_memory_usage(f"After Charges Calculation {step + 1}")
        # Calculate total charge
        total_charge = np.sum(charges) if charges is not None else None
        log_memory_usage(f"After Total Charge Calculation {step + 1}")

        # Step completion time
        total_time = time.time() - start_time

        # Write results to outfile
        output_file.write(f"Step {step + 1}:\n")
        output_file.write(f"Energy = {energy:.4f} eV\n")
        output_file.write("Forces = \n")
        output_file.write(f"{forces}\n")
        output_file.write("Stress (kBar) =\n")
        output_file.write(f"[{stress_formatted}]\n")
        if charges is not None:
            output_file.write("Charges = \n")
            output_file.write(f"{charges}\n")
            output_file.write(f"Total Charge = {total_charge}\n")
        output_file.write("\n")
        
        # Write the step time to the log file
        log_file.write(f"Step {step + 1} calculation time (sec): {total_time:.4f}\n")

        # Clean up variables to free memory
        del energy, forces, stress, charges, total_charge
        gc.collect()  # Force garbage collection to free CPU memory
        
        # Clear CUDA cache only if GPU is being used
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()    
            
        # Console output (optional)
        print(f"Step {step + 1} complete. Results saved to {log_file_path}")

    # Calculate total time for all steps
    overall_total_time = time.time() - overall_start_time
    log_file.write(f"Total calculation time (sec): {overall_total_time:.4f}\n")

print("Simulation complete. Results saved to", output_file_path)
print("Simulation complete. Total results saved to", log_file_path)
