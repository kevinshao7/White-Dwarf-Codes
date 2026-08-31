"""Generate DAIS production unforced runs.

This creates 40 runs total:
- 4 coupling conditions, c0 through c3
- 10 initial velocities, log-spaced from 1e5 to 1e8 cm/s

Weakly coupled conditions c0 and c2 get 12 hour GPU wall time.
Strongly coupled conditions c1 and c3 get 4 hour GPU wall time.

Run this script from this directory or from anywhere:

python expand.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
INPUT_TEMPLATE = BASE_DIR / "unforced_base.in"
SBATCH_TEMPLATE = BASE_DIR / "unforced_base.sh"

VELOCITIES_CM_S = np.logspace(5.0, 8.0, 10)
DEFAULT_NH = 100_000
DEFAULT_NSI = 300
DEFAULT_CUTOFF_LS = 3

COUPLING_CASES = {
    0: {"T": 5_000.0, "ZH": 0.16, "ZSi": 0.26, "rhoH": 1.0e-5, "rhoS": 1.79e-4},
    1: {"T": 5_000.0, "ZH": 0.65, "ZSi": 3.82, "rhoH": 1.0, "rhoS": 7.45},
    2: {"T": 100_000.0, "ZH": 0.93, "ZSi": 4.27, "rhoH": 1.0e-5, "rhoS": 6.2e-5},
    3: {"T": 100_000.0, "ZH": 0.68, "ZSi": 3.81, "rhoH": 1.0, "rhoS": 7.2},
}
WEAK_CONDITIONS = {0, 2}
STRONG_CONDITIONS = {1, 3}

NA = 6.022e23
PI = math.pi
ME = 9.1093837e-31
MH = 1.008e-3 / NA
MSI = 28.09e-3 / NA
QE = 1.60217663e-19
HBAR = 1.054571817e-34
KB = 1.38e-23
EPS0 = 8.8541878188e-12


def replace_line(text: str, prefix: str, replacement: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}.*$", re.MULTILINE)
    result, count = pattern.subn(replacement, text)
    if count != 1:
        raise RuntimeError(f"Expected one line beginning with {prefix!r}, found {count}")
    return result


def replace_exact(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new)


def screening_length_m(case: dict[str, float]) -> float:
    """Return lS in metres with the corrected Debye speed and TF factor."""
    ne = 1.0e6 * NA * case["ZH"] * case["rhoH"] / 1.008
    ve = math.sqrt(KB * case["T"] / ME)
    wp = math.sqrt(ne * QE**2 / (ME * EPS0))
    debye = ve / wp
    tf = (3.0 * ne * PI**2) ** (2.0 / 3.0) * HBAR**2 / (2.0 * ME * KB)
    return debye * (1.0 + 2.0 * tf / (3.0 * case["T"])) ** 0.25


def lambertw_positive(x: float) -> float:
    if x <= 0.0:
        raise ValueError("Lambert-W argument must be positive")
    w = x if x < 1.0 else math.log(x) - math.log(math.log(x)) if x > math.e else math.log(x)
    for _ in range(100):
        ew = math.exp(w)
        residual = w * ew - x
        denominator = ew * (w + 1.0) - (w + 2.0) * residual / (2.0 * w + 2.0)
        update = residual / denominator
        w -= update
        if abs(update) <= 1.0e-14 * max(1.0, abs(w)):
            return w
    raise RuntimeError("Lambert-W iteration did not converge")


def collision_cross_section_cm2(velocity_cm_s: float, case: dict[str, float]) -> float:
    """Yukawa turning-point cross section in cm^2 using corrected lS."""
    thermal_h = math.sqrt(3.0 * KB * case["T"] / MH)
    thermal_si = math.sqrt(3.0 * KB * case["T"] / MSI)
    # The production LAMMPS input uses vthh + vths + vb as the collision-speed
    # scale.  Use the same positive relative speed here; subtracting thermal
    # speeds made the lowest-velocity cases emit an unphysical zero cross section.
    vrel = 0.01 * velocity_cm_s + thermal_h + thermal_si
    vcom = MH * vrel / (MH + MSI)
    available_energy = 0.5 * MH * (vrel - vcom) ** 2 + 0.5 * MSI * vcom**2
    coulomb_prefactor = QE**2 * case["ZH"] * case["ZSi"] / (4.0 * PI * EPS0)
    kappa = 1.0 / screening_length_m(case)
    turning_radius = lambertw_positive(kappa / (available_energy / coulomb_prefactor)) / kappa
    return 1.0e4 * PI * turning_radius**2


def make_input(template: str, name: str, velocity_cm_s: float, condition: int, case: dict[str, float]) -> str:
    result = template
    result = replace_line(result, "variable vb equal ", f"variable vb equal {velocity_cm_s:.6e}")
    result = replace_line(result, "variable T0 equal ", f"variable T0 equal {case['T']:.6e}")
    result = replace_line(result, "variable rhoH equal ", f"variable rhoH equal {case['rhoH']:.6e} #gcc")
    result = replace_line(result, "variable rhoS equal ", f"variable rhoS equal {case['rhoS']:.6e} #gcc")
    result = replace_line(result, "variable ZH equal ", f"variable ZH equal {case['ZH']:.6e}")
    result = replace_line(result, "variable ZSi equal ", f"variable ZSi equal {case['ZSi']:.6e}")
    result = replace_line(
        result,
        "variable ccs equal ",
        f"variable ccs equal {collision_cross_section_cm2(velocity_cm_s, case):.6e} #collision cross section in cm^2",
    )
    result = replace_line(result, "variable NH equal ", f"variable NH equal {DEFAULT_NH}")
    result = replace_line(result, "variable NSi equal ", f"variable NSi equal {DEFAULT_NSI}")
    result = replace_line(
        result,
        "variable ve equal ",
        "variable ve equal $(sqrt(v_kb*v_T0/v_me)) #1D electron thermal speed for standard Debye length",
    )
    result = replace_line(
        result,
        "variable lD equal ",
        "variable lD equal $(v_ve/v_wp) #sqrt(e0*kB*T/(ne*qe^2)), SI metres",
    )
    result = replace_line(result, "variable lS equal ", "variable lS equal $(v_lD*((1+(2*v_TF/(3*v_T0)))^(1/4)))")
    result = replace_line(result, "variable k0 equal ", "variable k0 equal $(1/(100*v_lS)) #inverse screening length in cm^-1")
    result = replace_line(
        result,
        "variable rc_cm equal ",
        f"variable rc_cm equal $({100 * DEFAULT_CUTOFF_LS}*v_lS) #cutoff is {DEFAULT_CUTOFF_LS} times screening length, in cm",
    )
    result = replace_exact(result, "dump mydmp Si custom 100 traj.txt id type vx vy vz", f"dump mydmp Si custom 100 traj_{name}.txt id type vx vy vz")
    result = replace_exact(result, "log firstlog", f"log {name}.log")
    return result


def wall_time(condition: int) -> str:
    if condition in WEAK_CONDITIONS:
        return "12:00:00"
    if condition in STRONG_CONDITIONS:
        return "04:00:00"
    raise ValueError(f"unexpected condition {condition}")


def make_sbatch(template: str, name: str, condition: int) -> str:
    result = template
    result = replace_line(result, "#SBATCH --job-name=", f"#SBATCH --job-name={name}")
    result = replace_line(result, "#SBATCH --time=", f"#SBATCH --time={wall_time(condition)}")
    result = replace_line(
        result,
        "#SBATCH --chdir=",
        "#SBATCH --chdir=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/daisslurm/daisproduction",
    )
    result = replace_line(result, "#SBATCH --output=", f"#SBATCH --output={name}_%j.out")
    result = replace_line(result, "#SBATCH --error=", f"#SBATCH --error={name}_%j.err")
    result = replace_exact(result, "INPUT=unforced_base.in", f"INPUT={name}.in")
    result = replace_exact(result, 'LMP_LOG="unforced_base_${SLURM_JOB_ID}.lammps.log"', f'LMP_LOG="{name}_${{SLURM_JOB_ID}}.lammps.log"')
    return result


def main() -> None:
    input_template = INPUT_TEMPLATE.read_text(encoding="utf-8")
    sbatch_template = SBATCH_TEMPLATE.read_text(encoding="utf-8")
    expected: set[Path] = set()

    for condition, case in COUPLING_CASES.items():
        for velocity in VELOCITIES_CM_S:
            name = f"unforcedprod_v{velocity:.1e}_c{condition}"
            input_path = HERE / f"{name}.in"
            sbatch_path = HERE / f"{name}.sh"
            input_path.write_text(make_input(input_template, name, float(velocity), condition, case), encoding="utf-8", newline="\n")
            sbatch_path.write_text(make_sbatch(sbatch_template, name, condition), encoding="utf-8", newline="\n")
            expected.update((input_path, sbatch_path))

    actual = set(HERE.glob("unforcedprod_*.in")) | set(HERE.glob("unforcedprod_*.sh"))
    if actual != expected:
        unexpected = sorted(path.name for path in actual - expected)
        missing = sorted(path.name for path in expected - actual)
        raise RuntimeError(f"Generated-file mismatch; unexpected={unexpected}, missing={missing}")
    if len(list(HERE.glob("unforcedprod_*.in"))) != 40 or len(list(HERE.glob("unforcedprod_*.sh"))) != 40:
        raise RuntimeError("Expected exactly 40 production .in files and 40 production .sh files")
    print("Generated and validated 40 DAIS production inputs and 40 submit scripts.")


if __name__ == "__main__":
    main()
