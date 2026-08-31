"""Generate the DAIS convergence sweep from the unchanged base templates."""
"""
& "C:\Program Files\Git\usr\bin\scp.exe" -r -o "ProxyJump=kshao@gate1.mpcdf.mpg.de" "kshao@dais11:/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais/daisconvergencesweep/." "/c/Users/shaoq/Documents/Mainz/mlip/outputsfull/daisconvergencesweep/"

"""
from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
INPUT_TEMPLATE = BASE_DIR / "unforced_base.in"
SBATCH_TEMPLATE = BASE_DIR / "unforced_base.sh"

VELOCITY_CM_S = 3.0e7
DEFAULT_NH = 100_000
DEFAULT_NSI = 1_000

# Cases are the c0 and c1 entries from ../expand.py.
COUPLING_CASES = {
    "c0": {"T": 5_000.0, "ZH": 0.16, "ZSi": 0.26, "rhoH": 1.0e-5, "rhoS": 1.79e-4},
    "c1": {"T": 5_000.0, "ZH": 0.65, "ZSi": 3.82, "rhoH": 1.0, "rhoS": 7.45},
}

SWEEPS = {
    "cutoff": [(f"{factor}lS", DEFAULT_NH, DEFAULT_NSI, factor) for factor in (1, 2, 3, 4)],
    "box": [(f"NSi{nsi}", 100 * nsi, nsi, 2) for nsi in (100, 300, 1_000, 3_000)],
    "ratio": [(f"NSi{nsi}", DEFAULT_NH, nsi, 2) for nsi in (30, 100, 300, 1_000)],
}

# SI constants; hbar is the reduced Planck constant, not Planck's constant.
NA = 6.022e23
PI = math.pi
ME = 9.1093837e-31
MH = 1.008e-3 / NA
MSI = 28.09e-3 / NA
QE = 1.60217663e-19
HBAR = 1.054571817e-34
KB = 1.38e-23
EPS0 = 8.8541878188e-12


def replace_once(text: str, old: str, new: str, source: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r} in {source}, found {count}")
    return text.replace(old, new)


def screening_length(case: dict[str, float]) -> float:
    """Return lS in metres using the standard single-species electron Debye length."""
    ne = 1.0e6 * NA * case["ZH"] * case["rhoH"] / 1.008
    # lD = sqrt(epsilon_0 k_B T / (n_e e^2)); there is no erroneous sqrt(3).
    ld = math.sqrt(EPS0 * KB * case["T"] / (ne * QE**2))
    tf = (3.0 * ne * PI**2) ** (2.0 / 3.0) * HBAR**2 / (2.0 * ME * KB)
    return ld * (1.0 + 2.0 * tf / (3.0 * case["T"])) ** 0.25


def lambertw_positive(x: float) -> float:
    """Return the principal real Lambert W for x > 0 using Halley's method."""
    if x <= 0.0:
        raise ValueError("This sweep requires a positive Lambert-W argument")
    w = x if x < 1.0 else math.log(x) - math.log(math.log(x)) if x > math.e else math.log(x)
    for _ in range(100):
        ew = math.exp(w)
        residual = w * ew - x
        denominator = ew * (w + 1.0) - (w + 2.0) * residual / (2.0 * w + 2.0)
        update = residual / denominator
        w -= update
        if abs(update) <= 1.0e-14 * max(1.0, abs(w)):
            if not math.isclose(w * math.exp(w), x, rel_tol=1.0e-13, abs_tol=0.0):
                raise RuntimeError("Lambert-W residual check failed")
            return w
    raise RuntimeError("Lambert-W iteration did not converge")


def collision_cross_section_cm2(case: dict[str, float]) -> float:
    """Match the existing DAIS Yukawa turning-point calculation at 3e7 cm/s."""
    tsh = math.sqrt(3.0 * KB * case["T"] / MH)
    tssi = math.sqrt(3.0 * KB * case["T"] / MSI)
    vrel = 0.01 * VELOCITY_CM_S - tsh - tssi
    vcom = MH * vrel / (MH + MSI)
    available_energy = 0.5 * MH * (vrel - vcom) ** 2 + 0.5 * MSI * vcom**2
    coulomb_prefactor = QE**2 * case["ZH"] * case["ZSi"] / (4.0 * PI * EPS0)
    kappa = 1.0 / screening_length(case)
    turning_radius = lambertw_positive(kappa / (available_energy / coulomb_prefactor)) / kappa
    return 1.0e4 * PI * turning_radius**2


def make_input(template: str, name: str, case: dict[str, float], nh: int, nsi: int, cutoff: int) -> str:
    replacements = {
        "variable vb equal 1.0e7": f"variable vb equal {VELOCITY_CM_S:.1e}",
        "variable T0 equal 100000": f"variable T0 equal {case['T']:.5e}",
        "variable rhoH equal 1.0e-5 #gcc": f"variable rhoH equal {case['rhoH']:.5e} #gcc",
        "variable rhoS equal 1.79e-4 #gcc": f"variable rhoS equal {case['rhoS']:.5e} #gcc",
        "variable ZH equal 0.955": f"variable ZH equal {case['ZH']:.5e}",
        "variable ZSi equal 4.91": f"variable ZSi equal {case['ZSi']:.5e}",
        "variable ccs equal 1e-6 #collision cross section in cm^2": f"variable ccs equal {collision_cross_section_cm2(case):.5e} #collision cross section in cm^2",
        "variable NH equal 100000": f"variable NH equal {nh}",
        "variable NSi equal 300": f"variable NSi equal {nsi}",
        "variable ve equal $(sqrt(v_kb*v_T0/v_me)) #1D electron thermal speed for standard Debye length": "variable ve equal $(sqrt(v_kb*v_T0/v_me)) #1D electron thermal speed for standard Debye length",
        "variable lD equal $(v_ve/v_wp) #sqrt(e0*kB*T/(ne*qe^2)), SI metres": "variable lD equal $(v_ve/v_wp) #sqrt(e0*kB*T/(ne*qe^2)), SI metres",
        "variable lS equal $(v_lD*((1+(2*v_TF/(3*v_T0)))^(1/4)))": "variable lS equal $(v_lD*((1+(2*v_TF/(3*v_T0)))^(1/4)))",
        "variable rc_cm equal $(300*v_lS) #cutoff is in centimeters, three times screening length": f"variable rc_cm equal $({100 * cutoff}*v_lS) #cutoff is {cutoff} times screening length, in cm",
        "dump mydmp Si custom 100 traj.txt id type vx vy vz": f"dump mydmp Si custom 100 traj_{name}.txt id type vx vy vz",
        "log firstlog": f"log {name}.log",
    }
    result = template
    for old, new in replacements.items():
        result = replace_once(result, old, new, INPUT_TEMPLATE)
    return result


def make_sbatch(template: str, name: str) -> str:
    replacements = {
        "#SBATCH --job-name=unforced_base": f"#SBATCH --job-name={name}",
        "#SBATCH --chdir=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais": "#SBATCH --chdir=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais/daisconvergencesweep",
        "#SBATCH --output=unforced_base_%j.out": f"#SBATCH --output={name}_%j.out",
        "#SBATCH --error=unforced_base_%j.err": f"#SBATCH --error={name}_%j.err",
        "INPUT=unforced_base.in": f"INPUT={name}.in",
        'LMP_LOG="unforced_base_${SLURM_JOB_ID}.lammps.log"': f'LMP_LOG="{name}_${{SLURM_JOB_ID}}.lammps.log"',
    }
    result = template
    for old, new in replacements.items():
        result = replace_once(result, old, new, SBATCH_TEMPLATE)
    return result


def main() -> None:
    input_template = INPUT_TEMPLATE.read_text(encoding="utf-8")
    sbatch_template = SBATCH_TEMPLATE.read_text(encoding="utf-8")
    expected: set[Path] = set()

    for coupling, case in COUPLING_CASES.items():
        for family, configurations in SWEEPS.items():
            for value_label, nh, nsi, cutoff in configurations:
                name = f"dais_{family}_{value_label}_{coupling}"
                input_path = HERE / f"{name}.in"
                sbatch_path = HERE / f"{name}.sh"
                input_path.write_text(make_input(input_template, name, case, nh, nsi, cutoff), encoding="utf-8", newline="\n")
                sbatch_path.write_text(make_sbatch(sbatch_template, name), encoding="utf-8", newline="\n")
                expected.update((input_path, sbatch_path))

    actual = set(HERE.glob("dais_*.in")) | set(HERE.glob("dais_*.sh"))
    if actual != expected:
        unexpected = sorted(path.name for path in actual - expected)
        missing = sorted(path.name for path in expected - actual)
        raise RuntimeError(f"Generated-file mismatch; unexpected={unexpected}, missing={missing}")
    if len(list(HERE.glob("*.in"))) != 24 or len(list(HERE.glob("*.sh"))) != 24:
        raise RuntimeError("Expected exactly 24 .in and 24 .sh files")

    print("Generated and validated 24 .in and 24 .sh files.")


if __name__ == "__main__":
    main()
