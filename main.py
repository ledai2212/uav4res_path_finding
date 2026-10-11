
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(ROOT, "simulation-scenarios")

SCENARIOS = {
    "uav4res": os.path.join(SCEN, "UAV4Res.py"),
    "tsunami": os.path.join(SCEN, "tsunami.py"),
    "randomuav": os.path.join(SCEN, "randomUAV.py"),
    "qmix": os.path.join(SCEN, "qmix", "qmix.py"),
    "qmix-test": os.path.join(SCEN, "qmix", "test_demo.py"),
}


def run_script(script_path, extra_args=()):
    return subprocess.Popen(
        [sys.executable, script_path, *extra_args],
        cwd=ROOT
    )


if __name__ == "__main__":
    name = sys.argv[1].lower() if len(sys.argv) > 1 else "uav4res"

    if name not in SCENARIOS:
        sys.exit(
            f"Scenario không hợp lệ: {name!r}. "
            f"Chọn một trong: {', '.join(SCENARIOS)}"
        )

    process = run_script(SCENARIOS[name], sys.argv[2:])
    process.wait()
