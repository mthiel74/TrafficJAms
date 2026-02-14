"""Run all traffic simulations and save results."""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficjams import lwr, payne_whitham, idm, bando, nagel_schreckenberg
from trafficjams import network_assignment, queueing

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = [
        ("LWR (Lighthill-Whitham-Richards)", lwr),
        ("Payne-Whitham", payne_whitham),
        ("Intelligent Driver Model", idm),
        ("Bando Optimal Velocity", bando),
        ("Nagel-Schreckenberg", nagel_schreckenberg),
        ("Network Assignment", network_assignment),
        ("M/D/1 Queueing", queueing),
    ]

    for name, module in models:
        print(f"Running {name}...")
        results = module.simulate()
        path = module.plot(results, save_path=os.path.join(RESULTS_DIR,
                           os.path.basename(module.__file__).replace(".py", ".png")))
        print(f"  -> Saved {path}")

    print(f"\nAll simulations complete. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_all()
