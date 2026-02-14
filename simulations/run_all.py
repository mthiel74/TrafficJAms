"""Run all traffic simulations and save results."""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficjams import lwr, payne_whitham, idm, bando, nagel_schreckenberg
from trafficjams import network_assignment, queueing

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

MODELS = {
    "lwr": ("LWR (Lighthill-Whitham-Richards)", lwr),
    "payne_whitham": ("Payne-Whitham", payne_whitham),
    "idm": ("Intelligent Driver Model", idm),
    "bando": ("Bando Optimal Velocity", bando),
    "nasch": ("Nagel-Schreckenberg", nagel_schreckenberg),
    "network": ("Network Assignment", network_assignment),
    "queueing": ("M/D/1 Queueing", queueing),
}


def parse_params(param_strings):
    """Parse 'key=value' strings into a dict with numeric conversion."""
    params = {}
    if not param_strings:
        return params
    for s in param_strings:
        key, val = s.split("=", 1)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        params[key] = val
    return params


def run_model(name, module, params, results_dir):
    print(f"Running {name}...")
    results = module.simulate(**params)
    save_path = os.path.join(results_dir,
                             os.path.basename(module.__file__).replace(".py", ".png"))
    path = module.plot(results, save_path=save_path)
    print(f"  -> Saved {path}")
    return results


def run_all(selected=None, params=None, results_dir=None):
    results_dir = results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    params = params or {}

    for key, (name, module) in MODELS.items():
        if selected and key not in selected:
            continue
        run_model(name, module, params, results_dir)

    print(f"\nSimulations complete. Results saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="TrafficJAms: Run traffic simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m simulations.run_all                    # Run all models\n"
               "  python -m simulations.run_all -m lwr idm         # Run specific models\n"
               "  python -m simulations.run_all -m idm -p n_vehicles=100 T=200\n"
               "  python -m simulations.run_all --list              # List available models\n")
    parser.add_argument("-m", "--models", nargs="+", choices=list(MODELS.keys()),
                        help="Models to run (default: all)")
    parser.add_argument("-p", "--params", nargs="+", metavar="KEY=VALUE",
                        help="Parameters to pass to simulate() (e.g. n_vehicles=100)")
    parser.add_argument("-o", "--output", default=RESULTS_DIR,
                        help="Output directory for results")
    parser.add_argument("--list", action="store_true",
                        help="List available models and exit")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for key, (name, _) in MODELS.items():
            print(f"  {key:20s} {name}")
        return

    params = parse_params(args.params)
    run_all(selected=args.models, params=params, results_dir=args.output)


if __name__ == "__main__":
    main()
