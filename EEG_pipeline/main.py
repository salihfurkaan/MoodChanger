"""
main.py
Interactive EEG Pipeline demo.

Run:
    python main.py                    # default: relaxed state, 10 s
    python main.py --state anxious    # specific state
    python main.py --tour             # cycle through all states
    python main.py --help
"""

import argparse
import sys
from pipeline import EEGPipeline


STATES = ["relaxed", "focused", "anxious", "drowsy", "meditative"]

BANNER = """
╔══════════════════════════════════════════════════════╗
║             EEG Analysis Pipeline                    ║
║                                                      ║
║   Muse-compatible · Fake data · Real DSP             ║
╚══════════════════════════════════════════════════════╝
"""


def run_single(state: str, duration: float, quiet: bool, realtime: bool):
    print(BANNER)
    pipeline = EEGPipeline(state=state, verbose=not quiet)
    pipeline.run(duration_sec=duration, realtime=realtime, print_every=4)


def run_tour(duration_per_state: float = 5.0):
    """Cycle through every mental state so you can see the classifier in action."""
    print(BANNER)
    print("🗺️  TOUR MODE — cycling through all states\n")
    for state in STATES:
        pipeline = EEGPipeline(state=state, verbose=False)
        pipeline.run(duration_sec=duration_per_state, print_every=999)
        last = pipeline.last_result
        if last:
            print(f"  ✓ {state:12s} → predicted: {last.label:12s}  "
                  f"conf={last.confidence:.0%}\n")


def interactive_menu():
    """Simple text menu for exploring states."""
    print(BANNER)
    print("Choose a mental state to simulate:\n")
    for i, s in enumerate(STATES, 1):
        print(f"  {i}. {s}")
    print("  6. Tour (all states)")
    print("  0. Exit\n")

    choice = input("Enter choice [1-6]: ").strip()
    if choice == "0":
        sys.exit(0)
    elif choice == "6":
        run_tour()
    elif choice in [str(i) for i in range(1, 6)]:
        state = STATES[int(choice) - 1]
        dur   = input("Duration in seconds [10]: ").strip() or "10"
        try:
            dur = float(dur)
        except ValueError:
            dur = 10.0
        pipeline = EEGPipeline(state=state, verbose=True)
        pipeline.run(duration_sec=dur, print_every=4)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Analysis Pipeline – fake Muse data → mental state"
    )
    parser.add_argument(
        "--state", choices=STATES, default=None,
        help="Mental state to simulate (skips interactive menu)"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Simulated duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--tour", action="store_true",
        help="Cycle through all states automatically"
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Pace output to wall-clock time (adds delays)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )
    args = parser.parse_args()

    if args.tour:
        run_tour(duration_per_state=args.duration)
    elif args.state:
        run_single(args.state, args.duration, args.quiet, args.realtime)
    else:
        interactive_menu()