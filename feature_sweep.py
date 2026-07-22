"""
feature_sweep.py

A separate, standalone script that helps you pick K_BEST_FEATURES (how
many of the 115 N-BaIoT features to keep) for fl_ids_nbaiot.py.

WHY THIS IS A SEPARATE FILE:
Testing several different feature counts means running federated training
several times in a row, which takes a while. Keeping it separate means
your normal, everyday runs of fl_ids_nbaiot.py stay fast — you only run
this file when you actually want to (re)check what K_BEST_FEATURES should
be, not every single time.

HOW TO USE THIS FILE:
1. Make sure the N-BaIoT dataset folder is in the same folder as this
   script (same as fl_ids_nbaiot.py needs it).
2. Run:  python feature_sweep.py
3. Wait for it to finish (it trains several small models, one per feature
   count in K_SWEEP_VALUES below — this can take a few minutes).
4. Read the printed table. It shows accuracy / MCC / latency for each
   feature count tested.
5. At the end it prints a suggested K_BEST_FEATURES value. Look at the
   table yourself before trusting it — then go update K_BEST_FEATURES in
   fl_ids_nbaiot.py's main() function to match your decision.

This script imports everything it needs from fl_ids_nbaiot.py — no code
is copied or duplicated between the two files.
"""

from fl_ids_nbaiot import (
    run_feature_count_sweep,
    load_nbaiot_federated,
    load_xiiotid_federated,
    DATASET_ROOT,
    XIIOTID_CSV_PATH,
)

# ------------------------------------------------------------------
# Sweep configuration — adjust as needed
# ------------------------------------------------------------------

# Which feature counts to test. Only include numbers you'd actually be
# comfortable using in your final report — the script will only ever
# recommend a number from this list, it won't suggest anything outside it.
K_SWEEP_VALUES = [50,60,65,70,75]

# How many federated learning "rounds" to use for each test. Lower than
# your main training run's round count on purpose — this script already
# runs one full training PER value in K_SWEEP_VALUES above, so keeping
# each one shorter keeps the total time reasonable.
K_SWEEP_ROUNDS = 5

# HOW TO PICK THE "BEST" FEATURE COUNT — pick one of these three:
#
#   "efficiency" — picks the feature count that gives the best performance
#                  PER MILLISECOND of processing time. Recommended if you
#                  care about keeping the model fast/lightweight.
#
#   "best"       — picks whichever feature count scores the highest (MCC),
#                  ignoring speed completely. Note: this will usually pick
#                  the LARGEST number in K_SWEEP_VALUES, since more
#                  features almost always helps accuracy a little, even if
#                  it makes the model slower.
#
#   "floor"      — picks the SMALLEST feature count that is "good enough"
#                  (scores at least MIN_ACCEPTABLE_MCC below), even if a
#                  bigger number would score higher.
SELECTION_MODE = "efficiency"

# Only used if SELECTION_MODE = "floor" above.
# MCC ranges from -1 (worst) to +1 (perfect), with 0 meaning "random
# guessing". This is NOT the same scale as accuracy or F1-score.
MIN_ACCEPTABLE_MCC = 0.7


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"N-BaIoT dataset not found at '{DATASET_ROOT}'. "
            "Run this script from the same folder as your dataset."
        )

    print(f"Testing feature counts {K_SWEEP_VALUES} on N-BaIoT ({DATASET_ROOT})...")
    df, suggested_k = run_feature_count_sweep(
        loader_fn=load_nbaiot_federated,
        loader_kwargs=dict(
            dataset_root=DATASET_ROOT,
            max_samples_per_device=5000,
            check_leakage=False,  # the leakage check is not needed here; run fl_ids_nbaiot.py separately for that
        ),
        k_values=K_SWEEP_VALUES,
        num_rounds=K_SWEEP_ROUNDS,
        min_acceptable_mcc=MIN_ACCEPTABLE_MCC,
        selection_mode=SELECTION_MODE,
    )

    print(f"\n{'='*60}")
    print(f"  DONE.")
    print(f"  Suggested K_BEST_FEATURES = {suggested_k}")
    print(f"  Now go update that number in fl_ids_nbaiot.py's main() function.")
    print(f"  (Read the table above first — don't just trust this number blindly.)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
