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
K_SWEEP_VALUES = [50,60,65,70,75]

K_SWEEP_ROUNDS = 5

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
