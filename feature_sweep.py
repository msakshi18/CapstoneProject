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
#                  PER MILLISECOND of processing time.

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
            check_leakage=False,  # the leakage check is not needed here; running fl_ids_nbaiot.py separately for that
        ),
        k_values=K_SWEEP_VALUES,
        num_rounds=K_SWEEP_ROUNDS,
        min_acceptable_mcc=MIN_ACCEPTABLE_MCC,
        selection_mode=SELECTION_MODE,
    )

    print(f"\n{'='*60}")
    print(f"  DONE.")
    print(f"  Suggested K_BEST_FEATURES = {suggested_k}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
