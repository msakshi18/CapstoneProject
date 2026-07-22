"""
Federated Learning Intrusion Detection System — N-BaIoT Dataset
================================================================
Implements FedAvg (baseline) and FedProx (Non-IID optimisation)
as outlined in: "Optimising Federated Learning Based Intrusion Detection
for IoT Systems" — Sakshi Mahajan, MSc CS (Data Analytics), Uni. of Galway

"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.feature_selection import SelectKBest, chi2
from scipy.spatial import cKDTree
from pathlib import Path

# DATASET LOADING — N-BaIoT
DATASET_ROOT = Path("N-BaIoT")

# Known device folder names in the N-BaIoT dataset
DEVICE_FOLDERS = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]


def _split_device(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    split_strategy: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits one device's data into train/test.

    split_strategy:
      "random" — the original behaviour: sklearn's stratified random split.
                 If the underlying CSV rows are temporally adjacent capture
                 windows, a random split can place near-identical rows from
                 the same attack burst on both sides — this is the N-BaIoT
                 leakage pattern.
      "time"   — takes the CSV's row order as a time axis: the first
                 (1 - test_size) fraction of rows become train, the last
                 fraction become test, with NO shuffling. This keeps whole
                 attack bursts on one side of the split, which is the
                 standard fix for this kind of leakage.
    """
    if split_strategy == "time":
        n = len(X)
        split_at = int(n * (1 - test_size))
        return X[:split_at], X[split_at:], y[:split_at], y[split_at:]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def check_train_test_leakage(
    train_features: list[np.ndarray],
    test_features: list[np.ndarray],
    client_names: list[str],
    near_dup_distance: float = 1e-3,
    max_test_samples_checked: int = 3000,
    random_state: int = 42,
) -> dict:
    """
    Per-device leakage check between a client's train and test split.

    Reports, for each device:
      - exact duplicate rows appearing in both train and test
      - near-duplicate rows: test rows whose nearest neighbour in the
        train set is within `near_dup_distance` (raw, unscaled feature
        space). This is the specific N-BaIoT leakage pattern — consecutive
        flow-capture windows during the same attack burst produce near-
        identical statistical features, so a random split can leak an
        almost-identical row across the split.

    Uses a KD-tree per device so this stays fast even on large clients;
    test rows are subsampled if there are more than `max_test_samples_checked`.
    A near-duplicate rate above ~5% is a strong signal that reported
    accuracy is inflated by leakage rather than genuine generalisation.
    """
    print("\n" + "=" * 60)
    print("  TRAIN/TEST LEAKAGE CHECK")
    print("=" * 60)
    rng = np.random.RandomState(random_state)
    report = {}

    #name, X_tr, X_te = client_names[0], train_features[0], test_features[0]
    for name, X_tr, X_te in zip(client_names, train_features, test_features):
        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        # Exact duplicates: hash each row, check membership.
        train_row_set = {tuple(row) for row in X_tr}
        exact_dupes = sum(1 for row in X_te if tuple(row) in train_row_set)

        # Near-duplicates via nearest-neighbour distance.
        if len(X_te) > max_test_samples_checked:
            sample_idx = rng.choice(len(X_te), max_test_samples_checked, replace=False)
            test_sample = X_te[sample_idx]
        else:
            test_sample = X_te

        # KD-tree query is fast even on large train sets, so we can afford to
        # check every test row for its nearest neighbour in the train set.
        tree = cKDTree(X_tr)
        dists, _ = tree.query(test_sample, k=1)
        near_dupes = int((dists < near_dup_distance).sum())
        near_dupe_rate = near_dupes / len(test_sample)

        report[name] = {
            "exact_duplicates": exact_dupes,
            "test_rows_checked": len(test_sample),
            "near_duplicates": near_dupes,
            "near_duplicate_rate": near_dupe_rate,
        }

        flag = ""
        if exact_dupes > 0 or near_dupe_rate > 0.05:
            flag = "  <-- LIKELY LEAKAGE"
        print(f"    {name:<45} exact: {exact_dupes:>4}  |  near-dupes: {near_dupes:>4}/{len(test_sample)} "
              f"({near_dupe_rate:.1%}){flag}")

    total_near = sum(r["near_duplicates"] for r in report.values())
    total_checked = sum(r["test_rows_checked"] for r in report.values())
    total_exact = sum(r["exact_duplicates"] for r in report.values())
    overall_rate = total_near / total_checked if total_checked else 0.0
    print(f"\n  Overall: {total_exact} exact duplicates, "
          f"{total_near}/{total_checked} near-duplicates ({overall_rate:.1%})")
    if total_exact > 0 or overall_rate > 0.05:
        print("  -> Meaningful leakage risk. Consider split_strategy='time' or "
              "leave-one-device-out evaluation before trusting accuracy numbers.")
    else:
        print("  -> No strong leakage signal from this check.")
    return report


def load_device_data(device_path: Path) -> pd.DataFrame:
    """
    Loads benign + attack CSV files for one IoT device.
    Labels: 0 = Benign, 1 = Malicious
    """
    dfs = []

    # Benign traffic
    benign_dir = device_path / "benign"
    if benign_dir.exists():
        for csv_file in benign_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            df["label"] = 0
            dfs.append(df)

    # Attack traffic (Mirai)
    mirai_dir = device_path / "mirai_attacks"
    if mirai_dir.exists():
        for csv_file in mirai_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            df["label"] = 1
            dfs.append(df)

    # Attack traffic (Gafgyt)
    gafgyt_dir = device_path / "gafgyt_attacks"
    if gafgyt_dir.exists():
        for csv_file in gafgyt_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            df["label"] = 1
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found under {device_path}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna()

    # N-BaIoT's raw CSVs contain genuine duplicate flow-statistic rows
    # (not just near-duplicates from adjacent capture windows). If these
    # survive into both the train and test split, the model "generalizes"
    # by recognising rows it was literally trained on. Dedup here, before
    # any split, so duplicates can't land on both sides.
    #
    # Dedup at float32 precision, matching what the model actually sees
    # downstream (features get cast to float32 for training) — two rows
    # that differ only past float32's ~7 significant digits are, for the
    # model's purposes, the same row, even if pandas' float64 view treats
    # them as distinct.
    feature_cols = [c for c in combined.columns if c != "label"]
    n_before = len(combined)
    dup_mask = combined[feature_cols].astype(np.float32).duplicated()
    combined = combined[~dup_mask]
    n_dropped = int(dup_mask.sum())
    if n_dropped > 0:
        print(f"    Dropped {n_dropped:,} exact-duplicate rows "
              f"({100 * n_dropped / n_before:.1f}% of {device_path.name})")

    return combined


def load_flat_device_data(dataset_root: Path, device_id: str) -> pd.DataFrame:
    """
    Loads flat N-BaIoT CSV files that are named like `1.benign.csv`.

    The workspace copy of the dataset is stored in this layout, so we group
    files by device prefix and infer the label from the filename.
    """
    dfs = []
    for csv_file in sorted(dataset_root.glob(f"{device_id}.*.csv")):
        label = 0 if ".benign." in csv_file.name else 1
        df = pd.read_csv(csv_file)
        df["label"] = label
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found for device {device_id} under {dataset_root}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna()

    # Same rationale as load_device_data: dedup at float32 precision (what
    # the model actually sees) before any split, so duplicates can't leak
    # across train/test.
    feature_cols = [c for c in combined.columns if c != "label"]
    n_before = len(combined)
    dup_mask = combined[feature_cols].astype(np.float32).duplicated()
    combined = combined[~dup_mask]
    n_dropped = int(dup_mask.sum())
    if n_dropped > 0:
        print(f"    Dropped {n_dropped:,} exact-duplicate rows "
              f"({100 * n_dropped / n_before:.1f}% of device {device_id})")

    return combined


def load_nbaiot_federated(
    dataset_root: Path = DATASET_ROOT,
    max_samples_per_device: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
    k_best_features: int | None = 40,
    split_strategy: str = "random",
    check_leakage: bool = True,
) -> tuple[list[DataLoader], list[DataLoader], StandardScaler, list[int], list[str]]:
    """
    Loads N-BaIoT data and partitions it into per-device (Non-IID) FL clients.

    Each IoT device naturally produces a different traffic distribution —
    this is the real Non-IID problem described in the research gaps.

    Returns:
        train_loaders  — one DataLoader per client (device)
        test_loaders   — one DataLoader per client
        scaler         — fitted StandardScaler (save for inference)
    """
    print("Loading N-BaIoT dataset...")
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    client_names = []

    folder_devices = [device_name for device_name in DEVICE_FOLDERS if (dataset_root / device_name).exists()]
    flat_device_ids = sorted(
        {
            csv_file.stem.split(".", 1)[0]
            for csv_file in dataset_root.glob("*.csv")
            if csv_file.stem.split(".", 1)[0].isdigit()
        }
    )

    if folder_devices:
        for device_name in folder_devices:
            device_path = dataset_root / device_name
            try:
                df = load_device_data(device_path)
                if len(df) > max_samples_per_device:
                    if split_strategy == "time":
                        df = df.iloc[:max_samples_per_device]  # keep native row order intact
                    else:
                        df = df.sample(n=max_samples_per_device, random_state=random_state)
                X_dev = df.iloc[:, :-1].values.astype(np.float32)
                y_dev = df["label"].values.astype(np.int64)
                X_tr, X_te, y_tr, y_te = _split_device(
                    X_dev, y_dev, test_size, random_state, split_strategy
                )
                client_names.append(device_name)
                train_features.append(X_tr)
                train_labels.append(y_tr)
                test_features.append(X_te)
                test_labels.append(y_te)
                print(f"  Loaded {device_name}: {len(df):,} samples "
                      f"({(df['label'] == 0).sum():,} benign, "
                      f"{(df['label'] == 1).sum():,} attack)")
            except Exception as e:
                print(f"  WARNING: Could not load {device_name}: {e}")
    elif flat_device_ids:
        for device_id in flat_device_ids:
            client_name = f"Device_{device_id}"
            try:
                df = load_flat_device_data(dataset_root, device_id)
                if len(df) > max_samples_per_device:
                    if split_strategy == "time":
                        df = df.iloc[:max_samples_per_device]  # keep native row order intact
                    else:
                        df = df.sample(n=max_samples_per_device, random_state=random_state)
                X_dev = df.iloc[:, :-1].values.astype(np.float32)
                y_dev = df["label"].values.astype(np.int64)
                X_tr, X_te, y_tr, y_te = _split_device(
                    X_dev, y_dev, test_size, random_state, split_strategy
                )
                client_names.append(client_name)
                train_features.append(X_tr)
                train_labels.append(y_tr)
                test_features.append(X_te)
                test_labels.append(y_te)
                print(f"  Loaded {client_name}: {len(df):,} samples "
                      f"({(df['label'] == 0).sum():,} benign, "
                      f"{(df['label'] == 1).sum():,} attack)")
            except Exception as e:
                print(f"  WARNING: Could not load {client_name}: {e}")

    if not train_features:
        raise RuntimeError(
            f"No N-BaIoT data found at '{dataset_root}'.\n"
            "Please download the dataset and set DATASET_ROOT correctly."
        )

    # Fit scaler/selector on TRAINING data only to avoid leakage into the test split.
    X_train_all = np.vstack(train_features)
    y_train_all = np.concatenate(train_labels)

    # ------------------------------------------------------------------
    # Leakage check on the RAW feature space (before any reduction).
    # This is the scientifically meaningful check: it asks whether the
    # original flow records themselves are duplicated/near-identical
    # across train/test. Checking AFTER chi2 selection would be wrong —
    # N-BaIoT's 115 features are highly redundant by construction (the
    # same stats computed over 5 overlapping time windows), so collapsing
    # to k_best_features can make genuinely distinct rows collide in the
    # reduced space and look like leakage that isn't really there.
    # ------------------------------------------------------------------
    if check_leakage:
        print("\n[Leakage check — RAW 115-feature space, before feature selection]")
        check_train_test_leakage(train_features, test_features, client_names)

    # ------------------------------------------------------------------
    # Chi-squared feature selection (115 -> k_best_features)
    # chi2 requires non-negative inputs, so we min-max scale first, select
    # the top-k features on the *global* training pool, then z-score only
    # the retained columns for the models. This is what shrinks the model
    # for lightweight / Raspberry-Pi-class inference.
    # ------------------------------------------------------------------
    selected_idx = list(range(X_train_all.shape[1]))
    if k_best_features is not None and k_best_features < X_train_all.shape[1]:
        minmax = MinMaxScaler()
        X_train_nonneg = minmax.fit_transform(X_train_all)
        selector = SelectKBest(score_func=chi2, k=k_best_features)
        selector.fit(X_train_nonneg, y_train_all)
        selected_idx = np.where(selector.get_support())[0].tolist()
        print(f"\nChi-squared feature selection: {X_train_all.shape[1]} -> {len(selected_idx)} features")

        train_features = [X[:, selected_idx] for X in train_features]
        test_features  = [X[:, selected_idx] for X in test_features]
        X_train_all = np.vstack(train_features)

        if check_leakage:
            print(f"\n[Leakage check — reduced {len(selected_idx)}-feature space, for comparison only]")
            print("(Higher numbers here than above are expected and are NOT extra real leakage —")
            print(" they're coincidental collisions caused by dropping redundant columns.)")
            check_train_test_leakage(train_features, test_features, client_names)

    scaler = StandardScaler()
    scaler.fit(X_train_all)

    train_loaders, test_loaders = [], []

    for device_name, X_tr, y_tr, X_te, y_te in zip(
        client_names, train_features, train_labels, test_features, test_labels
    ):
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        test_ds  = TensorDataset(torch.tensor(X_te),  torch.tensor(y_te))

        train_loaders.append(DataLoader(train_ds, batch_size=64, shuffle=True))
        test_loaders.append(DataLoader(test_ds,  batch_size=64, shuffle=False))

    print(f"\nTotal clients (devices): {len(train_loaders)}")
    print(f"Feature dimensions    : {X_train_all.shape[1]}")
    return train_loaders, test_loaders, scaler, selected_idx, client_names


def split_train_validation_loaders(
    train_loaders: list[DataLoader],
    val_split: float = 0.2,
    random_state: int = 42,
) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Split each client's training dataset into a train subset and a validation subset.

    This keeps the test loaders untouched while giving the FL loop a validation
    signal for early stopping.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    train_sub_loaders = []
    val_loaders = []

    for client_idx, loader in enumerate(train_loaders):
        dataset = loader.dataset
        dataset_size = len(dataset)

        if dataset_size < 2:
            train_sub_loaders.append(loader)
            val_loaders.append(DataLoader(dataset, batch_size=loader.batch_size or 64, shuffle=False))
            continue

        val_size = max(1, int(dataset_size * val_split))
        train_size = dataset_size - val_size
        if train_size < 1:
            train_size = dataset_size - 1
            val_size = 1

        generator = torch.Generator().manual_seed(random_state + client_idx)
        permutation = torch.randperm(dataset_size, generator=generator).tolist()
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        batch_size = loader.batch_size or 64
        train_sub_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))

    return train_sub_loaders, val_loaders


# =============================================================================
# 2. DEMO / SYNTHETIC DATA (i used it to test the code before the real dataset for
#  quicker development and testing)
# =============================================================================

def make_synthetic_noniid_loaders(
    num_clients: int = 9,
    input_size: int = 115,
    num_classes: int = 2,
    samples_per_client: int = 1000,
    random_state: int = 42,
) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Synthetic Non-IID data that mimics the N-BaIoT heterogeneity:
      - Each device has a different class-imbalance ratio
        (some cameras mostly see attacks; thermostats mostly benign)
      - Feature distributions are shifted per device

    Use this for development/testing before the real dataset is available.
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Create attack ratios for each client (0.05 to 0.95)
    attack_ratios = np.linspace(0.05, 0.95, num_clients)

    train_loaders, test_loaders = [], []


    for i in range(num_clients):
        ratio = attack_ratios[i]
        n_attack = int(samples_per_client * ratio)
        n_benign = samples_per_client - n_attack

        # Device-specific feature shift simulates different traffic patterns
        device_shift = np.random.randn(input_size).astype(np.float32) * 0.5

        benign_feat = np.random.randn(n_benign, input_size).astype(np.float32) + device_shift
        attack_feat = np.random.randn(n_attack, input_size).astype(np.float32) + device_shift + 1.5

        X = np.vstack([benign_feat, attack_feat])
        y = np.array([0] * n_benign + [1] * n_attack, dtype=np.int64)

        perm = np.random.permutation(len(y))
        X, y = X[perm], y[perm]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state)

        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        test_ds  = TensorDataset(torch.tensor(X_te),  torch.tensor(y_te))

        train_loaders.append(DataLoader(train_ds, batch_size=64, shuffle=True))
        test_loaders.append(DataLoader(test_ds,  batch_size=64, shuffle=False))

    print(f"[Synthetic Non-IID] Created {num_clients} clients with varying attack ratios.")
    print(f"  Attack ratios: {[f'{r:.2f}' for r in attack_ratios]}")
    return train_loaders, test_loaders


# =============================================================================
# 3. MODEL — Lightweight MLP (IoT-suitable)
# =============================================================================

class SimpleMLP(nn.Module):
    """
    Lightweight 2-layer MLP for binary intrusion detection.
    Designed to fit within IoT edge device constraints.
    Input: 115 N-BaIoT statistical flow features
    Output: 2 classes (Benign / Malicious)
    """
    def __init__(self, input_size: int = 115, hidden_size: int = 32, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),        # light regularisation
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 3b. CLASS-WEIGHT HELPER — for severe per-client label skew
# =============================================================================

def compute_class_weights(loader: DataLoader, num_classes: int = 2) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from a single client's own
    local training labels (not the global dataset). Used to counteract
    severe per-client label skew — e.g. a client whose local data is 99%
    one class, where an unweighted loss lets gradients from the majority
    class dominate and the model never learns the minority class on that
    client, even for data it directly trained on.

    Weight formula: w_c = N / (num_classes * count_c), i.e. rarer classes
    get proportionally larger weight. Falls back to uniform weights (all
    1.0) if a class is entirely absent from this client's local data,
    since a weight can't meaningfully compensate for zero examples.
    """
    all_targets = torch.cat([target for _, target in loader])
    counts = torch.bincount(all_targets, minlength=num_classes).float()
    if (counts == 0).any():
        return torch.ones(num_classes)
    weights = len(all_targets) / (num_classes * counts)
    return weights


# =============================================================================
# 4. CLIENT TRAINING — FedAvg
# =============================================================================

def client_update_fedavg(
    client_model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epochs: int = 2,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """
    Standard local training for FedAvg — no proximal term.

    class_weights (optional): per-class weights passed to CrossEntropyLoss,
    computed from THIS client's own local label distribution. Critical
    under severe per-client label skew (e.g. a client that's 99% one
    class) — without this, the loss gradient is dominated by the majority
    class and the model learns almost nothing about the minority class on
    that client, which then drags down that client's own precision/recall
    even on data it trained on.
    """
    client_model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    initial_loss, final_loss = None, None

    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(client_model(data), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    return client_model.state_dict(), initial_loss, final_loss


# =============================================================================
# 5. CLIENT TRAINING — FedProx
# =============================================================================

def client_update_fedprox(
    client_model: nn.Module,
    global_model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epochs: int = 2,
    mu: float = 0.01,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """
    FedProx local training with proximal regularisation term.

    The key difference from FedAvg:

        L_FedProx = L_task + (μ/2) * ||w - w_global||²

    The proximal term (μ/2)||w - w_global||² penalises the local model
    for drifting too far from the global model. This is critical for
    Non-IID data because:
      - IoT cameras have very different traffic from thermostats
      - Without the term, local models overfit their own distribution
      - With μ > 0, updates stay anchored near the global optimum

    class_weights (optional): per-class weights computed from this
    client's own local label distribution, passed to CrossEntropyLoss.
    Note this addresses a DIFFERENT problem than μ: μ corrects for
    parameter/feature drift between clients; class_weights corrects for
    label-distribution skew WITHIN a client's own local training. Both
    can be needed at once under severe non-IID label skew.

    Args:
        mu: proximal coefficient (0 = pure FedAvg; higher = tighter anchor)
            Typical range: 0.001 – 0.1. Start with 0.01 for N-BaIoT.
    """
    client_model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Freeze a snapshot of the global weights (used in proximal term)
    global_params = {
        name: param.detach().clone()
        for name, param in global_model.named_parameters()
    }

    initial_loss, final_loss = None, None

    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for data, target in train_loader:
            optimizer.zero_grad()

            # --- Task loss (cross-entropy) ---
            task_loss = criterion(client_model(data), target)

            #  Proximal term: penalise drift from global model 
            # ||w_local - w_global||^2  (summed across all parameters)
            prox_loss = torch.tensor(0.0)
            for name, param in client_model.named_parameters():
                prox_loss = prox_loss + torch.norm(param - global_params[name]) ** 2

            loss = task_loss + (mu / 2.0) * prox_loss
            loss.backward()
            optimizer.step()
            epoch_loss += task_loss.item()
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    return client_model.state_dict(), initial_loss, final_loss


# =============================================================================
# 6. SERVER AGGREGATION — FedAvg (same for both algorithms)
# =============================================================================

def quantize_state_dict(state_dict: dict, dtype: torch.dtype = torch.float16) -> dict:
    """
    Simulates update compression for the uplink (client -> server transfer).

    Casting the client's weight tensors to float16 halves the payload size
    before "transmission". The server upcasts back to float32 before
    aggregating, so training precision is unaffected — only the simulated
    wire size changes. Returns the quantised dict plus nothing else; use
    `state_dict_size_bytes` to measure the before/after payload.
    """
    return {k: v.to(dtype) for k, v in state_dict.items()}


def state_dict_size_bytes(state_dict: dict) -> int:
    """Rough payload size (bytes) of a state_dict, for compression reporting."""
    return sum(v.element_size() * v.nelement() for v in state_dict.values())


def server_aggregate(
    global_model: nn.Module,
    client_weights: list[dict],
    client_sizes: list[int] | None = None,
) -> nn.Module:
    """
    FedAvg aggregation: weighted mean of client weight tensors.
    If client_sizes are not supplied, falls back to an equal-weight mean.
    """
    global_dict = global_model.state_dict()

    if client_sizes is None:
        client_sizes = [1] * len(client_weights)

    total_size = float(sum(client_sizes))
    client_scalars = [size / total_size for size in client_sizes]

    for k in global_dict.keys():
        aggregated = torch.zeros_like(global_dict[k], dtype=torch.float32)
        for client_state, scalar in zip(client_weights, client_scalars):
            aggregated += client_state[k].float() * scalar
        global_dict[k] = aggregated.to(global_dict[k].dtype)
    global_model.load_state_dict(global_dict)
    return global_model


# =============================================================================
# 7. EVALUATION
# =============================================================================

def evaluate_global_model(
    global_model: nn.Module,
    test_loaders: list[DataLoader],
    device_names: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluates the global model on each client's local test set.
    Returns per-client and overall accuracy.
    """
    global_model.eval()
    all_preds, all_targets = [], []
    per_client_acc = []

    with torch.no_grad():
        for i, loader in enumerate(test_loaders):
            correct, total = 0, 0
            preds_c, targets_c = [], []
            for data, target in loader:
                outputs = global_model(data)
                predicted = outputs.argmax(dim=1)
                correct  += (predicted == target).sum().item()
                total    += target.size(0)
                preds_c.extend(predicted.tolist())
                targets_c.extend(target.tolist())

            acc = correct / total if total > 0 else 0.0
            per_client_acc.append(acc)
            all_preds.extend(preds_c)
            all_targets.extend(targets_c)

            name = device_names[i] if device_names else f"Client {i+1}"
            if verbose:
                print(f"    {name:<45} Acc: {acc:.4f}")

    overall_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    # Macro-F1 treats both classes equally regardless of how imbalanced
    # they are — unlike accuracy, a model that just predicts the majority
    # class on a 99%-skewed client scores poorly here, not well.
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    # MCC (Matthews Correlation Coefficient) uses all four confusion-matrix
    # quadrants (TP, TN, FP, FN) symmetrically, unlike F1 which weights
    # toward the positive class. Range is -1 to +1 (0 = random, 1 = perfect,
    # -1 = perfectly wrong) rather than F1's 0-to-1 range — thresholds
    # tuned for F1 do NOT transfer directly to MCC. A model that just
    # predicts the majority class on a skewed client scores MCC ~0, not a
    # deceptively high number, same protection F1 gives but arguably more
    # robust under imbalance since it penalises both false positives and
    # false negatives symmetrically.
    mcc = matthews_corrcoef(all_targets, all_preds)
    return {
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "per_client_accuracy": per_client_acc,
        "all_predictions": all_preds,
        "all_targets": all_targets,
    }


# =============================================================================
# 8. FEDERATED LEARNING RUNNER
# =============================================================================

def run_federated_learning(
    algorithm: str,                    # "fedavg" or "fedprox"
    train_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    input_size: int = 115,
    hidden_size: int = 32,
    num_classes: int = 2,
    num_rounds: int = 10,
    local_epochs: int = 1,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    mu: float = 0.01,                  # FedProx only — proximal coefficient
    val_split: float = 0.2,
    patience: int = 3,
    min_delta: float = 1e-4,
    device_names: list[str] | None = None,
    client_fraction: float = 1.0,      # Client selection: fraction of clients sampled per round
    conditional_update_delta: float = 0.0,  # Skip an update if local loss doesn't improve by at least this much
    quantize_updates: bool = False,    # Simulate uplink compression (float32 -> float16)
    use_class_weights: bool = True,    # Per-client inverse-frequency class weighting — critical under label skew
    random_state: int = 42,
) -> tuple[nn.Module, list[float], list[float], dict]:
    """
    Main federated learning loop.

    FedAvg:   standard averaging — all clients train freely
    FedProx:  adds proximal term μ/2 ||w - w_global||² to each client's loss

    Returns the trained global model, round-by-round test accuracy history,
    round-by-round train accuracy history, and a stats dict with cumulative
    communication cost and convergence-round info (used by the ablation
    table / ADR-style comparison across configs).
    """
    assert algorithm in ("fedavg", "fedprox"), "algorithm must be 'fedavg' or 'fedprox'"
    num_clients = len(train_loaders)

    print(f"\n{'='*60}")
    print(f"  Algorithm  : {algorithm.upper()}")
    print(f"  Clients    : {num_clients}")
    print(f"  Rounds     : {num_rounds}")
    print(f"  LR         : {lr}  |  Local epochs: {local_epochs}")
    print(f"  Weight decay: {weight_decay}")
    if algorithm == "fedprox":
        print(f"  Mu (μ)     : {mu}  (proximal coefficient)")
    print(f"  Val split  : {val_split}  |  Patience: {patience}")
    print(f"{'='*60}")

    train_loaders, val_loaders = split_train_validation_loaders(
        train_loaders, val_split=val_split
    )

    global_model = SimpleMLP(input_size, hidden_size, num_classes)
    accuracy_history = []
    train_accuracy_history = []
    best_val_metric = float("-inf")
    best_model_state = copy.deepcopy(global_model.state_dict())
    stale_rounds = 0
    best_round = 0                    # 1-indexed round that produced best_model_state — proxy for "rounds to converge"
    total_bytes_before = 0            # cumulative uncompressed uplink payload across the whole run
    total_bytes_after = 0             # cumulative payload actually "sent" (post-compression if enabled)

    for round_idx in range(num_rounds):
        # ------------------------------------------------------------
        # Client selection: sample a fraction of clients this round
        # (mirrors real cross-device FL, e.g. "top 20 of 100 devices").
        # With client_fraction=1.0 (default) all clients participate,
        # matching the original 9-device behaviour.
        # ------------------------------------------------------------
        n_selected = max(1, int(round(num_clients * client_fraction)))
        rng = np.random.RandomState(random_state + round_idx)
        selected_clients = sorted(rng.choice(num_clients, size=n_selected, replace=False).tolist())

        client_weights = []
        client_sizes = []
        skipped_clients = 0
        bytes_before, bytes_after = 0, 0

        for i in selected_clients:
            client_model = copy.deepcopy(global_model)
            optimizer    = optim.SGD(
                client_model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )

            class_weights = compute_class_weights(train_loaders[i], num_classes) if use_class_weights else None

            if algorithm == "fedavg":
                weights, loss_before, loss_after = client_update_fedavg(
                    client_model, optimizer, train_loaders[i], epochs=local_epochs,
                    class_weights=class_weights,
                )
            else:  # fedprox
                weights, loss_before, loss_after = client_update_fedprox(
                    client_model, global_model, optimizer, train_loaders[i],
                    epochs=local_epochs, mu=mu, class_weights=class_weights,
                )

            # ------------------------------------------------------------
            # Conditional update: only ship this client's weights to the
            # server if local loss improved by at least `conditional_update_delta`.
            # Skipping stale/unhelpful updates cuts uplink traffic.
            # ------------------------------------------------------------
            improvement = (loss_before or 0.0) - (loss_after or 0.0)
            if conditional_update_delta > 0 and improvement < conditional_update_delta:
                skipped_clients += 1
                continue

            # ------------------------------------------------------------
            # Update compression: simulate halving the payload with fp16
            # before it goes "over the wire" to the server.
            # ------------------------------------------------------------
            bytes_before += state_dict_size_bytes(weights)
            if quantize_updates:
                weights = quantize_state_dict(weights)          # fp32 -> fp16 ("on the wire")
                bytes_after += state_dict_size_bytes(weights)
                weights = {k: v.float() for k, v in weights.items()}  # upcast back for aggregation
            else:
                bytes_after += state_dict_size_bytes(weights)

            client_weights.append(weights)
            client_sizes.append(len(train_loaders[i].dataset))

        if skipped_clients:
            print(f"  Conditional updates: {skipped_clients}/{len(selected_clients)} client(s) skipped (no sufficient local improvement)")
        if quantize_updates and bytes_before:
            print(f"  Update compression : {bytes_before/1024:.1f} KB -> {bytes_after/1024:.1f} KB "
                  f"({100 * (1 - bytes_after / bytes_before):.0f}% smaller)")
        total_bytes_before += bytes_before
        total_bytes_after += bytes_after

        if not client_weights:
            print("  No client updates this round — keeping previous global model.")
        else:
            # Aggregate on server
            global_model = server_aggregate(global_model, client_weights, client_sizes)

        # Evaluate after this round
        print(f"\nRound {round_idx + 1}/{num_rounds} — Validation accuracy:")
        val_results = evaluate_global_model(global_model, val_loaders, device_names)
        val_accuracy = val_results["overall_accuracy"]
        val_macro_f1 = val_results["macro_f1"]
        print(f"  Validation macro-F1: {val_macro_f1:.4f}  (accuracy: {val_accuracy:.4f})")

        # Model selection uses macro-F1, not raw accuracy — under severe
        # per-client label skew, a model that just predicts the majority
        # class scores deceptively high accuracy while doing nothing
        # useful; macro-F1 penalises that.
        if val_macro_f1 > best_val_metric + min_delta:
            best_val_metric = val_macro_f1
            best_model_state = copy.deepcopy(global_model.state_dict())
            best_round = round_idx + 1
            stale_rounds = 0
        else:
            stale_rounds += 1

        print(f"\nRound {round_idx + 1}/{num_rounds} — Per-client accuracy:")
        results = evaluate_global_model(global_model, test_loaders, device_names)
        overall = results["overall_accuracy"]
        accuracy_history.append(overall)

        # Train-side accuracy (same clients' train_loaders, silent) — the gap
        # between this and `overall` (test) is the actual overfitting signal.
        # A flat test curve alone (like the round-by-round print above) does
        # NOT show overfitting on its own.
        train_results = evaluate_global_model(global_model, train_loaders, device_names, verbose=False)
        train_accuracy_history.append(train_results["overall_accuracy"])

        print(f"  >> Overall accuracy: {overall:.4f}  (train: {train_results['overall_accuracy']:.4f}, "
              f"gap: {train_results['overall_accuracy'] - overall:+.4f})")

        if stale_rounds >= patience:
            print(
                f"\nEarly stopping triggered after {round_idx + 1} rounds: "
                f"validation accuracy did not improve for {patience} rounds."
            )
            break

    global_model.load_state_dict(best_model_state)

    stats = {
        "rounds_run": round_idx + 1,
        "best_round": best_round,          # rounds-to-converge proxy
        "total_bytes_before": total_bytes_before,
        "total_bytes_after": total_bytes_after,
        "final_test_accuracy": accuracy_history[-1] if accuracy_history else None,
    }
    return global_model, accuracy_history, train_accuracy_history, stats

# =============================================================================
# 8b. MODEL PRUNING & EDGE-DEVICE BENCHMARK (Lightweight Model Design)
# =============================================================================

def prune_global_model(
    model: nn.Module,
    amount: float = 0.3,
) -> nn.Module:
    """
    Magnitude-based unstructured pruning of every Linear layer.

    Zeroes out the smallest-magnitude `amount` fraction of weights in each
    Linear layer. The pruning mask stays registered (not yet "removed") so
    that a subsequent fine-tuning pass can't undo the zeros — call
    `finalize_pruning()` once fine-tuning is done to bake the mask in
    permanently. This is what shrinks the effective parameter count for
    Raspberry-Pi-class inference, per the "Lightweight Model Design" goal.
    """
    pruned = copy.deepcopy(model)
    for module in pruned.net:
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return pruned


def finalize_pruning(model: nn.Module) -> nn.Module:
    """Bakes the pruning mask into the weights permanently (call after fine-tuning)."""
    for module in model.net:
        if isinstance(module, nn.Linear) and prune.is_pruned(module):
            prune.remove(module, "weight")
    return model


def fine_tune(
    model: nn.Module,
    train_loaders: list[DataLoader],
    epochs: int = 2,
    lr: float = 0.001,
) -> nn.Module:
    """A few epochs of centralised fine-tuning to recover any accuracy lost to pruning."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for loader in train_loaders:
            for data, target in loader:
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
    return model


def compute_model_flops(model: nn.Module) -> dict:
    """
    Computes MACs (multiply-accumulate operations) and FLOPs for a single
    forward pass, counting only nn.Linear layers (ReLU/Dropout are ~free
    by comparison and standard practice omits them). Bias-add terms are
    a small additional cost not counted here, consistent with how FLOPs
    are typically reported for lightweight architectures in the literature
    (e.g. MobileNet-style papers) — this is a slight underestimate, not
    an overestimate.

        MACs  = sum over Linear layers of (in_features * out_features)
        FLOPs = 2 * MACs   (each MAC = 1 multiply + 1 add)
    """
    total_macs = 0
    for module in model.net:
        if isinstance(module, nn.Linear):
            total_macs += module.in_features * module.out_features
    return {"macs": total_macs, "flops": 2 * total_macs}


def benchmark_model(model: nn.Module, input_size: int, label: str = "") -> dict:
    """
    Reports the standard set of metrics used to justify a "lightweight
    model" claim in the FL-IDS / edge-deployment literature:

      - Parameter count (total and non-zero, post-pruning)
      - Model size on disk (KB) — actual serialized weight footprint,
        directly relevant to storage-constrained IoT/edge devices
      - FLOPs / MACs per inference — the standard computational-complexity
        metric for comparing "lightweight" architectures (independent of
        the benchmarking machine's speed, unlike latency)
      - CPU inference latency per sample — a proxy for Raspberry-Pi-class
        feasibility, since such devices have no GPU
      - Throughput (samples/sec) — the inverse of latency, useful for
        framing as "can this keep up with incoming traffic"
    """
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    model_size_kb = state_dict_size_bytes(model.state_dict()) / 1024
    flop_stats = compute_model_flops(model)

    model.eval()
    dummy = torch.randn(1, input_size)
    with torch.no_grad():
        for _ in range(10):          # warm-up
            model(dummy)
        import time
        start = time.perf_counter()
        for _ in range(200):
            model(dummy)
        elapsed_ms = (time.perf_counter() - start) / 200 * 1000
    throughput = 1000.0 / elapsed_ms if elapsed_ms > 0 else float("inf")

    print(f"  [{label}] Params: {total_params:,}  |  Non-zero: {nonzero_params:,} "
          f"({100 * nonzero_params / total_params:.1f}%)  |  Size: {model_size_kb:.2f} KB  |  "
          f"MACs: {flop_stats['macs']:,}  |  FLOPs: {flop_stats['flops']:,}  |  "
          f"Latency: {elapsed_ms:.3f} ms/sample  |  Throughput: {throughput:.0f} samples/sec")
    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "model_size_kb": model_size_kb,
        "macs": flop_stats["macs"],
        "flops": flop_stats["flops"],
        "latency_ms": elapsed_ms,
        "throughput_samples_per_sec": throughput,
    }


def compare_model_footprints(baseline_stats: dict, optimized_stats: dict, label: str = "") -> dict:
    """
    Computes before/after reduction percentages between two benchmark_model()
    results — the "X% smaller, Y% faster" numbers that actually justify a
    lightweight-model claim, rather than reporting two sets of raw numbers
    side by side and leaving the reader to do the subtraction.
    """
    def pct_reduction(before, after):
        return 100 * (1 - after / before) if before else 0.0

    comparison = {
        "param_reduction_pct": pct_reduction(baseline_stats["total_params"], optimized_stats["nonzero_params"]),
        "size_reduction_pct": pct_reduction(baseline_stats["model_size_kb"], optimized_stats["model_size_kb"]),
        "flops_reduction_pct": pct_reduction(baseline_stats["flops"], optimized_stats["flops"]),
        "latency_reduction_pct": pct_reduction(baseline_stats["latency_ms"], optimized_stats["latency_ms"]),
        "speedup_factor": (baseline_stats["latency_ms"] / optimized_stats["latency_ms"]
                            if optimized_stats["latency_ms"] > 0 else float("inf")),
    }
    print(f"\n  [{label}] vs baseline: "
          f"{comparison['param_reduction_pct']:.1f}% fewer non-zero params, "
          f"{comparison['size_reduction_pct']:.1f}% smaller on disk, "
          f"{comparison['flops_reduction_pct']:.1f}% fewer FLOPs, "
          f"{comparison['speedup_factor']:.2f}x faster inference")
    return comparison


# =============================================================================
# 8b2. FEATURE COUNT SWEEP — empirically justify k_best_features
# =============================================================================

def run_feature_count_sweep(
    loader_fn,
    loader_kwargs: dict,
    k_values: list[int],
    hidden_size: int = 32,
    num_classes: int = 2,
    num_rounds: int = 6,
    local_epochs: int = 2,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    mu: float = 0.01,
    min_acceptable_mcc: float | None = None,
    selection_mode: str = "auto",
) -> tuple[pd.DataFrame, int]:
    """
    Empirically justifies the choice of k_best_features by sweeping several
    values and reporting accuracy, macro-F1, MCC, model size, and FLOPs for
    each — instead of asserting a single number (e.g. "40") without evidence.

    Selection is ranked by MCC (Matthews Correlation Coefficient), not
    macro-F1. MCC uses all four confusion-matrix quadrants symmetrically
    (TP, TN, FP, FN), which is generally considered more robust than F1
    under class imbalance — relevant given this project's history of
    severely skewed per-client label distributions. Macro-F1 and accuracy
    are still reported in the table for reference/comparison, just not
    used to rank.

    IMPORTANT — MCC's range is -1 to +1 (0 = random, 1 = perfect, -1 =
    perfectly wrong), NOT F1's 0-to-1 range. A threshold tuned for F1
    (e.g. 0.85) does NOT mean the same thing for MCC — an MCC of 0.85 is
    already very strong; don't reuse F1-scale thresholds here.

    loader_fn: load_nbaiot_federated
    loader_kwargs: all kwargs for loader_fn EXCEPT k_best_features (that's
        swept). e.g. for N-BaIoT: {"dataset_root": DATASET_ROOT,
        "max_samples_per_device": 5000, "check_leakage": False}
        (leakage check is disabled here — this sweep already re-runs it
        redundantly at every k; run it once separately instead)

    selection_mode: which of FOUR philosophies to use for picking a
    single k from the sweep results:

      - "best":  picks the k with the HIGHEST MCC outright, ignoring
        latency/size entirely. Note this will systematically drift toward
        the LARGEST k in your range, since more features almost never
        hurts MCC but always costs more latency/FLOPs — "best" is not
        actually a lightweight-aware mode, even within a constrained range.

      - "efficiency": picks the k with the highest MCC-per-millisecond
        (MCC / latency_ms) — genuinely trades off performance against
        system cost, rather than ignoring cost like "best" does. This is
        the mode to use when you explicitly care about latency, not just
        whether k_values was pre-constrained to a "reasonable" range.

      - "floor": picks the SMALLEST k anywhere in the sweep whose MCC
        clears min_acceptable_mcc, regardless of whether a larger k scores
        higher. Use this when you have a wide k_values range and want the
        lightest model that's merely "good enough", not the best performer.

      - "auto" (default): backward-compatible behaviour — uses "floor" if
        min_acceptable_mcc is set, otherwise "elbow" (smallest k within
        0.02 MCC of the best MCC in the sweep — chases the ceiling first,
        then economizes slightly).

    Either way, look at the full table yourself before deciding — the
    returned suggestion is a starting point, not a verdict. A Pareto-
    frontier view (which k values are not dominated by any other k on
    BOTH MCC and latency simultaneously) is always printed regardless of
    selection_mode, since that's the most honest picture of the tradeoff.
    """
    rows = []
    for k in k_values:
        print(f"\n{'#'*60}\n  FEATURE COUNT SWEEP: k = {k}\n{'#'*60}")
        kwargs = {**loader_kwargs, "k_best_features": k}
        loaded = loader_fn(**kwargs)
        train_loaders, test_loaders, scaler, selected_idx = loaded[0], loaded[1], loaded[2], loaded[3]
        device_names = loaded[4] if len(loaded) > 4 else [f"Client_{i+1}" for i in range(len(train_loaders))]
        input_size = len(selected_idx)

        model, acc_hist, train_hist, stats = run_federated_learning(
            algorithm="fedprox",
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            device_names=device_names,
        )

        eval_result = evaluate_global_model(model, test_loaders, device_names, verbose=False)
        bench = benchmark_model(model, input_size, label=f"k={k}")

        # Efficiency = MCC achieved per millisecond of inference latency.
        # Rewards k values that get good performance WITHOUT paying for it
        # in latency, unlike "best" which ignores cost entirely. Guard
        # against divide-by-zero / negative MCC producing a meaningless
        # negative-latency-adjusted score.
        mcc_val = eval_result["mcc"]
        latency_val = bench["latency_ms"]
        efficiency = (max(mcc_val, 0.0) / latency_val) if latency_val > 0 else 0.0

        rows.append({
            "k_features": k,
            "Accuracy": round(eval_result["overall_accuracy"], 4),
            "Macro-F1": round(eval_result["macro_f1"], 4),
            "MCC": round(mcc_val, 4),
            "Total Params": bench["total_params"],
            "Model Size (KB)": round(bench["model_size_kb"], 2),
            "FLOPs": bench["flops"],
            "Latency (ms/sample)": round(latency_val, 4),
            "Efficiency (MCC/ms)": round(efficiency, 2),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*60}\n  FEATURE COUNT SWEEP SUMMARY\n{'='*60}")
    print(df.to_string(index=False))
    # Sorted by size ascending too, so the size/accuracy tradeoff is visible
    # at a glance regardless of which selection mode is used below.
    print(f"\n  Sorted by Model Size (lightest first):")
    print(df.sort_values("Model Size (KB)").to_string(index=False))

    # Pareto frontier: a k is "dominated" if some OTHER k has both >= MCC
    # AND <= latency (i.e. that other k is strictly better or equal on
    # both axes). Non-dominated k values are the honest set of real
    # tradeoff options — printed regardless of selection_mode.
    is_dominated = []
    for _, row in df.iterrows():
        dominated = ((df["MCC"] >= row["MCC"]) & (df["Latency (ms/sample)"] < row["Latency (ms/sample)"])).any() or \
                    ((df["MCC"] > row["MCC"]) & (df["Latency (ms/sample)"] <= row["Latency (ms/sample)"])).any()
        is_dominated.append(dominated)
    pareto_df = df[~pd.Series(is_dominated, index=df.index)].sort_values("Latency (ms/sample)")
    print(f"\n  Pareto-optimal k values (MCC vs latency — no other k beats one of these on both axes):")
    print(pareto_df[["k_features", "MCC", "Latency (ms/sample)", "Efficiency (MCC/ms)"]].to_string(index=False))

    best_mcc = df["MCC"].max()

    # Resolve "auto" into a concrete mode, for backward compatibility.
    resolved_mode = selection_mode
    if resolved_mode == "auto":
        resolved_mode = "floor" if min_acceptable_mcc is not None else "elbow"

    if resolved_mode == "best":
        # Pure best-performer: ignores latency/size entirely. Will tend to
        # pick the largest k in your range — see docstring warning above.
        best_row = df.loc[df["MCC"].idxmax()]
        suggested_k = best_row["k_features"]
        print(f"\n  Best-performer selection: highest MCC among the tested k values (latency NOT considered)")
        print(f"  Suggested k: {int(suggested_k)}  (MCC: {best_row['MCC']:.4f}, "
              f"Latency: {best_row['Latency (ms/sample)']:.4f} ms/sample)")

    elif resolved_mode == "efficiency":
        # MCC-per-millisecond: genuinely trades off performance against cost.
        best_row = df.loc[df["Efficiency (MCC/ms)"].idxmax()]
        suggested_k = best_row["k_features"]
        print(f"\n  Efficiency selection: highest MCC-per-millisecond among the tested k values")
        print(f"  Suggested k: {int(suggested_k)}  (MCC: {best_row['MCC']:.4f}, "
              f"Latency: {best_row['Latency (ms/sample)']:.4f} ms/sample, "
              f"Efficiency: {best_row['Efficiency (MCC/ms)']:.2f} MCC/ms)")
        print(f"  For comparison, the highest raw MCC in the sweep was "
              f"{best_mcc:.4f} at k={int(df.loc[df['MCC'].idxmax(), 'k_features'])} "
              f"(latency {df.loc[df['MCC'].idxmax(), 'Latency (ms/sample)']:.4f} ms/sample) — "
              f"efficiency selection may trade some of that MCC away for lower latency.")

    elif resolved_mode == "floor":
        # Lightweight-first: smallest k anywhere that clears the floor,
        # regardless of how much better a bigger k scores.
        if min_acceptable_mcc is None:
            raise ValueError("selection_mode='floor' requires min_acceptable_mcc to be set.")
        acceptable = df[df["MCC"] >= min_acceptable_mcc].sort_values("k_features")
        if len(acceptable) == 0:
            print(f"\n  WARNING: no k in the sweep reached min_acceptable_mcc={min_acceptable_mcc:.4f} "
                  f"(best achieved was {best_mcc:.4f}). Falling back to the k with the best MCC in the sweep — "
                  f"lower min_acceptable_mcc or extend k_values and re-run if this isn't good enough.")
            suggested_k = df.loc[df["MCC"].idxmax(), "k_features"]
        else:
            suggested_k = acceptable.iloc[0]["k_features"]
            print(f"\n  Lightweight-first selection: smallest k clearing MCC >= {min_acceptable_mcc:.4f}")
            print(f"  Suggested k: {int(suggested_k)}  (MCC at this k: {acceptable.iloc[0]['MCC']:.4f}, "
                  f"vs best MCC in sweep: {best_mcc:.4f} at k={int(df.loc[df['MCC'].idxmax(), 'k_features'])})")

    else:  # "elbow"
        # Smallest k within 0.02 MCC of the ceiling (scaled down from the
        # old 1-percentage-point F1 margin, since MCC's -1..+1 range means
        # a 0.01 gap is proportionally larger than the same gap in F1).
        candidates = df[df["MCC"] >= best_mcc - 0.02].sort_values("k_features")
        suggested_k = candidates.iloc[0]["k_features"]
        print(f"\n  Elbow selection: smallest k within 0.02 of best MCC ({best_mcc:.4f})")
        print(f"  Suggested k: {int(suggested_k)}")

    print("  (This is a starting point, not an automatic answer — review the full table above.)")

    df.to_csv("feature_count_sweep_results.csv", index=False)
    print("\nSweep results saved to: feature_count_sweep_results.csv")
    return df, int(suggested_k)


# =============================================================================
# 8c. ABLATION STUDY — one table covering all four research gaps
# =============================================================================

def run_ablation_study(
    train_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    device_names: list[str],
    input_size: int,
    hidden_size: int = 32,
    num_classes: int = 2,
    num_rounds: int = 8,
    local_epochs: int = 2,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    mu: float = 0.01,
) -> pd.DataFrame:
    """
    Runs FedProx under four configurations and reports one comparison table:

      1. Baseline           — no client selection, no conditional updates,
                              no compression, no pruning
      2. + Comm. reduction  — client_fraction=0.7, conditional updates,
                              fp16 update compression (research gap #1)
      3. + Pruning          — baseline + post-hoc magnitude pruning,
                              fine-tuned (research gap #2)
      4. All combined       — comm. reduction + pruning together
                              (the number to actually report as your
                              headline "lightweight + low-communication" result)

    Reports, per config: final test accuracy, rounds-to-converge (the round
    that produced the best validation model), total KB "sent" over the
    whole run, total/non-zero parameter count, and CPU inference latency —
    i.e. one row per research gap, directly comparable.
    """
    configs = {
        "1. Baseline":          dict(client_fraction=1.0, conditional_update_delta=0.0, quantize_updates=False, prune=False),
        "2. + Comm. reduction": dict(client_fraction=0.7, conditional_update_delta=1e-3, quantize_updates=True,  prune=False),
        "3. + Pruning":         dict(client_fraction=1.0, conditional_update_delta=0.0, quantize_updates=False, prune=True),
        "4. All combined":      dict(client_fraction=0.7, conditional_update_delta=1e-3, quantize_updates=True,  prune=True),
    }

    rows = []
    for name, cfg in configs.items():
        print(f"\n{'#'*60}\n  ABLATION CONFIG: {name}\n{'#'*60}")
        model, acc_hist, train_hist, stats = run_federated_learning(
            algorithm="fedprox",
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            device_names=device_names,
            client_fraction=cfg["client_fraction"],
            conditional_update_delta=cfg["conditional_update_delta"],
            quantize_updates=cfg["quantize_updates"],
        )

        final_acc = stats["final_test_accuracy"]
        if cfg["prune"]:
            model = prune_global_model(model, amount=0.3)
            model = fine_tune(model, train_loaders, epochs=2, lr=0.001)
            model = finalize_pruning(model)
            eval_result = evaluate_global_model(model, test_loaders, device_names, verbose=False)
            final_acc = eval_result["overall_accuracy"]

        bench = benchmark_model(model, input_size, label=name)
        kb_sent = stats["total_bytes_after"] / 1024 if stats["total_bytes_after"] else 0.0

        if name == "1. Baseline":
            baseline_bench = bench   # every later config compares its footprint reduction against this

        comparison = compare_model_footprints(baseline_bench, bench, label=name)

        rows.append({
            "Config": name,
            "Final Accuracy": round(final_acc, 4),
            "Rounds to Converge": stats["best_round"],
            "Total KB Sent": round(kb_sent, 1),
            "Total Params": bench["total_params"],
            "Non-zero Params": bench["nonzero_params"],
            "Model Size (KB)": round(bench["model_size_kb"], 2),
            "FLOPs": bench["flops"],
            "Latency (ms/sample)": round(bench["latency_ms"], 4),
            "Throughput (samples/sec)": round(bench["throughput_samples_per_sec"], 0),
            "Size Reduction vs Baseline (%)": round(comparison["size_reduction_pct"], 1),
            "Speedup vs Baseline (x)": round(comparison["speedup_factor"], 2),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*60}\n  ABLATION SUMMARY TABLE\n{'='*60}")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 8d. LEAVE-ONE-DEVICE-OUT GENERALIZATION TEST (Non-IID gap, cross-device)
# =============================================================================

def leave_one_device_out_eval(
    train_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    device_names: list[str],
    input_size: int,
    hidden_size: int = 32,
    num_classes: int = 2,
    algorithm: str = "fedprox",
    num_rounds: int = 8,
    local_epochs: int = 2,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    mu: float = 0.01,
) -> dict:
    """
    Leave-one-device-out (LODO) generalization test.

    For each device, trains a federated model using ONLY the other 8
    devices as clients, then evaluates on the held-out device's FULL data
    (its train + test loaders combined — none of it was used in training
    at all). This is a much stronger check than an in-device random split:
    it asks whether the model generalizes to a device it has never seen,
    which is the real claim behind "FedProx handles non-IID heterogeneity"
    — a within-device train/test split can't tell you that on its own.

    Returns {device_name: held_out_accuracy}.
    """
    results = {}
    num_devices = len(train_loaders)

    for held_out_idx in range(num_devices):
        held_out_name = device_names[held_out_idx]
        other_train_loaders = [train_loaders[i] for i in range(num_devices) if i != held_out_idx]
        other_test_loaders  = [test_loaders[i]  for i in range(num_devices) if i != held_out_idx]
        other_names = [d for i, d in enumerate(device_names) if i != held_out_idx]

        print(f"\n{'='*60}")
        print(f"  LEAVE-ONE-DEVICE-OUT: holding out {held_out_name} ({held_out_idx + 1}/{num_devices})")
        print(f"{'='*60}")

        model, _, _, _ = run_federated_learning(
            algorithm=algorithm,
            train_loaders=other_train_loaders,
            test_loaders=other_test_loaders,
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            device_names=other_names,
        )

        # Evaluate on the held-out device's FULL data — genuinely unseen.
        held_out_full = ConcatDataset([train_loaders[held_out_idx].dataset, test_loaders[held_out_idx].dataset])
        held_out_loader = DataLoader(held_out_full, batch_size=64, shuffle=False)

        eval_result = evaluate_global_model(model, [held_out_loader], [held_out_name], verbose=False)
        acc = eval_result["overall_accuracy"]
        results[held_out_name] = acc
        print(f"  >> {held_out_name} held-out accuracy (never seen in training): {acc:.4f}")

    print(f"\n{'='*60}")
    print("  LEAVE-ONE-DEVICE-OUT SUMMARY")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"    {name:<45} {acc:.4f}")
    avg_acc = sum(results.values()) / len(results)
    print(f"\n  Average held-out accuracy: {avg_acc:.4f}")
    print("  (Compare this to your in-device test accuracy. If it's close, the model")
    print("   genuinely generalizes across devices. If it's much lower, the model is")
    print("   fitting per-device quirks rather than transferable attack signatures.)")
    return results


# =============================================================================
# 9. PLOTTING CONVERGENCE
# =============================================================================

def plot_convergence(
    fedavg_history: list[float],
    fedprox_history: list[float],
    fedavg_train_history: list[float] | None = None,
    fedprox_train_history: list[float] | None = None,
    save_path: str = "convergence_comparison.png",
) -> Path:
    """
    Plot round-by-round convergence for FedAvg and FedProx.

    If train-accuracy histories are supplied, they're plotted as dashed
    lines alongside the test curves — the gap between dashed (train) and
    solid (test) is the actual overfitting signal. A flat test curve on
    its own does not show overfitting; it only shows the model reached a
    plateau, which can equally mean underfitting or a low learning rate.

    Saves the figure to disk so it can be inspected even when running in a
    headless terminal session.
    """
    rounds = range(1, max(len(fedavg_history), len(fedprox_history)) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(fedavg_history) + 1), fedavg_history, marker="o", linewidth=2, label="FedAvg (test)", color="tab:blue")
    plt.plot(range(1, len(fedprox_history) + 1), fedprox_history, marker="s", linewidth=2, label="FedProx (test)", color="tab:orange")
    if fedavg_train_history:
        plt.plot(range(1, len(fedavg_train_history) + 1), fedavg_train_history,
                  linestyle="--", linewidth=1.5, label="FedAvg (train)", color="tab:blue", alpha=0.6)
    if fedprox_train_history:
        plt.plot(range(1, len(fedprox_train_history) + 1), fedprox_train_history,
                  linestyle="--", linewidth=1.5, label="FedProx (train)", color="tab:orange", alpha=0.6)
    plt.title("Federated Convergence Comparison")
    plt.xlabel("Communication Round")
    plt.ylabel("Overall Accuracy")
    plt.xticks(list(rounds))
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_path = Path(save_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nConvergence plot saved to: {output_path.resolve()}")
    return output_path


# =============================================================================
# 10. MAIN — COMPARE FedAvg vs FedProx
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # STEP 1: Load the dataset.
    #
    # If the N-BaIoT folder exists on this computer, we use the real
    # data. Otherwise, we generate fake ("synthetic") data so the script
    # can still be tested without needing the real dataset downloaded.
    # ------------------------------------------------------------------
    use_real_data = DATASET_ROOT.exists()

    # How many of the 115 original features to keep after feature
    # selection. Fewer features = a smaller, faster ("lightweight") model,
    # but too few can hurt accuracy. This number should be chosen using
    # feature_sweep.py (a separate script), not guessed — run that script
    # once, look at its results table, then set this number here.
    K_BEST_FEATURES = 60

    # How to split each device's data into "train" and "test" portions:
    #   "random" — shuffle the rows randomly before splitting (default)
    #   "time"   — keep the original row order; first rows = train, last
    #              rows = test. Useful for checking that "random" isn't
    #              accidentally leaking very similar rows across the split.
    SPLIT_STRATEGY = "random"

    if use_real_data:
        print("Real N-BaIoT dataset found — loading...")
        train_loaders, test_loaders, scaler, selected_idx, device_names = load_nbaiot_federated(
            dataset_root=DATASET_ROOT,
            max_samples_per_device=5000,
            k_best_features=K_BEST_FEATURES,
            split_strategy=SPLIT_STRATEGY,
            check_leakage=True,
        )
        input_size = len(selected_idx)
    else:
        print(f"N-BaIoT dataset not found at '{DATASET_ROOT}'.")
        print("Running with FAKE (synthetic) data instead, just so the script can still run.\n")
        print(f"  To use the real dataset: put the N-BaIoT folder at '{DATASET_ROOT}'\n")
        train_loaders, test_loaders = make_synthetic_noniid_loaders(
            num_clients=9, input_size=115, samples_per_client=1000
        )
        device_names = [f"Device_{i+1}" for i in range(len(train_loaders))]
        input_size = 115

    num_clients = len(train_loaders)

    # ------------------------------------------------------------------
    # Shared hyperparameters
    #
    # client_fraction / conditional_update_delta / quantize_updates
    # implement the "Communication Cost Reduction" goal:
    #   - client_fraction < 1.0   -> only a sampled subset of clients
    #                                train + upload each round
    #                                (the "top 20 of 100 devices" idea —
    #                                 with 9 real devices this samples a
    #                                 subset of 9; see note below if you
    #                                 want a literal 100-client simulation)
    #   - conditional_update_delta -> a client's update is dropped if its
    #                                local loss didn't improve enough
    #   - quantize_updates         -> simulates fp16 compression on the
    #                                uplink before server aggregation
    # ------------------------------------------------------------------
    CONFIG = dict(
        input_size   = input_size,
        hidden_size  = 20,
        num_classes  = 2,
        num_rounds   = 10,
        local_epochs = 2,
        lr           = 0.001,
        weight_decay = 1e-4,
        client_fraction = 0.7,
        conditional_update_delta = 1e-3,
        quantize_updates = True,
    )

    # ------------------------------------------------------------------
    # Run FedAvg baseline
    # ------------------------------------------------------------------
    fedavg_model, fedavg_acc, fedavg_train_acc, fedavg_stats = run_federated_learning(
        algorithm="fedavg",
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        device_names=device_names,
        **CONFIG,
    )

    # ------------------------------------------------------------------
    # Run FedProx  (μ = 0.01 — good starting point for N-BaIoT)
    # ------------------------------------------------------------------
    fedprox_model, fedprox_acc, fedprox_train_acc, fedprox_stats = run_federated_learning(
        algorithm="fedprox",
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        device_names=device_names,
        mu=0.005,        # <-- tune this: try 0.001, 0.01, 0.05, 0.1
        **CONFIG,
    )

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  FINAL RESULTS — FedAvg vs FedProx")
    print("="*60)
    print(f"  FedAvg  final accuracy : {fedavg_acc[-1]:.4f}  (train: {fedavg_train_acc[-1]:.4f}, "
          f"gap: {fedavg_train_acc[-1] - fedavg_acc[-1]:+.4f})")
    print(f"  FedProx final accuracy : {fedprox_acc[-1]:.4f}  (train: {fedprox_train_acc[-1]:.4f}, "
          f"gap: {fedprox_train_acc[-1] - fedprox_acc[-1]:+.4f})")

    improvement = (fedprox_acc[-1] - fedavg_acc[-1]) * 100
    sign = "+" if improvement >= 0 else ""
    print(f"  FedProx improvement    : {sign}{improvement:.2f}%")
    print("\nRound-by-round accuracy:")
    print(f"  {'Round':<8} {'FedAvg':>10} {'FedProx':>10}")
    for r, (fa, fp) in enumerate(zip(fedavg_acc, fedprox_acc), 1):
        print(f"  {r:<8} {fa:>10.4f} {fp:>10.4f}")

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------
    plot_convergence(fedavg_acc, fedprox_acc, fedavg_train_acc, fedprox_train_acc)

    # ------------------------------------------------------------------
    # Detailed classification report on final global models
    # ------------------------------------------------------------------
    print("\n--- FedAvg — Classification Report (all test data) ---")
    r = evaluate_global_model(fedavg_model, test_loaders, device_names)
    print(classification_report(r["all_targets"], r["all_predictions"],
                                 target_names=["Benign", "Malicious"]))

    print("--- FedProx — Classification Report (all test data) ---")
    r = evaluate_global_model(fedprox_model, test_loaders, device_names)
    print(classification_report(r["all_targets"], r["all_predictions"],
                                 target_names=["Benign", "Malicious"]))

    # ------------------------------------------------------------------
    # Lightweight Model Design — prune the best (FedProx) model and
    # benchmark it against the unpruned version, per the slide's
    # "run inference on Raspberry Pi-class hardware" target.
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  MODEL PRUNING — Lightweight Model Design")
    print("="*60)
    benchmark_model(fedprox_model, input_size, label="FedProx (unpruned)")

    pruned_model = prune_global_model(fedprox_model, amount=0.3)
    pruned_model = fine_tune(pruned_model, train_loaders, epochs=2, lr=0.001)
    pruned_model = finalize_pruning(pruned_model)
    benchmark_model(pruned_model, input_size, label="FedProx (pruned 30%, fine-tuned)")

    print("\n--- Pruned FedProx — Classification Report (all test data) ---")
    r = evaluate_global_model(pruned_model, test_loaders, device_names)
    print(classification_report(r["all_targets"], r["all_predictions"],
                                 target_names=["Benign", "Malicious"]))

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------
    torch.save(fedavg_model.state_dict(),  "fedavg_model.pt")
    torch.save(fedprox_model.state_dict(), "fedprox_model.pt")
    torch.save(pruned_model.state_dict(),  "fedprox_model_pruned.pt")
    print("\nModels saved: fedavg_model.pt  |  fedprox_model.pt  |  fedprox_model_pruned.pt")

    # ------------------------------------------------------------------
    # Ablation table — one row per research gap (comm. reduction,
    # lightweight design, both combined), directly comparable numbers
    # for the dissertation results section.
    # ------------------------------------------------------------------
    ablation_df = run_ablation_study(
        train_loaders, test_loaders, device_names, input_size,
        hidden_size=CONFIG["hidden_size"],
        num_rounds=CONFIG["num_rounds"],
        local_epochs=CONFIG["local_epochs"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    ablation_df.to_csv("ablation_results.csv", index=False)
    print("\nAblation table saved to: ablation_results.csv")

    # ------------------------------------------------------------------
    # Leave-one-device-out generalization test — the real evidence for
    # the "FedProx handles non-IID heterogeneity" claim: does the model
    # transfer to a device it has never trained on at all?
    # ------------------------------------------------------------------
    lodo_results = leave_one_device_out_eval(
        train_loaders, test_loaders, device_names, input_size,
        hidden_size=CONFIG["hidden_size"],
        num_rounds=CONFIG["num_rounds"],
        local_epochs=CONFIG["local_epochs"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    pd.DataFrame(
        [{"device": k, "held_out_accuracy": v} for k, v in lodo_results.items()]
    ).to_csv("leave_one_device_out_results.csv", index=False)
    print("\nLODO results saved to: leave_one_device_out_results.csv")


if __name__ == "__main__":
    main()
