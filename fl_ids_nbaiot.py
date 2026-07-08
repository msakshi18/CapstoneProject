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
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
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
    return combined


def load_nbaiot_federated(
    dataset_root: Path = DATASET_ROOT,
    max_samples_per_device: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
    k_best_features: int | None = 40,
) -> tuple[list[DataLoader], list[DataLoader], StandardScaler, list[int]]:
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
                    df = df.sample(n=max_samples_per_device, random_state=random_state)
                X_dev = df.iloc[:, :-1].values.astype(np.float32)
                y_dev = df["label"].values.astype(np.int64)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_dev,
                    y_dev,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_dev,
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
                    df = df.sample(n=max_samples_per_device, random_state=random_state)
                X_dev = df.iloc[:, :-1].values.astype(np.float32)
                y_dev = df["label"].values.astype(np.int64)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_dev,
                    y_dev,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_dev,
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
    return train_loaders, test_loaders, scaler, selected_idx


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
# 4. CLIENT TRAINING — FedAvg
# =============================================================================

def client_update_fedavg(
    client_model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epochs: int = 2,
) -> dict:
    """Standard local training for FedAvg — no proximal term."""
    client_model.train()
    criterion = nn.CrossEntropyLoss()
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

    Args:
        mu: proximal coefficient (0 = pure FedAvg; higher = tighter anchor)
            Typical range: 0.001 – 0.1. Start with 0.01 for N-BaIoT.
    """
    client_model.train()
    criterion = nn.CrossEntropyLoss()

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
            print(f"    {name:<45} Acc: {acc:.4f}")

    overall_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    return {
        "overall_accuracy": overall_acc,
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
    random_state: int = 42,
) -> tuple[nn.Module, list[float]]:
    """
    Main federated learning loop.

    FedAvg:   standard averaging — all clients train freely
    FedProx:  adds proximal term μ/2 ||w - w_global||² to each client's loss

    Returns the trained global model and round-by-round accuracy history.
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
    best_val_accuracy = float("-inf")
    best_model_state = copy.deepcopy(global_model.state_dict())
    stale_rounds = 0

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

            if algorithm == "fedavg":
                weights, loss_before, loss_after = client_update_fedavg(
                    client_model, optimizer, train_loaders[i], epochs=local_epochs
                )
            else:  # fedprox
                weights, loss_before, loss_after = client_update_fedprox(
                    client_model, global_model, optimizer, train_loaders[i],
                    epochs=local_epochs, mu=mu
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

        if not client_weights:
            print("  No client updates this round — keeping previous global model.")
        else:
            # Aggregate on server
            global_model = server_aggregate(global_model, client_weights, client_sizes)

        # Evaluate after this round
        print(f"\nRound {round_idx + 1}/{num_rounds} — Validation accuracy:")
        val_results = evaluate_global_model(global_model, val_loaders, device_names)
        val_accuracy = val_results["overall_accuracy"]

        if val_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(global_model.state_dict())
            stale_rounds = 0
        else:
            stale_rounds += 1

        print(f"\nRound {round_idx + 1}/{num_rounds} — Per-client accuracy:")
        results = evaluate_global_model(global_model, test_loaders, device_names)
        overall = results["overall_accuracy"]
        accuracy_history.append(overall)
        print(f"  >> Overall accuracy: {overall:.4f}")

        if stale_rounds >= patience:
            print(
                f"\nEarly stopping triggered after {round_idx + 1} rounds: "
                f"validation accuracy did not improve for {patience} rounds."
            )
            break

    global_model.load_state_dict(best_model_state)

    return global_model, accuracy_history

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


def benchmark_model(model: nn.Module, input_size: int, label: str = "") -> dict:
    """
    Reports parameter count, non-zero parameter count (post-pruning), and
    an approximate CPU inference latency per sample — a proxy for
    Raspberry-Pi-class edge feasibility (no GPU available on such hardware).
    """
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())

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

    print(f"  [{label}] Params: {total_params:,}  |  Non-zero: {nonzero_params:,} "
          f"({100 * nonzero_params / total_params:.1f}%)  |  Avg CPU latency: {elapsed_ms:.3f} ms/sample")
    return {"total_params": total_params, "nonzero_params": nonzero_params, "latency_ms": elapsed_ms}


# =============================================================================
# 9. PLOTTING CONVERGENCE
# =============================================================================

def plot_convergence(
    fedavg_history: list[float],
    fedprox_history: list[float],
    save_path: str = "convergence_comparison.png",
) -> Path:
    """
    Plot round-by-round convergence for FedAvg and FedProx.

    Saves the figure to disk so it can be inspected even when running in a
    headless terminal session.
    """
    rounds = range(1, max(len(fedavg_history), len(fedprox_history)) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(fedavg_history) + 1), fedavg_history, marker="o", linewidth=2, label="FedAvg")
    plt.plot(range(1, len(fedprox_history) + 1), fedprox_history, marker="s", linewidth=2, label="FedProx")
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
    # Dataset loading
    # ------------------------------------------------------------------
    use_real_data = DATASET_ROOT.exists()

    # k_best_features implements the "Lightweight Model Design" feature-selection
    # goal: 115 raw N-BaIoT features -> a smaller chi-squared-selected subset.
    K_BEST_FEATURES = 40

    if use_real_data:
        print("Real N-BaIoT dataset found — loading...")
        train_loaders, test_loaders, scaler, selected_idx = load_nbaiot_federated(
            dataset_root=DATASET_ROOT,
            max_samples_per_device=5000,
            k_best_features=K_BEST_FEATURES,
        )
        # Build device name list matching the loaded order
        device_names = [
            d for d in DEVICE_FOLDERS if (DATASET_ROOT / d).exists()
        ]
        input_size = len(selected_idx)
    else:
        print("N-BaIoT dataset not found at", DATASET_ROOT)
        print("Running with SYNTHETIC Non-IID data for development.\n")
        print("To use real data:")
        print("  1. Download from https://archive.ics.uci.edu/ml/datasets/")
        print("     detection_of_IoT_botnet_attacks_N_BaIoT")
        print(f"  2. Set DATASET_ROOT = Path('<your_path>') at top of this file\n")
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
        hidden_size  = 32,
        num_classes  = 2,
        num_rounds   = 8,
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
    fedavg_model, fedavg_acc = run_federated_learning(
        algorithm="fedavg",
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        device_names=device_names,
        **CONFIG,
    )

    # ------------------------------------------------------------------
    # Run FedProx  (μ = 0.01 — good starting point for N-BaIoT)
    # ------------------------------------------------------------------
    fedprox_model, fedprox_acc = run_federated_learning(
        algorithm="fedprox",
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        device_names=device_names,
        mu=0.001,        # <-- tune this: try 0.001, 0.01, 0.05, 0.1
        **CONFIG,
    )

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  FINAL RESULTS — FedAvg vs FedProx")
    print("="*60)
    print(f"  FedAvg  final accuracy : {fedavg_acc[-1]:.4f}")
    print(f"  FedProx final accuracy : {fedprox_acc[-1]:.4f}")

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
    plot_convergence(fedavg_acc, fedprox_acc)

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


if __name__ == "__main__":
    main()
