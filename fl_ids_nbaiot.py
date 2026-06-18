"""
Federated Learning Intrusion Detection System — N-BaIoT Dataset
================================================================
Implements FedAvg (baseline) and FedProx (Non-IID optimisation)
as outlined in: "Optimising Federated Learning Based Intrusion Detection
for IoT Systems" — Sakshi Mahajan, MSc CS (Data Analytics), Uni. of Galway

FedProx Reference:
  Li et al. (2020). "Federated Optimization in Heterogeneous Networks."
  MLSys 2020. https://arxiv.org/abs/1812.06127
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


# =============================================================================
# 1. DATASET LOADING — N-BaIoT
# =============================================================================
# N-BaIoT dataset structure:
#   Each IoT device has its own folder (e.g. Danmini_Doorbell, Ecobee_Thermostat…)
#   Each folder contains:
#       benign/           — normal traffic CSVs
#       mirai_attacks/    — Mirai botnet attack CSVs  (gafgyt_attacks/ for Gafgyt)
#       gafgyt_attacks/   — Gafgyt botnet attack CSVs
#
# Download from: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT
# or Kaggle: https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset
#
# Set DATASET_ROOT below to your local path:
DATASET_ROOT = Path(".Datasets/N-BaIoT")

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

    # --- Benign traffic ---
    benign_dir = device_path / "benign"
    if benign_dir.exists():
        for csv_file in benign_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            df["label"] = 0
            dfs.append(df)

    # --- Attack traffic (Mirai) ---
    mirai_dir = device_path / "mirai_attacks"
    if mirai_dir.exists():
        for csv_file in mirai_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            df["label"] = 1
            dfs.append(df)

    # --- Attack traffic (Gafgyt) ---
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


def load_nbaiot_federated(
    dataset_root: Path = DATASET_ROOT,
    max_samples_per_device: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[DataLoader], list[DataLoader], StandardScaler]:
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
    all_data = []
    client_indices = []  # track which rows belong to which device
    idx = 0

    available_devices = []
    for device_name in DEVICE_FOLDERS:
        device_path = dataset_root / device_name
        if device_path.exists():
            available_devices.append(device_name)
            try:
                df = load_device_data(device_path)
                # Cap samples per device to keep training manageable
                if len(df) > max_samples_per_device:
                    df = df.sample(n=max_samples_per_device, random_state=random_state)
                start = idx
                all_data.append(df)
                idx += len(df)
                client_indices.append((device_name, start, idx))
                print(f"  Loaded {device_name}: {len(df):,} samples "
                      f"({(df['label'] == 0).sum():,} benign, "
                      f"{(df['label'] == 1).sum():,} attack)")
            except Exception as e:
                print(f"  WARNING: Could not load {device_name}: {e}")

    if not all_data:
        raise RuntimeError(
            f"No N-BaIoT data found at '{dataset_root}'.\n"
            "Please download the dataset and set DATASET_ROOT correctly."
        )

    full_df = pd.concat(all_data, ignore_index=True)
    X = full_df.iloc[:, :-1].values.astype(np.float32)  # 115 features
    y = full_df["label"].values.astype(np.int64)

    # Fit scaler on ALL data (global normalisation)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    train_loaders, test_loaders = [], []

    for device_name, start, end in client_indices:
        X_dev = X[start:end]
        y_dev = y[start:end]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_dev, y_dev, test_size=test_size, random_state=random_state, stratify=y_dev
        )

        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        test_ds  = TensorDataset(torch.tensor(X_te),  torch.tensor(y_te))

        train_loaders.append(DataLoader(train_ds, batch_size=64, shuffle=True))
        test_loaders.append(DataLoader(test_ds,  batch_size=64, shuffle=False))

    print(f"\nTotal clients (devices): {len(train_loaders)}")
    print(f"Feature dimensions    : {X.shape[1]}")
    return train_loaders, test_loaders, scaler


# =============================================================================
# 2. DEMO / SYNTHETIC DATA (used when N-BaIoT is not yet downloaded)
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

    # Vary attack ratio per device (Non-IID: different distributions)
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

    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(client_model(data), target)
            loss.backward()
            optimizer.step()

    return client_model.state_dict()


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

    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()

            # --- Task loss (cross-entropy) ---
            task_loss = criterion(client_model(data), target)

            # --- Proximal term: penalise drift from global model ---
            # ||w_local - w_global||^2  (summed across all parameters)
            prox_loss = torch.tensor(0.0)
            for name, param in client_model.named_parameters():
                prox_loss = prox_loss + torch.norm(param - global_params[name]) ** 2

            loss = task_loss + (mu / 2.0) * prox_loss
            loss.backward()
            optimizer.step()

    return client_model.state_dict()


# =============================================================================
# 6. SERVER AGGREGATION — FedAvg (same for both algorithms)
# =============================================================================

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
    device_names: list[str] | None = None,
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
    print(f"{'='*60}")

    global_model = SimpleMLP(input_size, hidden_size, num_classes)
    accuracy_history = []

    for round_idx in range(num_rounds):
        client_weights = []
        client_sizes = []

        for i in range(num_clients):
            client_model = copy.deepcopy(global_model)
            optimizer    = optim.SGD(
                client_model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            client_sizes.append(len(train_loaders[i].dataset))

            if algorithm == "fedavg":
                weights = client_update_fedavg(
                    client_model, optimizer, train_loaders[i], epochs=local_epochs
                )
            else:  # fedprox
                weights = client_update_fedprox(
                    client_model, global_model, optimizer, train_loaders[i],
                    epochs=local_epochs, mu=mu
                )

            client_weights.append(weights)

        # Aggregate on server
        global_model = server_aggregate(global_model, client_weights, client_sizes)

        # Evaluate after this round
        print(f"\nRound {round_idx + 1}/{num_rounds} — Per-client accuracy:")
        results = evaluate_global_model(global_model, test_loaders, device_names)
        overall = results["overall_accuracy"]
        accuracy_history.append(overall)
        print(f"  >> Overall accuracy: {overall:.4f}")

    return global_model, accuracy_history


# =============================================================================
# 9. MAIN — COMPARE FedAvg vs FedProx
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # Dataset loading
    # Try real N-BaIoT first; fall back to synthetic Non-IID data
    # ------------------------------------------------------------------
    use_real_data = DATASET_ROOT.exists()

    if use_real_data:
        print("Real N-BaIoT dataset found — loading...")
        train_loaders, test_loaders, scaler = load_nbaiot_federated(
            dataset_root=DATASET_ROOT,
            max_samples_per_device=5000,
        )
        # Build device name list matching the loaded order
        device_names = [
            d for d in DEVICE_FOLDERS if (DATASET_ROOT / d).exists()
        ]
        input_size = 115
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
    # ------------------------------------------------------------------
    CONFIG = dict(
        input_size   = input_size,
        hidden_size  = 32,
        num_classes  = 2,
        num_rounds   = 10,
        local_epochs = 1,
        lr           = 0.005,
        weight_decay = 1e-4,
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
        mu=0.01,        # <-- tune this: try 0.001, 0.01, 0.05, 0.1
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
    # Save models
    # ------------------------------------------------------------------
    torch.save(fedavg_model.state_dict(),  "fedavg_model.pt")
    torch.save(fedprox_model.state_dict(), "fedprox_model.pt")
    print("\nModels saved: fedavg_model.pt  |  fedprox_model.pt")


if __name__ == "__main__":
    main()
