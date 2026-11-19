
# ============================================================
# TASK A: TRAINED MODEL PDE RESIDUAL VERIFICATION
# ============================================================

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

print("\n===== TASK A: TRAINED MODEL RESIDUAL =====")

# ------------------------------------------------------------
# 1. CHECK REQUIRED OBJECTS
# ------------------------------------------------------------

assert "model" in globals(), "Trained model has not been loaded."
assert "burgers_residual_eval" in globals(), \
    "Diagnostic residual function is missing."

print("Trained model:", type(model).__name__)
print("Device:", DEVICE)
print("Data type:", DTYPE)

# ------------------------------------------------------------
# 2. LOAD REFERENCE GRID COORDINATES
# ------------------------------------------------------------

# Try the configured path first, followed by known locations.
candidate_paths = []

if "REFERENCE_PATH" in globals():
    candidate_paths.append(Path(REFERENCE_PATH))

candidate_paths.extend([
    Path("/content/drive/MyDrive/1D_Burgers_PINN_Research_Study/reference_data/Burgers.npz"),
    Path("/content/reference_data/Burgers.npz")
])

reference_file = next(
    (p for p in candidate_paths if p.exists()),
    None
)

if reference_file is None:
    raise FileNotFoundError(
        "Burgers.npz not found in the configured or fallback locations."
    )

with np.load(reference_file, allow_pickle=False) as data:
    x_ref = np.asarray(data["x"]).reshape(-1)
    t_ref = np.asarray(data["t"]).reshape(-1)

print("\nReference coordinate file:", reference_file)
print("Number of x coordinates:", len(x_ref))
print("Number of t coordinates:", len(t_ref))

# Create a grid containing every (t, x) combination.
T_grid, X_grid = np.meshgrid(t_ref, x_ref, indexing="ij")

# Network input order must be [t, x].
tx_all = np.column_stack([
    T_grid.ravel(),
    X_grid.ravel()
])

print("Total evaluation points:", len(tx_all))

# ------------------------------------------------------------
# 3. CALCULATE RESIDUAL IN BATCHES
# ------------------------------------------------------------

BATCH_SIZE = 4096

residual_batches = []

model.eval()

# IMPORTANT:
# Do not use torch.no_grad() here.
# Autograd needs to calculate spatial and temporal derivatives.

for start in range(0, len(tx_all), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(tx_all))

    tx_batch = torch.as_tensor(
        tx_all[start:end],
        dtype=DTYPE,
        device=DEVICE
    )

    # Calculate residual using our self-contained function.
    residual_batch = burgers_residual_eval(model, tx_batch)

    residual_batches.append(
        residual_batch.detach().cpu().numpy().reshape(-1)
    )

    del tx_batch, residual_batch

    print(
        f"Processed points: {end}/{len(tx_all)}"
    )

# Combine all batches.
residual_flat = np.concatenate(residual_batches)

# Restore the original grid structure: (time, space).
residual_field = residual_flat.reshape(
    len(t_ref), len(x_ref)
)

# ------------------------------------------------------------
# 4. CHECK NUMERICAL VALIDITY
# ------------------------------------------------------------

if not np.all(np.isfinite(residual_field)):
    raise ValueError(
        "Residual contains NaN or infinite values."
    )

absolute_residual = np.abs(residual_field)

# ------------------------------------------------------------
# 5. CALCULATE RESIDUAL STATISTICS
# ------------------------------------------------------------

maximum_residual = np.max(absolute_residual)
mean_absolute_residual = np.mean(absolute_residual)
mean_signed_residual = np.mean(residual_field)

rms_residual = np.sqrt(
    np.mean(residual_field ** 2)
)

percentiles = np.percentile(
    absolute_residual,
    [50, 90, 95, 99]
)

max_index = np.unravel_index(
    np.argmax(absolute_residual),
    absolute_residual.shape
)

max_time = t_ref[max_index[0]]
max_space = x_ref[max_index[1]]

print("\n===== TRAINED MODEL RESIDUAL RESULTS =====")

print(f"Maximum |R|:       {maximum_residual:.8e}")
print(f"Mean |R|:          {mean_absolute_residual:.8e}")
print(f"Mean signed R:     {mean_signed_residual:.8e}")
print(f"RMS residual:      {rms_residual:.8e}")

print("\nResidual percentiles:")
print(f"50th:              {percentiles[0]:.8e}")
print(f"90th:              {percentiles[1]:.8e}")
print(f"95th:              {percentiles[2]:.8e}")
print(f"99th:              {percentiles[3]:.8e}")

print("\nLocation of maximum |R|:")
print(f"Time t:            {max_time:.6f}")
print(f"Position x:        {max_space:.6f}")

print("\nFinite residual values:", np.isfinite(residual_field).all())

# ------------------------------------------------------------
# 6. SAVE RESULTS
# ------------------------------------------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    RESULTS_DIR / "B1_TaskA_Trained_Model_Residual.npz",
    x=x_ref,
    t=t_ref,
    residual=residual_field,
    absolute_residual=absolute_residual
)

# ------------------------------------------------------------
# 7. PLOT RESIDUAL DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.imshow(
    absolute_residual,
    origin="lower",
    aspect="auto",
    extent=[
        x_ref.min(), x_ref.max(),
        t_ref.min(), t_ref.max()
    ]
)

plt.colorbar(label="Absolute PDE Residual |R|")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Task A: Trained Burgers PINN PDE Residual")

plt.tight_layout()

plot_path = RESULTS_DIR / "B1_TaskA_Trained_Model_Residual.png"
plt.savefig(plot_path, dpi=200)
plt.show()
plt.close()

print("\nSaved residual data and plot in:", RESULTS_DIR)
print("\n===== TASK A COMPLETED =====")
