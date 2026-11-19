
# ============================================================
# TASK C: ODD-SYMMETRY VERIFICATION
# ============================================================

print("\n===== TASK C: ODD-SYMMETRY VERIFICATION =====")

# ------------------------------------------------------------
# 1. LOAD REFERENCE DATA
# ------------------------------------------------------------

with np.load(REFERENCE_PATH, allow_pickle=False) as data:

    x_ref = np.asarray(data["x"]).reshape(-1)
    t_ref = np.asarray(data["t"]).reshape(-1)
    usol_raw = np.asarray(data["usol"])

# Arrange reference solution as (time, space)
if usol_raw.shape == (len(x_ref), len(t_ref)):
    u_reference = usol_raw.T

elif usol_raw.shape == (len(t_ref), len(x_ref)):
    u_reference = usol_raw

else:
    raise ValueError(
        f"Unexpected reference solution shape: {usol_raw.shape}"
    )

# ------------------------------------------------------------
# 2. VERIFY SPATIAL GRID SYMMETRY
# ------------------------------------------------------------

grid_symmetry_error = np.max(
    np.abs(x_ref + x_ref[::-1])
)

print("\n===== GRID CHECK =====")
print("Spatial grid points:", len(x_ref))
print("Grid symmetry error:", grid_symmetry_error)

if grid_symmetry_error > 1e-8:
    raise ValueError(
        "Spatial grid is not symmetric. "
        "Reference symmetry comparison needs interpolation."
    )

# ------------------------------------------------------------
# 3. CREATE ORIGINAL AND MIRRORED COORDINATES
# ------------------------------------------------------------

T_grid, X_grid = np.meshgrid(
    t_ref, x_ref, indexing="ij"
)

# Original coordinates: [t, x]
tx_original = np.column_stack([
    T_grid.ravel(),
    X_grid.ravel()
])

# Mirrored coordinates: [t, -x]
tx_mirrored = np.column_stack([
    T_grid.ravel(),
    (-X_grid).ravel()
])

print("\nTotal coordinate pairs:", len(tx_original))

# ------------------------------------------------------------
# 4. PREDICT BOTH ORIGINAL AND MIRRORED SOLUTIONS
# ------------------------------------------------------------

model.eval()

prediction_original = []
prediction_mirrored = []

BATCH_SIZE = 4096

with torch.no_grad():

    for start in range(0, len(tx_original), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(tx_original))

        # Combine original and mirrored coordinates into one batch.
        tx_pair = np.concatenate([
            tx_original[start:end],
            tx_mirrored[start:end]
        ], axis=0)

        tx_tensor = torch.as_tensor(
            tx_pair,
            dtype=DTYPE,
            device=DEVICE
        )

        u_pair = model(tx_tensor).cpu().numpy().reshape(-1)

        n = end - start

        prediction_original.append(u_pair[:n])
        prediction_mirrored.append(u_pair[n:])

        print(f"Processed points: {end}/{len(tx_original)}")

u_original = np.concatenate(prediction_original).reshape(
    len(t_ref), len(x_ref)
)

u_mirrored = np.concatenate(prediction_mirrored).reshape(
    len(t_ref), len(x_ref)
)

# ------------------------------------------------------------
# 5. CALCULATE PINN ODD-SYMMETRY ERROR
# ------------------------------------------------------------

# Expected: u(x,t) + u(-x,t) = 0

symmetry_error_pinn = np.abs(
    u_original + u_mirrored
)

# ------------------------------------------------------------
# 6. CALCULATE REFERENCE ODD-SYMMETRY ERROR
# ------------------------------------------------------------

# Because the x-grid is symmetric, reversing the spatial
# array gives the values at the corresponding -x positions.

symmetry_error_reference = np.abs(
    u_reference + u_reference[:, ::-1]
)

# ------------------------------------------------------------
# 7. CALCULATE STATISTICS
# ------------------------------------------------------------

def symmetry_statistics(error):

    return {
        "maximum": np.max(error),
        "mean": np.mean(error),
        "rms": np.sqrt(np.mean(error ** 2)),
        "percentile_95": np.percentile(error, 95),
        "percentile_99": np.percentile(error, 99)
    }

pinn_stats = symmetry_statistics(symmetry_error_pinn)
reference_stats = symmetry_statistics(symmetry_error_reference)

print("\n===== PINN SYMMETRY RESULTS =====")

print(f"Maximum error: {pinn_stats['maximum']:.8e}")
print(f"Mean error:    {pinn_stats['mean']:.8e}")
print(f"RMS error:     {pinn_stats['rms']:.8e}")
print(f"95th percentile: {pinn_stats['percentile_95']:.8e}")
print(f"99th percentile: {pinn_stats['percentile_99']:.8e}")

print("\n===== REFERENCE SYMMETRY RESULTS =====")

print(f"Maximum error: {reference_stats['maximum']:.8e}")
print(f"Mean error:    {reference_stats['mean']:.8e}")
print(f"RMS error:     {reference_stats['rms']:.8e}")

# ------------------------------------------------------------
# 8. FIND LOCATION OF MAXIMUM PINN SYMMETRY ERROR
# ------------------------------------------------------------

max_index = np.unravel_index(
    np.argmax(symmetry_error_pinn),
    symmetry_error_pinn.shape
)

max_symmetry_time = t_ref[max_index[0]]
max_symmetry_position = x_ref[max_index[1]]

print("\n===== MAXIMUM PINN SYMMETRY VIOLATION =====")
print(f"Time:     {max_symmetry_time:.6f}")
print(f"Position: {max_symmetry_position:.6f}")

# ------------------------------------------------------------
# 9. PLOT PINN SYMMETRY ERROR
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.imshow(
    symmetry_error_pinn,
    origin="lower",
    aspect="auto",
    extent=[
        x_ref.min(), x_ref.max(),
        t_ref.min(), t_ref.max()
    ]
)

plt.colorbar(label="Odd-Symmetry Error")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Task C: Trained Burgers PINN Odd-Symmetry Error")

plt.tight_layout()

symmetry_plot_path = (
    RESULTS_DIR / "B1_TaskC_Odd_Symmetry.png"
)

plt.savefig(symmetry_plot_path, dpi=200)
plt.show()
plt.close()

# ------------------------------------------------------------
# 10. SAVE NUMERICAL RESULTS
# ------------------------------------------------------------

np.savez_compressed(
    RESULTS_DIR / "B1_TaskC_Odd_Symmetry.npz",
    x=x_ref,
    t=t_ref,
    symmetry_error_pinn=symmetry_error_pinn,
    symmetry_error_reference=symmetry_error_reference,
    u_original=u_original,
    u_mirrored=u_mirrored
)

print("\nSaved Task C results to:", RESULTS_DIR)
print("\n===== TASK C COMPLETED =====")
