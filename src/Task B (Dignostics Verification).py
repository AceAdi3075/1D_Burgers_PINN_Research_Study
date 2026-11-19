
# ============================================================
# TASK B: TRAINED PINN VS REFERENCE SOLUTION
# ============================================================

print("\n===== TASK B: REFERENCE SOLUTION COMPARISON =====")

# ------------------------------------------------------------
# 1. LOAD REFERENCE SOLUTION
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

print("Reference solution shape:", u_reference.shape)

# ------------------------------------------------------------
# 2. CREATE REFERENCE EVALUATION GRID
# ------------------------------------------------------------

T_grid, X_grid = np.meshgrid(
    t_ref, x_ref, indexing="ij"
)

tx_eval = np.column_stack([
    T_grid.ravel(),
    X_grid.ravel()
])

print("Total evaluation points:", len(tx_eval))

# ------------------------------------------------------------
# 3. PREDICT USING TRAINED MODEL
# ------------------------------------------------------------

model.eval()

predictions = []
PREDICTION_BATCH_SIZE = 8192

with torch.no_grad():

    for start in range(
        0, len(tx_eval), PREDICTION_BATCH_SIZE
    ):

        end = min(
            start + PREDICTION_BATCH_SIZE,
            len(tx_eval)
        )

        tx_batch = torch.as_tensor(
            tx_eval[start:end],
            dtype=DTYPE,
            device=DEVICE
        )

        u_batch = model(tx_batch)

        predictions.append(
            u_batch.cpu().numpy().reshape(-1)
        )

        print(f"Predicted points: {end}/{len(tx_eval)}")

u_prediction = np.concatenate(
    predictions
).reshape(len(t_ref), len(x_ref))

# ------------------------------------------------------------
# 4. CALCULATE REFERENCE ERRORS
# ------------------------------------------------------------

error = u_prediction - u_reference
absolute_error = np.abs(error)

reference_l2 = np.linalg.norm(u_reference.ravel())
reference_linf = np.max(np.abs(u_reference))

if reference_l2 == 0 or reference_linf == 0:
    raise ValueError("Invalid reference norm.")

relative_l2 = (
    np.linalg.norm(error.ravel()) / reference_l2
)

relative_linf = (
    np.max(absolute_error) / reference_linf
)

max_absolute_error = np.max(absolute_error)

rmse = np.sqrt(np.mean(error ** 2))

# ------------------------------------------------------------
# 5. PRINT ERROR STATISTICS
# ------------------------------------------------------------

print("\n===== REFERENCE ERROR STATISTICS =====")

print(f"Relative L2 error:      {relative_l2:.8e}")
print(f"Relative Linf error:    {relative_linf:.8e}")
print(f"Maximum absolute error: {max_absolute_error:.8e}")
print(f"RMSE:                   {rmse:.8e}")

# ------------------------------------------------------------
# 6. COMPARE FINAL-TIME SPATIAL PROFILES
# ------------------------------------------------------------

final_index = int(np.argmax(t_ref))

u_ref_final = u_reference[final_index]
u_pred_final = u_prediction[final_index]

plt.figure(figsize=(9, 5))

plt.plot(
    x_ref,
    u_ref_final,
    label="Reference solution"
)

plt.plot(
    x_ref,
    u_pred_final,
    "--",
    label="Trained PINN prediction"
)

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(
    f"Reference vs PINN at t = {t_ref[final_index]:.2f}"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

profile_path = (
    RESULTS_DIR / "B1_TaskB_Final_Time_Profile.png"
)

plt.savefig(profile_path, dpi=200)
plt.show()
plt.close()

# ------------------------------------------------------------
# 7. ABSOLUTE ERROR FIELD
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.imshow(
    absolute_error,
    origin="lower",
    aspect="auto",
    extent=[
        x_ref.min(), x_ref.max(),
        t_ref.min(), t_ref.max()
    ]
)

plt.colorbar(label="Absolute error")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Burgers PINN: Absolute Error vs Reference")

plt.tight_layout()

error_path = (
    RESULTS_DIR / "B1_TaskB_Absolute_Error.png"
)

plt.savefig(error_path, dpi=200)
plt.show()
plt.close()

# ------------------------------------------------------------
# 8. ESTIMATE STEEPEST-GRADIENT LOCATIONS
# ------------------------------------------------------------

# Use the central domain to avoid interpreting boundary
# gradients as the interior steep front.

interior_mask = (
    (x_ref >= -0.8) & (x_ref <= 0.8)
)

reference_gradient = np.gradient(
    u_ref_final, x_ref
)

prediction_gradient = np.gradient(
    u_pred_final, x_ref
)

ref_candidates = np.where(
    interior_mask
)[0]

ref_front_index = ref_candidates[
    np.argmax(np.abs(reference_gradient[interior_mask]))
]

pred_front_index = ref_candidates[
    np.argmax(np.abs(prediction_gradient[interior_mask]))
]

reference_front_x = x_ref[ref_front_index]
prediction_front_x = x_ref[pred_front_index]

front_displacement = (
    prediction_front_x - reference_front_x
)

print("\n===== STEEPEST-GRADIENT COMPARISON =====")

print(f"Final time:             {t_ref[final_index]:.4f}")
print(f"Reference location:     {reference_front_x:.6f}")
print(f"PINN location:          {prediction_front_x:.6f}")
print(f"Signed displacement:    {front_displacement:.6f}")
print(f"Absolute displacement:  {abs(front_displacement):.6f}")

# ------------------------------------------------------------
# 9. SAVE NUMERICAL RESULTS
# ------------------------------------------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    RESULTS_DIR / "B1_TaskB_Reference_Comparison.npz",
    x=x_ref,
    t=t_ref,
    u_reference=u_reference,
    u_prediction=u_prediction,
    error=error,
    absolute_error=absolute_error
)

print("\nSaved Task B results to:", RESULTS_DIR)
print("\n===== TASK B COMPLETED =====")
