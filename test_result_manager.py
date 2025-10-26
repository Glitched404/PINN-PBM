"""Simple test to verify ResultManager functionality."""
import os
import numpy as np
import tensorflow as tf
from pinn_pbm.core.utils import ResultManager

print("="*60)
print("TESTING RESULTMANAGER - STEP 2 VALIDATION")
print("="*60)

# Test 1: Create ResultManager
print("\n1. Creating ResultManager...")
rm = ResultManager(
    problem_type='breakage',
    case_name='test_case',
    base_dir='results'
)
print(f"   ✓ Created: {rm}")
print(f"   ✓ Save directory: {rm.save_dir}")

# Test 2: Check directories were created
print("\n2. Checking directories...")
assert os.path.exists(rm.save_dir), "Main directory not created"
assert os.path.exists(os.path.join(rm.save_dir, "plots")), "Plots directory not created"
print("   ✓ Directories created successfully")

# Test 3: Save and load loss history
print("\n3. Testing loss history save/load...")
train_loss = [1.0, 0.5, 0.25, 0.1]
data_loss = [0.8, 0.4, 0.2, 0.08]
physics_loss = [0.2, 0.1, 0.05, 0.02]

rm.save_loss_history(train_loss, data_loss, physics_loss)
loaded_losses = rm.load_loss_history()

assert np.allclose(loaded_losses['train_loss'], train_loss), "Train loss mismatch"
assert np.allclose(loaded_losses['data_loss'], data_loss), "Data loss mismatch"
assert np.allclose(loaded_losses['physics_loss'], physics_loss), "Physics loss mismatch"
print("   ✓ Loss history saved and loaded correctly")

# Test 4: Save and load predictions
print("\n4. Testing predictions save/load...")
v_points = np.logspace(-3, 1, 50)
t_points = np.array([0.0, 2.0, 5.0, 10.0])
predictions = np.random.rand(4, 50)
exact = np.random.rand(4, 50)

rm.save_predictions(v_points, t_points, predictions, exact)
loaded_preds = rm.load_predictions()

assert np.allclose(loaded_preds['v_points'], v_points), "v_points mismatch"
assert np.allclose(loaded_preds['t_points'], t_points), "t_points mismatch"
assert np.allclose(loaded_preds['predictions'], predictions), "Predictions mismatch"
assert np.allclose(loaded_preds['exact_solutions'], exact), "Exact solutions mismatch"
print("   ✓ Predictions saved and loaded correctly")

# Test 5: Save and load metadata
print("\n5. Testing metadata save/load...")
config = {
    'case_type': 'case1',
    'v_min': 0.001,
    'v_max': 10.0,
    'epochs': 3000,
    'learning_rate': 0.0005
}
metrics = {
    'final_train_loss': 1.23e-4,
    'final_data_loss': 5.67e-5,
    'mean_relative_error': 0.012
}

rm.save_metadata(config, metrics)
loaded_metadata = rm.load_metadata()

assert loaded_metadata['config'] == config, "Config mismatch"
assert loaded_metadata['metrics'] == metrics, "Metrics mismatch"
assert loaded_metadata['timestamp'] == rm.timestamp, "Timestamp mismatch"
print("   ✓ Metadata saved and loaded correctly")

# Test 6: Create a simple model and test save/load
print("\n6. Testing model save/load...")
# Create a simple test model
test_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])
test_model.build((None, 2))

# Save model
rm.save_model(test_model)
print("   ✓ Model weights saved")

# Create new model with same architecture
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])
new_model.build((None, 2))

# Load weights
rm.load_model(new_model)
print("   ✓ Model weights loaded")

# Verify weights match
for orig_w, loaded_w in zip(test_model.get_weights(), new_model.get_weights()):
    assert np.allclose(orig_w, loaded_w), "Model weights mismatch"
print("   ✓ Model weights match after reload")

# Test 7: List saved files
print("\n7. Listing saved files...")
files = rm.list_saved_files()
print(f"   ✓ Found {len(files)} saved files:")
for f in files:
    print(f"     - {f}")

# Test 8: Get plot path
print("\n8. Testing plot path generation...")
plot_path = rm.get_plot_path('test_plot.png')
expected = os.path.join(rm.save_dir, "plots", "test_plot.png")
assert plot_path == expected, "Plot path incorrect"
print(f"   ✓ Plot path: {plot_path}")

# Test 9: List all runs
print("\n9. Testing list_all_runs...")
all_runs = ResultManager.list_all_runs('breakage', 'test_case')
assert rm.timestamp in all_runs, "Current run not in list"
print(f"   ✓ Found {len(all_runs)} run(s) for breakage/test_case")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - ResultManager is working perfectly!")
print("="*60)
print(f"\nTest results saved in: {rm.save_dir}")
print("You can safely delete this test directory when done.")
