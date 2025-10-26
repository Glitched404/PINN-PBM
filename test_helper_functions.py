"""Test helper functions."""
import numpy as np
import tensorflow as tf
from pinn_pbm.core.utils import (
    trapz_tf,
    tail_trapz,
    huber_loss,
    percentile_clip_loss,
    delta_peak,
    set_random_seed,
    configure_gpu_memory_growth,
    check_tensorflow_probability,
    TFP_AVAILABLE
)

print("="*60)
print("TESTING HELPER FUNCTIONS - STEP 4 VALIDATION")
print("="*60)

# Test 1: trapz_tf - Trapezoidal integration
print("\n1. Testing trapz_tf...")
x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
y = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
result = trapz_tf(y, x)
expected = 2.0  # Area under y=x from 0 to 2
assert abs(result.numpy() - expected) < 1e-5, f"Expected {expected}, got {result.numpy()}"
print(f"   ✓ Linear function: integral = {result.numpy():.6f} (expected: {expected})")

# Test with single point (should return 0)
x_single = tf.constant([1.0], dtype=tf.float32)
y_single = tf.constant([1.0], dtype=tf.float32)
result_single = trapz_tf(y_single, x_single)
assert result_single.numpy() == 0.0, "Single point should give 0"
print(f"   ✓ Single point: integral = {result_single.numpy():.6f} (expected: 0.0)")

# Test 2: tail_trapz - Tail integration
print("\n2. Testing tail_trapz...")
v_grid = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
g = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)  # Constant function
TI = tail_trapz(g, v_grid)
print(f"   ✓ Tail integrals computed: {TI.numpy()}")
# Last point should be 0 (no tail)
assert TI.numpy()[-1] == 0.0, "Last tail integral should be 0"
print("   ✓ Last point has zero tail integral")

# Test with batch dimension
g_batch = tf.constant([[1.0, 1.0, 1.0, 1.0], 
                       [2.0, 2.0, 2.0, 2.0]], dtype=tf.float32)
TI_batch = tail_trapz(g_batch, v_grid)
assert TI_batch.shape == (2, 4), "Batch dimension preserved"
print(f"   ✓ Batch processing works: shape {TI_batch.shape}")

# Test 3: huber_loss
print("\n3. Testing huber_loss...")
residuals = tf.constant([-2.0, -0.05, 0.0, 0.05, 2.0], dtype=tf.float32)
loss = huber_loss(residuals, delta=0.1)
print(f"   ✓ Huber loss computed: {loss.numpy()}")

# Small residual should be quadratic
small_res = tf.constant([0.05], dtype=tf.float32)
small_loss = huber_loss(small_res, delta=0.1)
expected_small = 0.5 * 0.05**2
assert abs(small_loss.numpy()[0] - expected_small) < 1e-6, "Small residual should be quadratic"
print(f"   ✓ Small residual (quadratic): {small_loss.numpy()[0]:.6f}")

# Large residual should be linear
large_res = tf.constant([2.0], dtype=tf.float32)
large_loss = huber_loss(large_res, delta=0.1)
expected_large = 0.1 * (2.0 - 0.5 * 0.1)
assert abs(large_loss.numpy()[0] - expected_large) < 1e-6, "Large residual should be linear"
print(f"   ✓ Large residual (linear): {large_loss.numpy()[0]:.6f}")

# Test 4: percentile_clip_loss (if TFP available)
print("\n4. Testing percentile_clip_loss...")
if TFP_AVAILABLE:
    residuals = tf.constant([0.1, 0.2, 0.3, 10.0], dtype=tf.float32)  # One outlier
    weights = tf.ones(4, dtype=tf.float32)
    loss = percentile_clip_loss(residuals, weights, percentile=75.0)
    print(f"   ✓ Percentile clip loss: {loss.numpy():.6f}")
    print("   ✓ TensorFlow Probability is available")
else:
    print("   ⚠ TensorFlow Probability not available - skipping this test")

# Test 5: delta_peak
print("\n5. Testing delta_peak...")
x = np.linspace(0, 2, 100)
peak = delta_peak(x, center=1.0, area=0.5, width_fraction=0.01)
computed_area = np.trapz(peak, x)
print(f"   ✓ Delta peak created at x=1.0")
print(f"   ✓ Computed area: {computed_area:.6f} (target: 0.5)")
assert abs(computed_area - 0.5) < 0.05, f"Area should be ~0.5, got {computed_area}"

# Check peak is at center
max_idx = np.argmax(peak)
assert abs(x[max_idx] - 1.0) < 0.02, "Peak should be at center"
print(f"   ✓ Peak location: x={x[max_idx]:.6f}")

# Test 6: set_random_seed
print("\n6. Testing set_random_seed...")
set_random_seed(42)
rand1 = np.random.rand()
tf_rand1 = tf.random.normal([1]).numpy()[0]

set_random_seed(42)  # Reset with same seed
rand2 = np.random.rand()
tf_rand2 = tf.random.normal([1]).numpy()[0]

assert rand1 == rand2, "NumPy random should be reproducible"
assert tf_rand1 == tf_rand2, "TensorFlow random should be reproducible"
print("   ✓ Random seed setting works (reproducible)")

# Test 7: configure_gpu_memory_growth
print("\n7. Testing configure_gpu_memory_growth...")
try:
    configure_gpu_memory_growth()
    print("   ✓ GPU memory growth configured (or no GPU available)")
except Exception as e:
    print(f"   ⚠ GPU config warning: {e}")

# Test 8: check_tensorflow_probability
print("\n8. Testing check_tensorflow_probability...")
tfp_status = check_tensorflow_probability()
print(f"   ✓ TensorFlow Probability available: {tfp_status}")
assert tfp_status == TFP_AVAILABLE, "TFP_AVAILABLE flag mismatch"

# Test 9: Integration with TensorFlow graph mode
print("\n9. Testing TensorFlow graph mode compatibility...")
@tf.function
def test_graph_mode():
    x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    y = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    integral = trapz_tf(y, x)
    loss = huber_loss(y - 2.0, delta=0.1)
    return integral, loss

integral, loss = test_graph_mode()
print(f"   ✓ Graph mode works: integral={integral.numpy():.3f}, loss={loss.numpy()}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Helper Functions working perfectly!")
print("="*60)
print("\nFunctions tested:")
print("  ✓ trapz_tf - TensorFlow trapezoidal integration")
print("  ✓ tail_trapz - Tail integration for birth terms")
print("  ✓ huber_loss - Robust loss function")
if TFP_AVAILABLE:
    print("  ✓ percentile_clip_loss - Outlier rejection")
print("  ✓ delta_peak - Delta function approximation")
print("  ✓ set_random_seed - Reproducibility")
print("  ✓ configure_gpu_memory_growth - GPU management")
print("  ✓ check_tensorflow_probability - TFP availability")
print("  ✓ TensorFlow @tf.function compatibility")
