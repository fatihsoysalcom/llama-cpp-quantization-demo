import numpy as np

def quantize_weight(weight, num_bits):
    """Simulates weight quantization to a specified number of bits."""
    if num_bits <= 0:
        return weight # No quantization

    # Determine the range of the weight
    min_val = np.min(weight)
    max_val = np.max(weight)

    # Calculate the scaling factor and zero point
    # For simplicity, we'll use a symmetric quantization around 0
    scale = max_val / ((2**(num_bits - 1)) - 1)
    if scale == 0:
        return np.zeros_like(weight) # Handle all-zero weights

    # Quantize to integer
    quantized = np.round(weight / scale)

    # Clip values to the representable range
    max_int = (2**(num_bits - 1)) - 1
    min_int = -(2**(num_bits - 1))
    quantized = np.clip(quantized, min_int, max_int)

    # Dequantize back to float for comparison
    dequantized = quantized * scale
    return dequantized

def main():
    print("--- llama.cpp Quantization Demo ---")

    # Simulate a small weight tensor (e.g., a layer's weights)
    # In real LLMs, these tensors are massive.
    original_weights = np.array([
        [0.5, -0.2, 1.1, -0.8],
        [-1.5, 0.9, 0.1, 0.3],
        [0.7, -0.4, -0.9, 1.2]
    ], dtype=np.float32)

    print("\nOriginal Weights (float32):\n", original_weights)
    print(f"Original size: {original_weights.nbytes} bytes")

    # --- Demonstrate 8-bit quantization (common in llama.cpp) ---
    num_bits_8bit = 8
    quantized_weights_8bit = quantize_weight(original_weights, num_bits_8bit)
    # In a real scenario, this would be stored as int8, saving space.
    # Here, we dequantize to float32 for demonstration of approximation.
    print(f"\nQuantized Weights ({num_bits_8bit}-bit approximation):\n", quantized_weights_8bit)
    # Approximate size reduction: original_weights.nbytes / num_bits_8bit * 8
    print(f"Approximate size after {num_bits_8bit}-bit quantization: {original_weights.nbytes / 2} bytes (assuming 8x reduction)")

    # --- Demonstrate 4-bit quantization (more aggressive, TurboQuant like) ---
    num_bits_4bit = 4
    quantized_weights_4bit = quantize_weight(original_weights, num_bits_4bit)
    print(f"\nQuantized Weights ({num_bits_4bit}-bit approximation):\n", quantized_weights_4bit)
    print(f"Approximate size after {num_bits_4bit}-bit quantization: {original_weights.nbytes / 4} bytes (assuming 4x reduction)")

    # --- Calculate approximation error ---
    error_8bit = np.mean(np.abs(original_weights - quantized_weights_8bit))
    error_4bit = np.mean(np.abs(original_weights - quantized_weights_4bit))

    print(f"\nAverage absolute error (8-bit): {error_8bit:.4f}")
    print(f"Average absolute error (4-bit): {error_4bit:.4f}")

    print("\nNote: This is a simplified simulation. Real TurboQuant involves more sophisticated techniques for minimizing error.")

if __name__ == "__main__":
    main()
