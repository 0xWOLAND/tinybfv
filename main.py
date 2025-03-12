from tinygrad import Tensor, dtypes
import numpy as np

def negacyclic_convolution(poly1, poly2, N, modulus):
    assert len(poly1) == N and len(poly2) == N, "Polynomials must be of length N"
    assert N > 0, "N must be positive"
    assert modulus > 0, "Modulus must be positive"
    
    # Convert inputs to tensors
    a = Tensor(poly1, dtype=dtypes.int32)
    b = Tensor(poly2, dtype=dtypes.int32)
    
    # Extend polynomials to 2N-1 to handle full convolution
    # This captures all terms before wrapping
    a_ext = Tensor.zeros(2 * N - 1, dtype=dtypes.int32).contiguous()
    b_ext = Tensor.zeros(2 * N - 1, dtype=dtypes.int32).contiguous()
    a_ext[:N] = a
    b_ext[:N] = b
    
    # Compute the convolution using tensor operations
    # We need to compute c[k] = sum(a[i] * b[j]) where i + j = k
    c = Tensor.zeros(2 * N - 1, dtype=dtypes.int32).contiguous()
    
    # Use tensor operations to compute convolution
    for k in range(2 * N - 1):
        # Create index tensors for valid i, j pairs where i + j = k
        i = Tensor.arange(max(0, k - N + 1), min(k + 1, N))
        j = k - i
        
        # Compute a[i] * b[j] for valid pairs
        prod = (a[i] * b[j]) % modulus
        c[k] = prod.sum() % modulus
    
    # Handle negacyclic wrapping (X^N = -1)
    result = Tensor.zeros(N, dtype=dtypes.int32).contiguous()
    for k in range(2 * N - 1):
        target = k % N
        sign = -1 if k >= N else 1
        result[target] = (result[target] + sign * c[k]) % modulus
    
    return result.numpy().tolist()

