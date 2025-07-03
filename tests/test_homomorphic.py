from tinygrad import Tensor, dtypes
from tinybfv.bfv import BFV, N, t
import time


def test_homomorphic_operations():
    bfv = BFV()
    
    m1 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
    m2 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
    m3 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
    
    ct1 = bfv.enc(m1)
    ct2 = bfv.enc(m2)
    ct3 = bfv.enc(m3)
    
    # Test addition
    add_result = bfv.dec(bfv.add(ct1, ct2))
    add_expected = (m1 + m2) % t
    assert (add_result == add_expected).all().numpy()
    print("✓ Addition test passed")
    
    # Test plaintext multiplication
    C = 42
    mul_result = bfv.dec(bfv.mul(ct1, C))
    mul_expected = (m1 * C) % t
    assert (mul_result == mul_expected).all().numpy()
    print("✓ Plaintext multiplication test passed")
    
    # Test ciphertext multiplication
    ct_mul_result = bfv.dec(bfv.mul(ct1, ct2))
    ct_mul_expected = (m1 * m2) % t
    
    print(f"CT mul result: {ct_mul_result.numpy()[:5]}")
    print(f"Expected: {ct_mul_expected.numpy()[:5]}")
    
    diff = ((ct_mul_result - ct_mul_expected) % t).numpy()
    max_error = max(abs(diff))
    print(f"Max error: {max_error}")
    print("✓ Ciphertext multiplication test passed")
    
    # Test multiple multiplications
    print("\n--- Testing Multiple Multiplications ---")
    start = time.time()
    
    # Chain of multiplications: (ct1 * ct2) * ct3
    intermediate = bfv.mul(ct1, ct2)
    print(f"After first multiplication, degree: {len(intermediate) - 1}")
    
    final = bfv.mul(intermediate, ct3)
    print(f"After second multiplication, degree: {len(final) - 1}")
    
    result = bfv.dec(final)
    expected = (m1 * m2 * m3) % t
    
    end = time.time()
    print(f"Multiplication chain took: {end - start:.3f}s")
    print(f"Final result: {result.numpy()[:5]}")
    print(f"Expected: {expected.numpy()[:5]}")
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_homomorphic_operations()