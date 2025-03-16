from tinygrad import Tensor, dtypes
from tinybfv.triton.bfv import N, t, keygen, encrypt, decrypt, homomorphic_add, homomorphic_mul


def test_homomorphic_operations():
    # Generate key pair
    sk, pk = keygen()

    # Create random messages
    m1 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
    m2 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)

    # Encrypt messages
    ct1 = encrypt(pk, m1)
    ct2 = encrypt(pk, m2)

    # Test homomorphic addition
    add_result = decrypt(sk, homomorphic_add(ct1, ct2))
    add_expected = (m1 + m2) % t
    assert (add_result == add_expected).all().numpy()

    # Test homomorphic multiplication
    C = 42  # Fixed constant for deterministic testing
    mul_result = decrypt(sk, homomorphic_mul(ct1, C))
    mul_expected = (m1 * C) % t
    assert (mul_result == mul_expected).all().numpy()


if __name__ == "__main__":
    test_homomorphic_operations()
