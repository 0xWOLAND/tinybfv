from tinygrad import Tensor, dtypes
from tinybfv.tinygrad.bfv import N, t, keygen as tinygrad_keygen, encrypt as tinygrad_encrypt, decrypt as tinygrad_decrypt, homomorphic_add as tinygrad_homomorphic_add, homomorphic_mul as tinygrad_homomorphic_mul
from tinybfv.triton.bfv import keygen as triton_keygen, encrypt as triton_encrypt, decrypt as triton_decrypt, homomorphic_add as triton_homomorphic_add, homomorphic_mul as triton_homomorphic_mul

def compare_homomorphic_operations():
    # Generate key pairs
    sk_tg, pk_tg = tinygrad_keygen()
    sk_tr, pk_tr = triton_keygen()

    # Create random messages
    m1 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
    m2 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)

    # Tinygrad operations
    ct1_tg = tinygrad_encrypt(pk_tg, m1)
    ct2_tg = tinygrad_encrypt(pk_tg, m2)
    add_tg = tinygrad_decrypt(sk_tg, tinygrad_homomorphic_add(ct1_tg, ct2_tg))
    mul_tg = tinygrad_decrypt(sk_tg, tinygrad_homomorphic_mul(ct1_tg, 42))

    # Triton operations
    ct1_tr = triton_encrypt(pk_tr, m1)
    ct2_tr = triton_encrypt(pk_tr, m2)
    add_tr = triton_decrypt(sk_tr, triton_homomorphic_add(ct1_tr, ct2_tr))
    mul_tr = triton_decrypt(sk_tr, triton_homomorphic_mul(ct1_tr, 42))

    # Verify results
    add_expected = (m1 + m2) % t
    mul_expected = (m1 * 42) % t
    
    assert (add_tg == add_expected).all().numpy()
    assert (mul_tg == mul_expected).all().numpy()
    assert (add_tr == add_expected).all().numpy()
    assert (mul_tr == mul_expected).all().numpy()

if __name__ == "__main__":
    compare_homomorphic_operations()