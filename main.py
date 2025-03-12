from tinygrad import Tensor, dtypes

P = 127  # Prime field modulus
N = 4    # Ring dimension (polynomial degree + 1)
t = 7    # Plaintext modulus
q = 127  # Ciphertext modulus

def poly_mul(a: Tensor, b: Tensor, N: int, modulus: int):
   '''
    Negacyclic convolution of two polynomials modulo X^N + 1 and modulus.
   '''
   c = Tensor.zeros(2 * N - 1, dtype=dtypes.int32).contiguous()
   
   for k in range(2 * N - 1):
       i = Tensor.arange(max(0, k - N + 1), min(k + 1, N))
       j = k - i
       c[k] = (a[i] * b[j]).sum() % modulus
   
   result = Tensor.zeros(N, dtype=dtypes.int32).contiguous()
   for k in range(2 * N - 1):
       target = k % N
       sign = -1 if k >= N else 1
       result[target] = (result[target] + sign * c[k]) % modulus
   
   return result

def sample_noise():
   return Tensor.randint((N, ), low=-2, high=3, dtype=dtypes.int32)

def keygen():
   s = Tensor.randint((N, ), low=0, high=2, dtype=dtypes.int32)
   a = Tensor.randint((N, ), low=0, high=q, dtype=dtypes.int32)
   e = sample_noise()
   pk0 = (poly_mul(a, s, N, q) + e) % q
   pk0 = -pk0 % q
   pk1 = a
   return s, (pk0, pk1)

def encrypt(pk: tuple[Tensor, Tensor], m: Tensor):
   u = Tensor.randint((N, ), low=0, high=2, dtype=dtypes.int32)
   e1, e2 = sample_noise(), sample_noise()
   delta = q // t
   scaled_m = (delta * m) % q
   ct0 = (poly_mul(pk[0], u, N, q) + e1 + scaled_m) % q
   ct1 = poly_mul(pk[1], u, N, q) + e2
   return (ct0, ct1)

def decrypt(sk: Tensor, ct: tuple[Tensor, Tensor]):
    """Decrypt ciphertext (ct0, ct1) using secret key sk."""
    ct0, ct1 = ct

    decrypted = (ct0 + poly_mul(ct1, sk, N, q)) % q
    scaled = (((decrypted * t) + (q // 2)) // q) % t
    
    return scaled

def homomorphic_add(ct1: tuple[Tensor, Tensor], ct2: tuple[Tensor, Tensor]):
   return (ct1[0] + ct2[0]) % q, (ct1[1] + ct2[1]) % q

if __name__ == "__main__":
   # Generate key pair
   s, pk = keygen()
   
   # Create random messages
   m1 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
   m2 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
   
   # Encrypt messages
   ct1 = encrypt(pk, m1)
   ct2 = encrypt(pk, m2)
   
   # Test homomorphic addition
   result = decrypt(s, homomorphic_add(ct1, ct2))
   expected = (m1 + m2) % t
   
   # Verify correctness
   assert (result == expected).all().numpy(), "Homomorphic addition test failed"
   print("Homomorphic addition test passed!")
   
   # Print some values for verification
   print(f"m1: {m1.numpy()}")
   print(f"m2: {m2.numpy()}")
   print(f"m1 + m2 (mod {t}): {expected.numpy()}")
   print(f"Decrypted result: {result.numpy()}")
