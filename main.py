from tinygrad import Tensor, dtypes

P = 127  # Prime field modulus
N = 4    # Ring dimension (polynomial degree + 1)
t = 7    # Plaintext modulus
q = 257 # Ciphertext modulus

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

def generate_evaluation_key(sk: Tensor, q: int, N: int):
    """
    Generate evaluation key for relinearization.
    
    Args:
        sk: Secret key
        q: Ciphertext modulus
        N: Ring dimension
        
    Returns:
        Evaluation key (rlk0, rlk1)
    """
    # Generate sk^2
    sk_squared = poly_mul(sk, sk, N, q)
    
    # Sample random polynomial
    a = Tensor.randint((N, ), low=0, high=q, dtype=dtypes.int32)
    
    # Sample error
    e = sample_noise()
    
    # Create evaluation key
    rlk0 = (poly_mul(a, sk, N, q) + e + sk_squared) % q
    rlk0 = (-rlk0) % q  # Negate as in keygen
    rlk1 = a
    
    return (rlk0, rlk1)

def relinearize(ct: tuple[Tensor, Tensor, Tensor], rlk: tuple[Tensor, Tensor], q: int, N: int):
    """
    Relinearize a ciphertext with 3 components back to 2 components.
    
    Args:
        ct: Ciphertext tuple (c0, c1, c2) after homomorphic multiplication
        rlk: Evaluation/relinearization key (rlk0, rlk1)
        q: Ciphertext modulus
        N: Ring dimension
        
    Returns:
        Relinearized ciphertext (c0', c1')
    """
    c0, c1, c2 = ct
    rlk0, rlk1 = rlk
    
    # Calculate new ciphertext components
    # c0' = c0 + rlk0 * c2
    # c1' = c1 + rlk1 * c2
    new_c0 = (c0 + poly_mul(rlk0, c2, N, q)) % q
    new_c1 = (c1 + poly_mul(rlk1, c2, N, q)) % q
    
    return (new_c0, new_c1)

# Modified homomorphic multiplication with relinearization
def homomorphic_mul(ct1: tuple[Tensor, Tensor], ct2: tuple[Tensor, Tensor], 
                                 rlk: tuple[Tensor, Tensor]):
    """
    Perform homomorphic multiplication followed by relinearization.
    
    Args:
        ct1, ct2: Input ciphertexts
        rlk: Evaluation/relinearization key
        q: Ciphertext modulus
        t: Plaintext modulus
        N: Ring dimension
        
    Returns:
        Relinearized ciphertext result of multiplication
    """
    # Perform homomorphic multiplication
    ct1_0, ct1_1 = ct1
    ct2_0, ct2_1 = ct2
    
    c0 = poly_mul(ct1_0, ct2_0, N, q)
    c1 = (poly_mul(ct1_0, ct2_1, N, q) + poly_mul(ct1_1, ct2_0, N, q)) % q
    c2 = poly_mul(ct1_1, ct2_1, N, q)
    
    # Scale by t/q and round
    scaled_c0 = ((t * c0.cast(dtypes.float32)) / q + 0.5).cast(dtypes.int32) % q
    scaled_c1 = ((t * c1.cast(dtypes.float32)) / q + 0.5).cast(dtypes.int32) % q
    scaled_c2 = ((t * c2.cast(dtypes.float32)) / q + 0.5).cast(dtypes.int32) % q
    
    # Perform relinearization
    return relinearize((scaled_c0, scaled_c1, scaled_c2), rlk, q, N)

# Example usage
if __name__ == "__main__":
    # Generate key pair
    sk, pk = keygen()
    
    # Generate evaluation key for relinearization
    rlk = generate_evaluation_key(sk, q, N)
    
    # Create random messages
    m1 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
    m2 = Tensor.randint((N,), low=0, high=t, dtype=dtypes.int32)
    
    # Encrypt messages
    ct1 = encrypt(pk, m1)
    ct2 = encrypt(pk, m2)
    
    # Test homomorphic addition
    add_result = decrypt(sk, homomorphic_add(ct1, ct2))
    add_expected = (m1 + m2) % t
    
    # Test homomorphic multiplication with relinearization
    mul_result = decrypt(sk, homomorphic_mul(ct1, ct2, rlk))
    mul_expected = (poly_mul(m1, m2, N, t)) % t
    
    # Verify correctness
    print(f"m1: {m1.numpy()}")
    print(f"m2: {m2.numpy()}")
    print(f"Addition expected: {add_expected.numpy()}")
    print(f"Addition result: {add_result.numpy()}")
    print(f"Multiplication expected: {mul_expected.numpy()}")
    print(f"Multiplication result: {mul_result.numpy()}")
    
    if (add_result == add_expected).all().numpy():
        print("Homomorphic addition test passed!")
    else:
        print("Homomorphic addition test failed!")
        
    if (mul_result == mul_expected).all().numpy():
        print("Homomorphic multiplication test passed!")
    else:
        print("Homomorphic multiplication test failed!")