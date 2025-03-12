import numpy as np
from tinygrad import Tensor
from algebra.poly.univariate import Polynomial
from algebra.ff.prime_field import PrimeField

class Field(PrimeField):
    P = 127
    w = 7

# Parameters
N = 4  # Ring dimension (polynomial degree + 1)
t = 7  # Plaintext modulus (increased to 7)
q = 127  # Ciphertext modulus (increased to 127)

# Polynomial ring operations
def poly_add(poly1, poly2, modulus):
    """Add two polynomials modulo modulus."""
    return [(a + b) % modulus for a, b in zip(poly1, poly2)]

def poly_mul(poly1, poly2, modulus):
    """Multiply two polynomials modulo X^N + 1 and modulus."""
    result = [0] * N
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            k = (i + j) % N
            sign = -1 if (i + j) >= N else 1  # Handle X^N = -1
            result[k] = (result[k] + sign * poly1[i] * poly2[j]) % modulus
    return result

# Noise sampling (adjusted for larger q)
def sample_noise():
    return [np.random.randint(-2, 3) for _ in range(N)]  # Small noise: [-2, -1, 0, 1, 2]

# Key generation
def keygen():
    """Generate secret key s and public key (pk0, pk1)."""
    s = [np.random.randint(0, 2) for _ in range(N)]  # Binary secret key for simplicity
    a = [np.random.randint(0, q) for _ in range(N)]  # Uniform random poly
    e = sample_noise()  # Noise
    pk0 = poly_add(poly_mul(a, s, q), e, q)  # pk0 = -(a * s + e)
    pk0 = [-x % q for x in pk0]  # Negate
    pk1 = a  # pk1 = a
    return s, (pk0, pk1)

# Encryption
def encrypt(pk, m):
    """Encrypt plaintext m (polynomial in Z_t)."""
    pk0, pk1 = pk
    u = [np.random.randint(0, 2) for _ in range(N)]  # Binary random poly
    e1, e2 = sample_noise(), sample_noise()  # Noise
    delta = (q // t)  # Scaling factor
    scaled_m = [(delta * x) % q for x in m]  # Scale plaintext
    ct0 = poly_add(poly_add(poly_mul(pk0, u, q), e1, q), scaled_m, q)  # ct0 = pk0 * u + e1 + delta * m
    ct1 = poly_add(poly_mul(pk1, u, q), e2, q)  # ct1 = pk1 * u + e2
    return (ct0, ct1)

# Decryption
def decrypt(s, ct):
    """Decrypt ciphertext (ct0, ct1) using secret key s."""
    ct0, ct1 = ct
    # Compute ct0 + ct1 * s
    decrypted = poly_add(ct0, poly_mul(ct1, s, q), q)
    # Scale down and round to recover m
    scaled = [(x * t + q // 2) // q % t for x in decrypted]  # Add q/2 for rounding
    return scaled

# Homomorphic addition
def homomorphic_add(ct1, ct2):
    """Add two ciphertexts."""
    ct1_0, ct1_1 = ct1
    ct2_0, ct2_1 = ct2
    return (poly_add(ct1_0, ct2_0, q), poly_add(ct1_1, ct2_1, q))

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)

    # Generate keys
    s, pk = keygen()

    # Plaintexts (polynomials in Z_t)
    m1 = [1,2,3,4]  # Represents 3
    m2 = [2,4,6,0]  # Represents 4

    # Encrypt
    ct1 = encrypt(pk, m1)
    ct2 = encrypt(pk, m2)

    # Homomorphic addition
    ct_sum = homomorphic_add(ct1, ct2)

    # Decrypt
    result = decrypt(s, ct_sum)
    print("Plaintext m1:", m1)
    print("Plaintext m2:", m2)
    print("Decrypted sum (mod t=7):", result)  # Should be [0, 0, 0, 0] (3 + 4 = 0 mod 7)