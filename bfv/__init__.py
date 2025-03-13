from tinygrad import Tensor, dtypes

N = 2**4  # Ring dimension (polynomial degree + 1)
t = 2**8  # Plaintext modulus
q = 2**20  # Ciphertext modulus


def poly_mul(a: Tensor, b: Tensor, N: int, modulus: int):
    """
    Negacyclic convolution of two polynomials modulo X^N + 1 and modulus.
    """
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
    return Tensor.normal((N,), mean=0, std=2).cast(dtypes.int32)


def keygen():
    s = Tensor.randint((N,), low=0, high=2, dtype=dtypes.int32)
    a = Tensor.randint((N,), low=0, high=q, dtype=dtypes.int32)
    e = sample_noise()
    pk0 = (poly_mul(a, s, N, q) + e) % q
    pk0 = -pk0 % q
    pk1 = a
    return s, (pk0, pk1)


def encrypt(pk: tuple[Tensor, Tensor], m: Tensor):
    u = Tensor.randint((N,), low=0, high=2, dtype=dtypes.int32)
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


def homomorphic_mul(ct: tuple[Tensor, Tensor], pt: int):
    ct0, ct1 = ct
    m = Tensor.zeros(N, dtype=dtypes.int32).contiguous() % t
    m[0] = pt
    _c0 = poly_mul(ct0, m, N, q)
    _c1 = poly_mul(ct1, m, N, q)
    return (_c0, _c1)

