from tinygrad import Tensor, dtypes
import math

N = 2**4
t = 2**8
q = 2**20
base = 2**16


def poly_mul(a: Tensor, b: Tensor, mod: int):
    a, b = a.cast(dtypes.int32), b.cast(dtypes.int32)
    
    # Vectorized negacyclic convolution
    i_idx = Tensor.arange(N).reshape(N, 1).expand(N, N)
    j_idx = Tensor.arange(N).reshape(1, N).expand(N, N)
    k_idx = Tensor.arange(N).reshape(1, 1, N).expand(N, N, N)
    
    a_exp = a[i_idx]
    b_exp = b[j_idx]
    prods = a_exp * b_exp
    degs = i_idx + j_idx
    
    # Accumulate all coefficients at once
    norm_mask = (degs.unsqueeze(2) == k_idx).cast(dtypes.int32)
    neg_mask = (degs.unsqueeze(2) == (k_idx + N)).cast(dtypes.int32)
    
    norm_contrib = (prods.unsqueeze(2) * norm_mask).sum(axis=(0, 1))
    neg_contrib = (prods.unsqueeze(2) * neg_mask).sum(axis=(0, 1))
    
    return (norm_contrib - neg_contrib) % mod


def noise():
    return Tensor.normal((N,), mean=0, std=2).cast(dtypes.int32)


def base_decomp(poly: Tensor, num_limbs: int):
    result = []
    temp = poly.cast(dtypes.int64)
    
    for _ in range(num_limbs):
        result.append((temp % base).cast(dtypes.int32))
        temp = temp // base
    
    return result


def keygen():
    s = Tensor.randint((N,), low=0, high=2, dtype=dtypes.int32)
    a = Tensor.randint((N,), low=0, high=q, dtype=dtypes.int32)
    e = noise()
    
    pk0 = (poly_mul(a, s, q) + e) % q
    pk0 = -pk0 % q
    pk = (pk0, a)
    
    num_limbs = math.ceil(math.log(q, base))
    s_sq = poly_mul(s, s, q)
    
    rlk = []
    for i in range(num_limbs):
        ai = Tensor.randint((N,), low=0, high=q, dtype=dtypes.int32)
        ei = noise()
        rlk0 = (-poly_mul(ai, s, q) + ei + (base ** i) * s_sq) % q
        rlk.append((rlk0, ai))
    
    return s, pk, rlk


def encrypt(pk, m: Tensor):
    pk0, pk1 = pk
    u = Tensor.randint((N,), low=0, high=2, dtype=dtypes.int32)
    e1, e2 = noise(), noise()
    
    scaled_m = ((q // t) * m) % q
    ct0 = (poly_mul(pk0, u, q) + e1 + scaled_m) % q
    ct1 = poly_mul(pk1, u, q) + e2
    return [ct0, ct1]


def decrypt(sk: Tensor, ct):
    dec = (ct[0] + poly_mul(ct[1], sk, q)) % q
    return (((dec * t) + (q // 2)) // q) % t


def add(ct1, ct2):
    max_len = max(len(ct1), len(ct2))
    zeros = Tensor.zeros(N, dtype=dtypes.int32)
    
    result = []
    for i in range(max_len):
        c1 = ct1[i] if i < len(ct1) else zeros
        c2 = ct2[i] if i < len(ct2) else zeros
        result.append((c1 + c2) % q)
    
    return result


def relinearize(ct, rlk):
    if len(ct) < 3:
        return ct
    
    *lower, c_high = ct
    decomp = base_decomp(c_high, len(rlk))
    
    if len(lower) == 1:
        lower.append(Tensor.zeros(N, dtype=dtypes.int32))
    
    result = list(lower)
    
    # Vectorized relinearization  
    for i, (rlk0, rlk1) in enumerate(rlk):
        result[0] = (result[0] + poly_mul(decomp[i], rlk0, q)) % q
        result[1] = (result[1] + poly_mul(decomp[i], rlk1, q)) % q
    
    return result


def mul_ct(ct1, ct2, rlk, lazy_threshold=3):
    # Compute product
    result_deg = len(ct1) + len(ct2) - 1
    result = [Tensor.zeros(N, dtype=dtypes.int32) for _ in range(result_deg)]
    
    for i, p1 in enumerate(ct1):
        for j, p2 in enumerate(ct2):
            result[i + j] = (result[i + j] + poly_mul(p1, p2, q)) % q
    
    # Lazy relinearization
    while len(result) > lazy_threshold + 1:
        result = relinearize(result, rlk)
    
    return result


def mul_pt(ct, pt: int):
    pt_poly = Tensor.zeros(N, dtype=dtypes.int32)
    pt_poly = pt_poly + (Tensor.arange(N) == 0).cast(dtypes.int32) * pt
    return [poly_mul(p, pt_poly, q) for p in ct]


class BFV:
    def __init__(self):
        self.sk, self.pk, self.rlk = keygen()
    
    def enc(self, m: Tensor):
        return encrypt(self.pk, m)
    
    def dec(self, ct):
        # Force to degree 1 for decryption
        while len(ct) > 2:
            ct = relinearize(ct, self.rlk)
        return decrypt(self.sk, ct)
    
    def add(self, ct1, ct2):
        return add(ct1, ct2)
    
    def mul(self, ct1, ct2_or_pt):
        if isinstance(ct2_or_pt, int):
            return mul_pt(ct1, ct2_or_pt)
        return mul_ct(ct1, ct2_or_pt, self.rlk)