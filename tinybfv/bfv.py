from tinygrad import Tensor, dtypes
import math

N = 2**4
t = 2**8
q = 2**20
base = 2**16


def poly_mul(a: Tensor, b: Tensor, mod: int):
    a, b = a.cast(dtypes.int32), b.cast(dtypes.int32)
    
    i_idx = Tensor.arange(N).reshape(-1, 1)
    j_idx = Tensor.arange(N).reshape(1, -1)
    
    a_exp = a[i_idx]
    b_exp = b[j_idx]
    prods = a_exp * b_exp
    degs = i_idx + j_idx
    
    result = Tensor.zeros(N, dtype=dtypes.int32)
    
    for k in range(N):
        norm_mask = (degs == k).cast(dtypes.int32)
        neg_mask = (degs == (k + N)).cast(dtypes.int32)
        
        coeff = (prods * norm_mask).sum() - (prods * neg_mask).sum()
        update_mask = (Tensor.arange(N) == k).cast(dtypes.int32)
        result = result + update_mask * coeff
    
    return result % mod


def noise():
    return Tensor.normal((N,), mean=0, std=2).cast(dtypes.int32)


def base_decomp(poly: Tensor, num_limbs: int):
    result = []
    temp = poly.cast(dtypes.int32)
    
    for _ in range(num_limbs):
        result.append(temp % base)
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
    rlk = []
    s_sq = poly_mul(s, s, q)
    
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
    return (ct0, ct1)


def decrypt(sk: Tensor, ct):
    ct0, ct1 = ct
    dec = (ct0 + poly_mul(ct1, sk, q)) % q
    return (((dec * t) + (q // 2)) // q) % t


def add(ct1, ct2):
    return (ct1[0] + ct2[0]) % q, (ct1[1] + ct2[1]) % q


def relinearize(ct, rlk):
    c0, c1, c2 = ct
    decomp = base_decomp(c2, len(rlk))
    
    res_c0, res_c1 = c0, c1
    
    for i, (rlk0, rlk1) in enumerate(rlk):
        res_c0 = (res_c0 + poly_mul(decomp[i], rlk0, q)) % q
        res_c1 = (res_c1 + poly_mul(decomp[i], rlk1, q)) % q
    
    return (res_c0, res_c1)


def mul_ct(ct1, ct2, rlk):
    a0, a1 = ct1
    b0, b1 = ct2
    
    c0 = poly_mul(a0, b0, q)
    c1 = (poly_mul(a0, b1, q) + poly_mul(a1, b0, q)) % q
    c2 = poly_mul(a1, b1, q)
    
    return relinearize((c0, c1, c2), rlk)


def mul_pt(ct, pt: int):
    ct0, ct1 = ct
    m = Tensor([pt] + [0] * (N-1), dtype=dtypes.int32)
    return (poly_mul(ct0, m, q), poly_mul(ct1, m, q))


class BFV:
    def __init__(self):
        self.sk, self.pk, self.rlk = keygen()
    
    def enc(self, m: Tensor):
        return encrypt(self.pk, m)
    
    def dec(self, ct):
        return decrypt(self.sk, ct)
    
    def add(self, ct1, ct2):
        return add(ct1, ct2)
    
    def mul(self, ct1, ct2_or_pt):
        if isinstance(ct2_or_pt, int):
            return mul_pt(ct1, ct2_or_pt)
        return mul_ct(ct1, ct2_or_pt, self.rlk)