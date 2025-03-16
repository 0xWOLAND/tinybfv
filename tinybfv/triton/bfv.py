import triton
import triton.language as tl
import torch

N = 2**4
t = 2**8
q = 2**20

@triton.jit
def uniform_random_kernel(output_ptr, seed: tl.constexpr, modulus: tl.constexpr, 
                         N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    random_vals = tl.rand(seed, idx)
    scaled_vals = random_vals * modulus
    rounded_vals = tl.floor(scaled_vals).to(tl.int32)
    tl.store(output_ptr + idx, rounded_vals, mask=mask)

@triton.jit
def sample_noise_kernel(output_ptr, seed: tl.constexpr, N: tl.constexpr, 
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    noise = tl.randn(seed, idx)
    scaled_noise = noise * 2.0
    rounded_noise = tl.floor(scaled_noise + 0.5).to(tl.int32)
    tl.store(output_ptr + idx, rounded_noise, mask=mask)

@triton.jit
def poly_mul_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr, modulus: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    k = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = k < N
    c = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    idx = tl.arange(0, BLOCK_SIZE)
    load_mask = idx < N
    a = tl.load(a_ptr + idx, mask=load_mask, other=0)
    b = tl.load(b_ptr + idx, mask=load_mask, other=0)
    
    for i in range(N):
        j = (k - i + N) % N
        valid_mask = mask & (i < N) & (j < N)
        sign = tl.where((i + j) >= N, -1, 1)
        a_val = tl.load(a_ptr + i, mask=i < N, other=0)
        b_val = tl.load(b_ptr + j, mask=j < N, other=0)
        contrib = tl.where(valid_mask, sign * a_val * b_val, 0)
        c = (c + contrib) % modulus
    
    tl.store(c_ptr + k, c, mask=mask)

def sample_uniform(modulus, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = torch.zeros(N, dtype=torch.int32, device=device)
    seed = int(torch.randint(0, 2**32, (1,)).item())  # Use a random seed
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    uniform_random_kernel[grid](output, seed, modulus, N, BLOCK_SIZE)
    return output

def sample_noise(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = torch.zeros(N, dtype=torch.int32, device=device)
    seed = int(torch.randint(0, 2**32, (1,)).item())  # Use a random seed
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sample_noise_kernel[grid](output, seed, N, BLOCK_SIZE)
    return output

def poly_mul(a, b, N_val, modulus):
    # Ensure same device
    device = a.device
    b = b.to(device)
    
    output = torch.zeros(N_val, dtype=torch.int32, device=device)
    BLOCK_SIZE = 16
    grid = lambda meta: (triton.cdiv(N_val, meta['BLOCK_SIZE']),)
    poly_mul_kernel[grid](a, b, output, N_val, modulus, BLOCK_SIZE)
    return output

def keygen(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s = sample_uniform(modulus=2, device=device)
    a = sample_uniform(modulus=q, device=device)
    e = sample_noise(device=device)
    pk0 = (-(poly_mul(a, s, N, q) + e)) % q
    return s, (pk0, a)

def encrypt(pk, m, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pk0, pk1 = pk
    # Ensure all tensors are on the same device
    pk0 = pk0.to(device)
    pk1 = pk1.to(device)
    m = m.to(device)
    
    u = sample_uniform(modulus=2, device=device)
    e1, e2 = sample_noise(device=device), sample_noise(device=device)
    delta = q // t
    scaled_m = (delta * m) % q
    ct0 = (poly_mul(pk0, u, N, q) + e1 + scaled_m) % q
    ct1 = (poly_mul(pk1, u, N, q) + e2) % q
    return (ct0, ct1)

def decrypt(sk, ct):
    # Ensure all tensors are on the same device
    device = sk.device
    ct0, ct1 = ct[0].to(device), ct[1].to(device)
    
    decrypted = (ct0 + poly_mul(ct1, sk, N, q)) % q
    return (((decrypted * t) + q//2) // q) % t

def homomorphic_add(ct1, ct2):
    # Ensure same device
    device = ct1[0].device
    ct1_0, ct1_1 = ct1
    ct2_0, ct2_1 = ct2[0].to(device), ct2[1].to(device)
    
    return (ct1_0 + ct2_0) % q, (ct1_1 + ct2_1) % q

def homomorphic_mul(ct, pt):
    # Ensure same device
    device = ct[0].device
    ct0, ct1 = ct
    
    # Create a polynomial with constant term pt
    m = torch.zeros(N, dtype=torch.int32, device=device)
    m[0] = pt
    
    return (poly_mul(ct0, m, N, q), poly_mul(ct1, m, N, q))
