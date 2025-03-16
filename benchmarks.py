import time
import torch
from tinygrad import Tensor, dtypes
from tinybfv.tinygrad.bfv import N as N_tiny, t as t_tiny, keygen as keygen_tiny, encrypt as encrypt_tiny, decrypt as decrypt_tiny, homomorphic_add as add_tiny, homomorphic_mul as mul_tiny
from tinybfv.triton.bfv import N as N_triton, t as t_triton, keygen as keygen_triton, encrypt as encrypt_triton, decrypt as decrypt_triton, homomorphic_add as add_triton, homomorphic_mul as mul_triton
from tabulate import tabulate

def benchmark_tinygrad():
    start = time.time()
    sk, pk = keygen_tiny()
    keygen_time = time.time() - start

    start = time.time()
    m1 = Tensor.randint((N_tiny,), low=0, high=t_tiny, dtype=dtypes.int32)
    m2 = Tensor.randint((N_tiny,), low=0, high=t_tiny, dtype=dtypes.int32)
    message_time = time.time() - start

    start = time.time()
    ct1 = encrypt_tiny(pk, m1)
    ct2 = encrypt_tiny(pk, m2)
    encrypt_time = time.time() - start

    start = time.time()
    add_ct = add_tiny(ct1, ct2)
    add_time = time.time() - start

    start = time.time()
    mul_ct = mul_tiny(ct1, 42)
    mul_time = time.time() - start

    start = time.time()
    add_result = decrypt_tiny(sk, add_ct)
    mul_result = decrypt_tiny(sk, mul_ct)
    decrypt_time = time.time() - start

    return {
        'keygen': keygen_time,
        'message': message_time,
        'encrypt': encrypt_time,
        'add': add_time,
        'mul': mul_time,
        'decrypt': decrypt_time
    }

def benchmark_triton():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    sk, pk = keygen_triton(device=device)
    keygen_time = time.time() - start

    start = time.time()
    m1 = torch.randint(0, t_triton, (N_triton,), dtype=torch.int32, device=device)
    m2 = torch.randint(0, t_triton, (N_triton,), dtype=torch.int32, device=device)
    message_time = time.time() - start

    start = time.time()
    ct1 = encrypt_triton(pk, m1, device=device)
    ct2 = encrypt_triton(pk, m2, device=device)
    encrypt_time = time.time() - start

    start = time.time()
    add_ct = add_triton(ct1, ct2)
    add_time = time.time() - start

    start = time.time()
    mul_ct = mul_triton(ct1, 42)
    mul_time = time.time() - start

    start = time.time()
    add_result = decrypt_triton(sk, add_ct)
    mul_result = decrypt_triton(sk, mul_ct)
    decrypt_time = time.time() - start

    return {
        'keygen': keygen_time,
        'message': message_time,
        'encrypt': encrypt_time,
        'add': add_time,
        'mul': mul_time,
        'decrypt': decrypt_time
    }

if __name__ == "__main__":
    tiny_results = benchmark_tinygrad()
    triton_results = benchmark_triton()

    # Prepare data for tabulate
    headers = ['Operation', 'Tinygrad (s)', 'Triton (s)', 'Speedup (x)']
    table_data = []
    for op in tiny_results.keys():
        tiny_time = tiny_results[op]
        triton_time = triton_results[op]
        speedup = tiny_time / triton_time if triton_time > 0 else float('inf')
        table_data.append([
            op,
            f"{tiny_time:.4f}",
            f"{triton_time:.4f}",
            f"{speedup:.2f}"
        ])

    # Print table using tabulate
    print("\nBenchmark Comparison:")
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))