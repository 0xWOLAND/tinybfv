import torch
from tinybfv.triton.bfv import N, t, keygen, encrypt, decrypt, homomorphic_add, homomorphic_mul

def test_homomorphic_operations():
    # Choose a consistent device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate key pair
    sk, pk = keygen(device=device)

    # Create random messages on the same device
    m1 = torch.randint(0, t, (N,), dtype=torch.int32, device=device)
    m2 = torch.randint(0, t, (N,), dtype=torch.int32, device=device)

    # Encrypt messages
    ct1 = encrypt(pk, m1, device=device)
    ct2 = encrypt(pk, m2, device=device)

    # Test homomorphic addition
    add_result = decrypt(sk, homomorphic_add(ct1, ct2))
    add_expected = (m1 + m2) % t
    print("Addition test:", torch.all(add_result == add_expected).item())

    # Test homomorphic multiplication
    C = 42  # Fixed constant for deterministic testing
    mul_result = decrypt(sk, homomorphic_mul(ct1, C))
    mul_expected = (m1 * C) % t
    print("Multiplication test:", torch.all(mul_result == mul_expected).item())

if __name__ == "__main__":
    test_homomorphic_operations()