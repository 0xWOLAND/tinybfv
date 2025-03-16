# `tinybfv` [![CI](https://github.com/0xWOLAND/tinybfv/actions/workflows/ci.yml/badge.svg)](https://github.com/0xWOLAND/tinybfv/actions/workflows/ci.yml)

A lightweight implementation of the BFV (Brakerski/Fan-Vercauteren) homomorphic encryption scheme using TinyGrad.

## Features

- Basic homomorphic addition and multiplication
- Key generation, encryption, and decryption

## Benchmarks

| Operation | Tinygrad (s) | Triton (s) | Speedup (x) |
|-----------|--------------|------------|-------------|
| keygen    | 2.7191       | 1.6114     | 1.69        |
| message   | 0.0055       | 0.0069     | 0.79        |
| encrypt   | 3.2442       | 1.5618     | 2.08        |
| add       | 0.0018       | 0.0000     | 36.22       |
| mul       | 1.5155       | 0.0006     | 2701.35     |
| decrypt   | 1.2678       | 0.0130     | 97.80       |