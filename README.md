# jigandmpnn

JAX/Equinox translation of [LigandMPNN](https://github.com/dauparas/LigandMPNN) for protein sequence design.

## Installation

```bash
pip install jigandmpnn
```

Or with uv:

```bash
uv add jigandmpnn
```

## Quick Start

### Load a model

```python
from jigandmpnn import load_protein_mpnn, load_soluble_mpnn

model = load_protein_mpnn()  # Standard ProteinMPNN
model = load_soluble_mpnn()  # For soluble proteins
```

### Sample sequences

```python
import jax
import jax.numpy as jnp
from jigandmpnn import load_protein_mpnn
from jigandmpnn.vendor.ligandmpnn import parse_PDB, featurize

# Load model
model = load_protein_mpnn()

# Prepare features
protein_dict, *_ = parse_PDB("structure.pdb", device="cpu", chains=[])
feature_dict = featurize(protein_dict, model_type="protein_mpnn")

# Convert to JAX arrays
feature_dict_jax = {k: jnp.array(v.numpy()) if hasattr(v, 'numpy') else v
                    for k, v in feature_dict.items()}

# Sample
key = jax.random.PRNGKey(42)
result = model.sample(feature_dict_jax, key=key, temperature=0.1)
print(result["S"])  # Sampled sequences
```

### Score sequences

```python
result = model.score(feature_dict_jax, use_sequence=True)
print(result["log_probs"])  # Log probabilities [B, L, 21]
```

## Available Models

Use `get_weight_path(model_name)` to get checkpoint paths. All weights are included in the package.

**Aliases (recommended defaults):**
- `protein_mpnn` - ProteinMPNN (noise=0.20)
- `ligand_mpnn` - LigandMPNN (noise=0.10)
- `soluble_mpnn` - SolubleMPNN (noise=0.20)

**ProteinMPNN variants** (different training noise levels):
- `proteinmpnn_v_48_002`, `proteinmpnn_v_48_010`, `proteinmpnn_v_48_020`, `proteinmpnn_v_48_030`

**LigandMPNN variants:**
- `ligandmpnn_v_32_005_25`, `ligandmpnn_v_32_010_25`, `ligandmpnn_v_32_020_25`, `ligandmpnn_v_32_030_25`

**SolubleMPNN variants:**
- `solublempnn_v_48_002`, `solublempnn_v_48_010`, `solublempnn_v_48_020`, `solublempnn_v_48_030`

**Membrane MPNNs:**
- `per_residue_label_membrane_mpnn_v_48_020`
- `global_label_membrane_mpnn_v_48_020`

**Side-chain packing:**
- `ligandmpnn_sc_v_32_002_16`

```python
from jigandmpnn import list_weights
print(list_weights())  # Show all available weights
```

## Features

- Full JAX/Equinox implementation of ProteinMPNN and LigandMPNN
- Pretrained weights included
- JIT-compilable with `eqx.filter_jit`
- Autoregressive sampling with `jax.lax.scan`
- Supports coordinate noise (`augment_eps`) for training

## Benchmarks

Sampling 512 sequences for a 311-residue protein (3DI3.pdb) on NVIDIA H100:

| Framework | Method | Throughput | Time/seq | Speedup |
|-----------|--------|------------|----------|---------|
| **JAX** | vmap | **3,879 seq/s** | **0.26ms** | **4.0x** |
| PyTorch | batched | 968 seq/s | 1.03ms | 1.0x |

JAX with `vmap` parallelizes sampling across sequences:

```python
import jax
import equinox as eqx

@eqx.filter_jit
def sample_batch(model, feature_dict, keys):
    def sample_single(key):
        return model.sample(feature_dict, key=key, temperature=0.1)
    return jax.vmap(sample_single)(keys)

# Sample 512 sequences in parallel
keys = jax.random.split(jax.random.PRNGKey(0), 512)
result = sample_batch(model, feature_dict, keys)
```

Run benchmarks:
```bash
uv run python benchmarks/bench_torch.py
uv run python benchmarks/bench_jax.py
```

## Demos

See `demos/` for complete examples:

```bash
# PyTorch reference
python demos/demo_torch.py

# JAX implementation
python demos/demo_jax.py
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v
```

## License

MIT License. See LICENSE.

Vendored LigandMPNN code is also MIT licensed (Copyright 2024 Justas Dauparas).

## Citation

If you use this code, please cite the original LigandMPNN paper:

```bibtex
@article{dauparas2023atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={bioRxiv},
  year={2023}
}
```
