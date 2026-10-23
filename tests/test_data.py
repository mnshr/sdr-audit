import numpy as np
from sdraudit.data.generator import SemiSyntheticGenerator
from sdraudit.data.gtex_loader import GTExDataLoader


def test_generator_all_settings():
    gen = SemiSyntheticGenerator(p=12, d=2, noise_std=0.1, random_state=42)

    for setting in ["null", "mean_shift", "variance_shift"]:
        X, Y, Z = gen.generate(n_samples=50, setting=setting)
        assert X.shape == (50, 12)
        assert Y.shape == (50, 1)
        assert Z.shape == (50, 2)


def test_gtex_loader_simulation():
    loader = GTExDataLoader()
    X, meta = loader.load_or_simulate_gtex(n_genes=60, n_samples=100, random_state=42)

    assert X.shape == (100, 60)
    assert len(meta) == 100
    assert "tissue" in meta.columns
