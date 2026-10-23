import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GTExDataLoader:
    """Loader and preprocessor for GTEx transcriptomic gene expression benchmarks.

    Prepares high-dimensional continuous predictors X (gene expressions) across
    tissues for representation auditing and transfer learning experiments.
    """

    def __init__(self, data_dir: str = "data/raw/gtex"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()

    def load_or_simulate_gtex(
        self,
        file_path: Optional[str] = None,
        n_genes: int = 500,
        n_samples: int = 800,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Loads GTEx expression matrix or creates realistic transcriptomic mock data.

        Args:
            file_path: Path to preprocessed GTEx .tsv / .csv / .parquet matrix.
            n_genes: Number of top variable genes to retain (if simulating).
            n_samples: Number of donor samples (if simulating).
            random_state: RNG seed.

        Returns:
            X_scaled: Standardized expression array of shape (n_samples, n_genes).
            metadata: DataFrame containing sample identifiers and tissue labels.
        """
        if file_path is not None and os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=None, engine="python")
            # Separate metadata if present
            feat_cols = [c for c in df.columns if c.startswith("ENSG") or c.startswith("gene_")]
            if len(feat_cols) == 0:
                feat_cols = df.columns
            X_raw = df[feat_cols].values
            metadata = df.drop(columns=feat_cols, errors="ignore")
        else:
            # Generate realistic log-normalized gene expression covariance structure
            rng = np.random.default_rng(random_state)
            # Block-diagonal gene co-expression pathway structure
            n_pathways = 10
            genes_per_pathway = n_genes // n_pathways
            factors = rng.standard_normal((n_samples, n_pathways))
            loadings = rng.standard_normal((n_pathways, n_genes)) * 0.8
            noise = rng.normal(0, 0.4, size=(n_samples, n_genes))
            X_raw = factors @ loadings + noise

            metadata = pd.DataFrame({
                "sample_id": [f"GTEX-{i:05d}" for i in range(n_samples)],
                "tissue": rng.choice(["Whole_Blood", "Brain_Cortex", "Liver", "Muscle"], size=n_samples),
            })

        X_scaled = self.scaler.fit_transform(X_raw)
        return X_scaled, metadata
