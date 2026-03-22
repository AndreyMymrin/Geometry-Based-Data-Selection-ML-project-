
import sys
import os
import math
import unittest
 
import numpy as np
import pandas as pd
 
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
 
from gds.scoring.effective_rank import (
    erank,
    layerwise_erank,
    average_erank,
    EffectiveRankScorer,
    rank_generators,
)
 
RNG = np.random.default_rng(0)
 
 
def _make_rank_k_matrix(n: int, d: int, k: int, rng=RNG) -> np.ndarray:
    """Return an n×d matrix of exactly rank k via SVD construction."""
    r = min(n, d)
    assert k <= r, "k cannot exceed min(n, d)"
    sigma = np.zeros(r)
    sigma[:k] = rng.uniform(1.0, 10.0, size=k)
    U = np.linalg.qr(rng.standard_normal((n, r)))[0]
    Vt = np.linalg.qr(rng.standard_normal((d, r)))[0].T
    return (U * sigma) @ Vt
 
 
def _make_hidden(n_layers: int, n_tok: int, d: int,
                 diversity: float = 1.0, rng=RNG) -> list[np.ndarray]:
    """Return a list of L activation matrices for one text."""
    layers = []
    for _ in range(n_layers):
        if diversity < 1.0:
            k = max(2, int(diversity * d))
            basis = rng.standard_normal((d, k))
            coords = rng.standard_normal((n_tok, k))
            X = coords @ basis.T
        else:
            X = rng.standard_normal((n_tok, d))
        layers.append(X.astype(np.float64))
    return layers
 
 
class TestErank(unittest.TestCase):
 
    def test_lower_bound_rank1_matrix(self):
        """Rank-1 matrix must give ERank exactly 1.0."""
        vec = RNG.standard_normal(32)
        X = np.outer(RNG.standard_normal(20), vec)   # all rows collinear
        result = erank(X)
        self.assertAlmostEqual(result, 1.0, delta=1e-6,
                               msg="Rank-1 matrix should have ERank = 1.0")
 
    def test_upper_bound_uniform_singular_values(self):
        """Matrix with equal singular values → ERank = min(n, d)."""
        n, d = 8, 5
        r = min(n, d)
        U = np.linalg.qr(RNG.standard_normal((n, r)))[0]
        Vt = np.linalg.qr(RNG.standard_normal((d, r)))[0].T
        sigma = np.ones(r)
        X = (U * sigma) @ Vt
        result = erank(X)
        self.assertAlmostEqual(result, float(r), places=5,
                               msg=f"Uniform SVs should give ERank = {r}")
 
    def test_result_within_valid_range(self):
        """ERank must lie in [1, min(n, d)] for random matrices."""
        for n, d in [(10, 5), (5, 10), (20, 20), (100, 32)]:
            X = RNG.standard_normal((n, d))
            result = erank(X)
            upper = float(min(n, d))
            self.assertGreaterEqual(result, 1.0 - 1e-9,
                                    msg=f"ERank below 1 for shape ({n},{d})")
            self.assertLessEqual(result, upper + 1e-6,
                                 msg=f"ERank above {upper} for shape ({n},{d})")
 
    def test_manual_2x2_known_result(self):
        """Verify formula against hand-computed value for a 2×2 diagonal matrix.
 
        X = diag(3, 1).  SVD singular values = [3, 1].
        p = [3/4, 1/4].
        H = -(3/4 * log(3/4) + 1/4 * log(1/4))
        ERank = exp(H)
        """
        X = np.diag([3.0, 1.0])
        p = np.array([3.0 / 4.0, 1.0 / 4.0])
        H = -np.sum(p * np.log(p))
        expected = math.exp(H)
        result = erank(X)
        self.assertAlmostEqual(result, expected, places=10,
                               msg="ERank does not match manual calculation")
 
    def test_anisotropic_matrix_is_low(self):
        """One dominant singular value → ERank close to 1."""
        n, d = 32, 64
        r = min(n, d)
        U = np.linalg.qr(RNG.standard_normal((n, r)))[0]
        Vt = np.linalg.qr(RNG.standard_normal((d, r)))[0].T
        sigma = np.full(r, 0.01)
        sigma[0] = 100.0
        X = (U * sigma) @ Vt
        result = erank(X)
        self.assertLess(result, 1.5,
                        msg="Anisotropic matrix should yield ERank near 1")
 
    def test_scale_invariance(self):
        """Multiplying X by a scalar must not change ERank."""
        X = RNG.standard_normal((15, 10))
        base = erank(X)
        for scale in [0.001, 0.5, 2.0, 1000.0]:
            result = erank(X * scale)
            self.assertAlmostEqual(result, base, places=8,
                                   msg=f"ERank changed with scale={scale}")
 
    def test_row_permutation_invariance(self):
        """Permuting rows must not change ERank (SVD is permutation-invariant)."""
        X = RNG.standard_normal((20, 8))
        base = erank(X)
        idx = RNG.permutation(20)
        result = erank(X[idx])
        self.assertAlmostEqual(result, base, places=10)
 
    def test_orthogonal_column_transform_invariance(self):
        """Right-multiplying by an orthogonal matrix must not change ERank."""
        X = RNG.standard_normal((15, 8))
        Q = np.linalg.qr(RNG.standard_normal((8, 8)))[0]
        base = erank(X)
        result = erank(X @ Q)
        self.assertAlmostEqual(result, base, places=8)
 
    def test_erank_increases_with_matrix_rank(self):
        """ERank = k when all k singular values are equal (analytical case).
 
        For a matrix with singular values [c, c, …, c] (k equal values)
        the normalised distribution is uniform: p_j = 1/k for j=1..k.
        Shannon entropy H = log(k), so ERank = exp(log(k)) = k exactly.
        Building matrices this way gives a deterministic, monotone sequence.
        """
        n, d = 20, 10
        U_full = np.linalg.qr(RNG.standard_normal((n, n)))[0]
        Vt_full = np.linalg.qr(RNG.standard_normal((d, d)))[0]
        values = []
        for k in range(1, d + 1):
            sigma = np.zeros(min(n, d))
            sigma[:k] = 1.0          # k equal singular values → ERank = k
            X = (U_full[:, :min(n,d)] * sigma) @ Vt_full[:min(n,d), :]
            values.append(erank(X))
        for i in range(len(values) - 1):
            self.assertAlmostEqual(
                values[i], float(i + 1), places=6,
                msg=f"ERank should be {i+1} for rank-{i+1} uniform-SV matrix"
            )
            self.assertLess(values[i], values[i + 1] - 1e-6,
                            msg=f"ERank not strictly increasing at step {i+1}")
 
    def test_zero_matrix_returns_one(self):
        """All-zero matrix (all SVs = 0) should return 1.0 via fallback."""
        X = np.zeros((10, 5))
        result = erank(X)
        self.assertAlmostEqual(result, 1.0, places=10)
 
    def test_float32_input_accepted(self):
        """float32 arrays should be handled (internally cast to float64)."""
        X = RNG.standard_normal((10, 4)).astype(np.float32)
        result = erank(X)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 1.0)
 
    def test_tall_matrix(self):
        """n > d: ERank is bounded by d."""
        X = RNG.standard_normal((100, 8))
        result = erank(X)
        self.assertLessEqual(result, 8.0 + 1e-9)
        self.assertGreaterEqual(result, 1.0)
 
    def test_wide_matrix(self):
        """n < d: ERank is bounded by n."""
        X = RNG.standard_normal((5, 100))
        result = erank(X)
        self.assertLessEqual(result, 5.0 + 1e-9)
        self.assertGreaterEqual(result, 1.0)
 
    def test_square_matrix(self):
        """n == d: ERank is bounded by n."""
        X = RNG.standard_normal((16, 16))
        result = erank(X)
        self.assertLessEqual(result, 16.0 + 1e-9)
 
    def test_1d_input_raises(self):
        with self.assertRaises(ValueError, msg="1-D input should raise"):
            erank(np.ones(10))
 
    def test_3d_input_raises(self):
        with self.assertRaises(ValueError, msg="3-D input should raise"):
            erank(np.ones((4, 4, 4)))
 
    def test_single_row_raises(self):
        with self.assertRaises(ValueError, msg="Single-row input should raise"):
            erank(np.ones((1, 8)))
 
    def test_minimum_valid_rows(self):
        """2 rows is the minimum; should not raise."""
        X = RNG.standard_normal((2, 8))
        result = erank(X)
        self.assertGreaterEqual(result, 1.0)
 
class TestLayerwiseErank(unittest.TestCase):
 
    def test_output_shape(self):
        """Output length must equal number of layers."""
        layers = _make_hidden(n_layers=6, n_tok=12, d=16)
        result = layerwise_erank(layers)
        self.assertEqual(result.shape, (6,))
 
    def test_output_dtype(self):
        layers = _make_hidden(n_layers=4, n_tok=10, d=8)
        result = layerwise_erank(layers)
        self.assertEqual(result.dtype, np.float64)
 
    def test_each_element_matches_direct_erank(self):
        """Every element must equal erank() applied to the same matrix."""
        layers = _make_hidden(n_layers=5, n_tok=15, d=12)
        result = layerwise_erank(layers)
        for i, X in enumerate(layers):
            expected = erank(X)
            self.assertAlmostEqual(result[i], expected, places=10,
                                   msg=f"Mismatch at layer {i}")
 
    def test_single_layer(self):
        """Single-element list should return a length-1 array."""
        layers = _make_hidden(n_layers=1, n_tok=10, d=8)
        result = layerwise_erank(layers)
        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(result[0], erank(layers[0]), places=10)
 
    def test_variable_layer_shapes(self):
        """Layers may have different (n, d); each is handled independently."""
        layers = [
            RNG.standard_normal((10, 32)),
            RNG.standard_normal((15, 64)),
            RNG.standard_normal((8, 16)),
        ]
        result = layerwise_erank(layers)
        self.assertEqual(result.shape, (3,))
        for i, X in enumerate(layers):
            self.assertAlmostEqual(result[i], erank(X), places=10)
 
    def test_all_values_in_valid_range(self):
        layers = _make_hidden(n_layers=8, n_tok=20, d=32)
        result = layerwise_erank(layers)
        self.assertTrue(np.all(result >= 1.0 - 1e-9))
        self.assertTrue(np.all(result <= 32.0 + 1e-9))
 
class TestAverageErank(unittest.TestCase):
 
    def test_equals_mean_of_layerwise(self):
        """Must exactly equal mean(layerwise_erank(hidden_states))."""
        layers = _make_hidden(n_layers=6, n_tok=12, d=16)
        expected = float(layerwise_erank(layers).mean())
        result = average_erank(layers)
        self.assertAlmostEqual(result, expected, places=12)
 
    def test_single_layer_equals_layer_erank(self):
        """With one layer, average_erank should equal erank of that layer."""
        layers = _make_hidden(n_layers=1, n_tok=10, d=8)
        expected = erank(layers[0])
        result = average_erank(layers)
        self.assertAlmostEqual(result, expected, places=10)
 
    def test_returns_float(self):
        layers = _make_hidden(n_layers=4, n_tok=12, d=16)
        result = average_erank(layers)
        self.assertIsInstance(result, float)
 
    def test_more_diverse_hidden_states_score_higher(self):
        """Fully random (high-diversity) activations should outscore
        low-rank (low-diversity) activations."""
        diverse = _make_hidden(n_layers=4, n_tok=20, d=32, diversity=1.0)
        sparse  = _make_hidden(n_layers=4, n_tok=20, d=32, diversity=0.1)
        self.assertGreater(average_erank(diverse), average_erank(sparse))
 
    def test_consistent_across_independent_calls(self):
        """average_erank is deterministic: same input → same output."""
        layers = _make_hidden(n_layers=4, n_tok=10, d=8)
        r1 = average_erank(layers)
        r2 = average_erank(layers)
        self.assertEqual(r1, r2)
 
class TestPaperEffectiveRankScorer(unittest.TestCase):
 
    def _make_batch(self, n_samples: int, n_layers: int = 4,
                    n_tok: int = 12, d: int = 16,
                    diversity: float = 1.0) -> list[list[np.ndarray]]:
        return [
            _make_hidden(n_layers, n_tok, d, diversity)
            for _ in range(n_samples)
        ]
 
    def test_output_columns(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(3)
        df = scorer.score([0, 1, 2], [0, 0, 1],
                          metadata={"hidden_states": hs})
        for col in ("sample_id", "label", "score", "rank", "method"):
            self.assertIn(col, df.columns, msg=f"Missing column: {col}")
 
    def test_output_row_count(self):
        scorer = EffectiveRankScorer()
        n = 5
        hs = self._make_batch(n)
        df = scorer.score(list(range(n)), [0] * n,
                          metadata={"hidden_states": hs})
        self.assertEqual(len(df), n)
 
    def test_method_column_value(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(3)
        df = scorer.score([0, 1, 2], [0, 0, 0],
                          metadata={"hidden_states": hs})
        self.assertTrue((df["method"] == "paper_erank").all())
 
    def test_sample_ids_preserved(self):
        scorer = EffectiveRankScorer()
        ids = [10, 20, 30, 40]
        hs = self._make_batch(4)
        df = scorer.score(ids, [0] * 4, metadata={"hidden_states": hs})
        self.assertEqual(set(df["sample_id"].tolist()), set(ids))
 
    def test_labels_preserved(self):
        """Each (sample_id, label) pair must survive the round-trip."""
        scorer = EffectiveRankScorer()
        ids    = [10, 20, 30, 40]
        labels = [1, 0, 2, 1]
        hs = self._make_batch(4)
        df = scorer.score(ids, labels, metadata={"hidden_states": hs})
        id_to_label_expected = dict(zip(ids, labels))
        for _, row in df.iterrows():
            self.assertEqual(
                row["label"], id_to_label_expected[row["sample_id"]],
                msg=f"Label mismatch for sample_id={row['sample_id']}"
            )
 
    def test_ranks_are_one_based(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(5)
        df = scorer.score(list(range(5)), [0] * 5,
                          metadata={"hidden_states": hs})
        self.assertEqual(df["rank"].min(), 1)
 
    def test_ranks_are_unique_and_contiguous(self):
        scorer = EffectiveRankScorer()
        n = 6
        hs = self._make_batch(n)
        df = scorer.score(list(range(n)), [0] * n,
                          metadata={"hidden_states": hs})
        self.assertEqual(sorted(df["rank"].tolist()), list(range(1, n + 1)))
 
    def test_rank_1_has_highest_score(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(5)
        df = scorer.score(list(range(5)), [0] * 5,
                          metadata={"hidden_states": hs})
        top = df[df["rank"] == 1]["score"].values[0]
        self.assertAlmostEqual(top, df["score"].max(), places=10)
 
    def test_sorted_ascending_by_rank(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(5)
        df = scorer.score(list(range(5)), [0] * 5,
                          metadata={"hidden_states": hs})
        self.assertEqual(df["rank"].tolist(),
                         sorted(df["rank"].tolist()))
 
    def test_higher_diversity_ranks_first(self):
        """A sample with isotropic activations must rank above a low-rank one."""
        scorer = EffectiveRankScorer()
        diverse_hs = _make_hidden(4, 20, 32, diversity=1.0)
        sparse_hs  = _make_hidden(4, 20, 32, diversity=0.05)
        hs = [diverse_hs, sparse_hs]
        df = scorer.score([0, 1], [0, 0], metadata={"hidden_states": hs})
        rank_diverse = df[df["sample_id"] == 0]["rank"].values[0]
        rank_sparse  = df[df["sample_id"] == 1]["rank"].values[0]
        self.assertLess(rank_diverse, rank_sparse,
                        msg="Diverse sample should have lower (better) rank number")
 
    def test_scores_match_average_erank(self):
        """Scores in the DataFrame must equal average_erank() directly."""
        scorer = EffectiveRankScorer()
        hs_list = self._make_batch(4)
        df = scorer.score(list(range(4)), [0] * 4,
                          metadata={"hidden_states": hs_list})
        df_sorted = df.sort_values("sample_id").reset_index(drop=True)
        for i, hs in enumerate(hs_list):
            expected = average_erank(hs)
            got = df_sorted.iloc[i]["score"]
            self.assertAlmostEqual(got, expected, places=10,
                                   msg=f"Score mismatch at sample {i}")
 
    def test_name_property(self):
        self.assertEqual(EffectiveRankScorer().name, "paper_erank")
 
    def test_metadata_populated_after_score(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(3)
        scorer.score([0, 1, 2], [0, 0, 0], metadata={"hidden_states": hs})
        meta = scorer.build_metadata()
        for key in ("method", "n_samples", "n_layers",
                    "mean_score", "min_score", "max_score"):
            self.assertIn(key, meta, msg=f"Missing metadata key: {key}")
 
    def test_metadata_is_copy(self):
        """build_metadata() must return a copy, not the internal dict."""
        scorer = EffectiveRankScorer()
        hs = self._make_batch(2)
        scorer.score([0, 1], [0, 0], metadata={"hidden_states": hs})
        meta1 = scorer.build_metadata()
        meta1["injected"] = True
        meta2 = scorer.build_metadata()
        self.assertNotIn("injected", meta2)
 
    def test_metadata_n_samples_correct(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(7)
        scorer.score(list(range(7)), [0] * 7, metadata={"hidden_states": hs})
        self.assertEqual(scorer.build_metadata()["n_samples"], 7)
 
    def test_metadata_n_layers_correct(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(3, n_layers=5)
        scorer.score([0, 1, 2], [0, 0, 0], metadata={"hidden_states": hs})
        self.assertEqual(scorer.build_metadata()["n_layers"], 5)
 
    def test_metadata_score_stats_consistent(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(4)
        df = scorer.score(list(range(4)), [0] * 4,
                          metadata={"hidden_states": hs})
        meta = scorer.build_metadata()
        self.assertAlmostEqual(meta["mean_score"], df["score"].mean(), places=10)
        self.assertAlmostEqual(meta["min_score"],  df["score"].min(),  places=10)
        self.assertAlmostEqual(meta["max_score"],  df["score"].max(),  places=10)
 
    def test_missing_metadata_raises(self):
        scorer = EffectiveRankScorer()
        with self.assertRaises(ValueError):
            scorer.score([0, 1], [0, 0], metadata=None)
 
    def test_missing_hidden_states_key_raises(self):
        scorer = EffectiveRankScorer()
        with self.assertRaises(ValueError):
            scorer.score([0], [0], metadata={"features": np.zeros((1, 4))})
 
    def test_length_mismatch_raises(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(3)
        with self.assertRaises(ValueError):
            scorer.score([0, 1], [0, 0], metadata={"hidden_states": hs})
 
    def test_single_sample(self):
        scorer = EffectiveRankScorer()
        hs = self._make_batch(1)
        df = scorer.score([42], [7], metadata={"hidden_states": hs})
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["rank"], 1)
        self.assertEqual(df.iloc[0]["sample_id"], 42)
        self.assertEqual(df.iloc[0]["label"], 7)
 
    def test_large_batch_runs_without_error(self):
        scorer = EffectiveRankScorer()
        n = 100
        hs = self._make_batch(n, n_layers=8, n_tok=50, d=64)
        df = scorer.score(list(range(n)), [0] * n,
                          metadata={"hidden_states": hs})
        self.assertEqual(len(df), n)
        self.assertEqual(sorted(df["rank"].tolist()), list(range(1, n + 1)))
 
class TestRankGenerators(unittest.TestCase):
 
    def _gen_data(self, names_diversity: dict) -> dict:
        """Build generator_hidden_states from a {name: diversity} mapping."""
        return {
            name: [
                _make_hidden(n_layers=4, n_tok=15, d=24, diversity=div)
                for _ in range(5)       # 5 texts per generator
            ]
            for name, div in names_diversity.items()
        }
 
    def test_output_columns(self):
        data = self._gen_data({"A": 1.0, "B": 0.5})
        df = rank_generators(data)
        for col in ("generator", "mean_erank", "std_erank", "n_texts", "rank"):
            self.assertIn(col, df.columns)
 
    def test_output_row_count(self):
        data = self._gen_data({"A": 1.0, "B": 0.5, "C": 0.2})
        df = rank_generators(data)
        self.assertEqual(len(df), 3)
 
    def test_ranks_are_one_based_contiguous(self):
        data = self._gen_data({"X": 1.0, "Y": 0.5, "Z": 0.1})
        df = rank_generators(data)
        self.assertEqual(sorted(df["rank"].tolist()), [1, 2, 3])
 
    def test_sorted_by_mean_erank_descending(self):
        data = self._gen_data({"A": 1.0, "B": 0.5, "C": 0.1})
        df = rank_generators(data)
        eranks = df["mean_erank"].tolist()
        self.assertEqual(eranks, sorted(eranks, reverse=True),
                         msg="DataFrame should be sorted by mean_erank desc")
 
    def test_diverse_generator_ranks_first(self):
        """Generator with more isotropic activations must get rank 1."""
        data = self._gen_data({"diverse": 1.0, "sparse": 0.05})
        df = rank_generators(data)
        rank_diverse = df[df["generator"] == "diverse"]["rank"].values[0]
        self.assertEqual(rank_diverse, 1,
                         msg="Most diverse generator should be rank 1")
 
    def test_three_generators_ordered_correctly(self):
        """Three generators with clearly separated diversity → correct order."""
        data = self._gen_data({"high": 1.0, "mid": 0.3, "low": 0.05})
        df = rank_generators(data)
        rank = dict(zip(df["generator"], df["rank"]))
        self.assertLess(rank["high"], rank["mid"])
        self.assertLess(rank["mid"],  rank["low"])
 
    def test_n_texts_column_correct(self):
        data = self._gen_data({"A": 1.0, "B": 0.5})
        df = rank_generators(data)
        self.assertTrue((df["n_texts"] == 5).all())
 
    def test_mean_erank_matches_manual(self):
        """mean_erank in output must equal manually computed average."""
        hs_list = [_make_hidden(4, 15, 24, 1.0) for _ in range(3)]
        data = {"gen": hs_list}
        df = rank_generators(data)
        expected = float(np.mean([average_erank(hs) for hs in hs_list]))
        got = df.iloc[0]["mean_erank"]
        self.assertAlmostEqual(got, expected, places=10)
 
    def test_single_generator(self):
        data = self._gen_data({"only_one": 1.0})
        df = rank_generators(data)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["rank"], 1)
 
    def test_generator_names_preserved(self):
        names = {"alpha": 1.0, "beta": 0.5, "gamma": 0.2}
        data = self._gen_data(names)
        df = rank_generators(data)
        self.assertEqual(set(df["generator"].tolist()), set(names.keys()))
 
class TestIntegration(unittest.TestCase):
 
    def test_scorer_and_rank_generators_agree(self):
        """PaperEffectiveRankScorer and rank_generators must produce the
        same ordering when given equivalent inputs."""
        rng = np.random.default_rng(99)
 
        gen_names = ["A", "B", "C"]
        gen_hs = {
            name: [_make_hidden(4, 15, 20, div, rng=rng) for _ in range(3)]
            for name, div in zip(gen_names, [1.0, 0.5, 0.1])
        }
 
        rg_df = rank_generators(gen_hs)
        rg_order = rg_df["generator"].tolist()
 
        rep_hs = [gen_hs[name][0] for name in gen_names]
        scorer = EffectiveRankScorer()
        sc_df = scorer.score(
            sample_ids=list(range(len(gen_names))),
            labels=[0] * len(gen_names),
            metadata={"hidden_states": rep_hs},
        )
        sc_order = [
            gen_names[sc_df.sort_values("rank").iloc[i]["sample_id"]]
            for i in range(len(gen_names))
        ]
 
        self.assertEqual(rg_order, sc_order,
                         msg="scorer and rank_generators disagree on ordering")
 
    def test_layerwise_average_consistency(self):
        """average_erank must equal sum(layerwise_erank) / L."""
        for n_layers in [1, 3, 8]:
            hs = _make_hidden(n_layers, 12, 16)
            lw = layerwise_erank(hs)
            avg = average_erank(hs)
            self.assertAlmostEqual(avg, lw.sum() / n_layers, places=12,
                                   msg=f"Inconsistency at n_layers={n_layers}")
 
 
if __name__ == "__main__":
    unittest.main(verbosity=2)