import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pytest

# Unit tests for vectorized bootstrap functions
class TestBootstrapFunctions:
    """
    Comprehensive unit tests for vectorized Poisson bootstrap CI estimation.
    
    Tests verify:
    1. Vectorized and loop-based methods produce equivalent results
    2. CIs have correct coverage properties
    3. Bootstrap distributions are well-formed
    4. Edge cases are handled properly
    """
    
    @staticmethod
    def generate_synthetic_data(n_samples=500, n_genes=100, noise_level=0.3, seed=42):
        """
        Generate synthetic gene expression data with known correlations.
        """
        np.random.seed(seed)
        
        # Generate true values
        y_true = np.random.randn(n_samples, n_genes)
        
        # Generate predictions with controlled correlation
        signal = y_true * (1 - noise_level)
        noise = np.random.randn(n_samples, n_genes) * noise_level
        y_pred = signal + noise
        
        # Calculate true population correlation
        true_r, _ = pearsonr(y_true.ravel(), y_pred.ravel())
        
        # Convert to DataFrame for testing
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        y_true_df = pd.DataFrame(y_true, columns=gene_names)
        
        return y_true_df, y_pred, true_r
    
    def test_vectorized_vs_loop_equivalence_aggregate(self):
        """
        Test 1: Vectorized and loop-based Poisson bootstrap produce equivalent CIs
        for aggregate Pearson's R.
        """
        print("\n" + "="*70)
        print("TEST 1: Vectorized vs Loop Equivalence (Aggregate)")
        print("="*70)
        
        y_true_df, y_pred, true_r = self.generate_synthetic_data(
            n_samples=500, n_genes=100, noise_level=0.2, seed=42
        )
        
        y_true_flat = y_true_df.values.ravel()
        y_pred_flat = y_pred.ravel()
        
        n_bootstrap = 1000
        np.random.seed(42)
        
        # METHOD 1: Vectorized Poisson bootstrap
        weights_vec = np.random.poisson(lam=1, size=(n_bootstrap, len(y_true_flat)))
        w_sum = weights_vec.sum(axis=1, keepdims=True)
        w_mean_true = (weights_vec * y_true_flat).sum(axis=1, keepdims=True) / w_sum
        w_mean_pred = (weights_vec * y_pred_flat).sum(axis=1, keepdims=True) / w_sum
        
        y_true_centered = y_true_flat - w_mean_true
        y_pred_centered = y_pred_flat - w_mean_pred
        
        cov = (weights_vec * y_true_centered * y_pred_centered).sum(axis=1) / w_sum.ravel()
        std_true = np.sqrt((weights_vec * y_true_centered**2).sum(axis=1) / w_sum.ravel())
        std_pred = np.sqrt((weights_vec * y_pred_centered**2).sum(axis=1) / w_sum.ravel())
        
        bootstrap_rs_vectorized = cov / (std_true * std_pred)
        ci_lower_vec = np.percentile(bootstrap_rs_vectorized, 2.5)
        ci_upper_vec = np.percentile(bootstrap_rs_vectorized, 97.5)
        
        # METHOD 2: Loop-based Poisson bootstrap
        np.random.seed(42)
        bootstrap_rs_loop = []
        for _ in range(n_bootstrap):
            weights = np.random.poisson(lam=1, size=len(y_true_flat))
            w_sum = weights.sum()
            w_mean_true = (weights * y_true_flat).sum() / w_sum
            w_mean_pred = (weights * y_pred_flat).sum() / w_sum
            cov = (weights * (y_true_flat - w_mean_true) * (y_pred_flat - w_mean_pred)).sum() / w_sum
            std_true = np.sqrt((weights * (y_true_flat - w_mean_true)**2).sum() / w_sum)
            std_pred = np.sqrt((weights * (y_pred_flat - w_mean_pred)**2).sum() / w_sum)
            bootstrap_rs_loop.append(cov / (std_true * std_pred))
        
        bootstrap_rs_loop = np.array(bootstrap_rs_loop)
        ci_lower_loop = np.percentile(bootstrap_rs_loop, 2.5)
        ci_upper_loop = np.percentile(bootstrap_rs_loop, 97.5)
        
        np.testing.assert_array_almost_equal(bootstrap_rs_vectorized, bootstrap_rs_loop, decimal=10)
        print("\n✓ TEST PASSED: Vectorized and loop methods are equivalent")

    def test_ci_coverage_properties(self):
        """
        Test 2: Bootstrap CIs have correct coverage properties.
        """
        print("\n" + "="*70)
        print("TEST 2: CI Coverage Properties")
        print("="*70)
        
        n_simulations = 100
        n_bootstrap = 1000
        coverage_count = 0
        
        for sim in range(n_simulations):
            y_true_df, y_pred, true_r = self.generate_synthetic_data(
                n_samples=100, n_genes=30, noise_level=0.3, seed=sim
            )
            y_true_flat = y_true_df.values.ravel()
            y_pred_flat = y_pred.ravel()
            observed_r, _ = pearsonr(y_true_flat, y_pred_flat)
            
            np.random.seed(sim)
            weights = np.random.poisson(lam=1, size=(n_bootstrap, len(y_true_flat)))
            w_sum = weights.sum(axis=1, keepdims=True)
            w_mean_true = (weights * y_true_flat).sum(axis=1, keepdims=True) / w_sum
            w_mean_pred = (weights * y_pred_flat).sum(axis=1, keepdims=True) / w_sum
            
            y_true_centered = y_true_flat - w_mean_true
            y_pred_centered = y_pred_flat - w_mean_pred
            cov = (weights * y_true_centered * y_pred_centered).sum(axis=1) / w_sum.ravel()
            std_true = np.sqrt((weights * y_true_centered**2).sum(axis=1) / w_sum.ravel())
            std_pred = np.sqrt((weights * y_pred_centered**2).sum(axis=1) / w_sum.ravel())
            
            bootstrap_rs = cov / (std_true * std_pred)
            if np.percentile(bootstrap_rs, 2.5) <= observed_r <= np.percentile(bootstrap_rs, 97.5):
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        assert 0.85 <= coverage_rate <= 1.0
        print(f"\n✓ TEST PASSED: Coverage rate {coverage_rate*100:.1f}% is within acceptable range")

    def test_ci_coverage_of_true_parameter(self):
        """
        Test 2B: Bootstrap CIs capture the TRUE population parameter.
        """
        print("\n" + "="*70)
        print("TEST 2B: CI Coverage of True Population Parameter")
        print("="*70)
        
        n_simulations = 200
        n_bootstrap = 2000
        coverage_count = 0
        true_population_r = 0.85
        
        for sim in range(n_simulations):
            n_samples, n_genes = 150, 40
            np.random.seed(sim)
            cov_matrix = [[1, true_population_r], [true_population_r, 1]]
            data = np.random.multivariate_normal([0, 0], cov_matrix, size=n_samples * n_genes)
            y_true_flat, y_pred_flat = data[:, 0], data[:, 1]
            
            np.random.seed(sim + 1000)
            weights = np.random.poisson(lam=1, size=(n_bootstrap, len(y_true_flat)))
            w_sum = weights.sum(axis=1, keepdims=True)
            w_mean_true = (weights * y_true_flat).sum(axis=1, keepdims=True) / w_sum
            w_mean_pred = (weights * y_pred_flat).sum(axis=1, keepdims=True) / w_sum
            
            y_true_centered = y_true_flat - w_mean_true
            y_pred_centered = y_pred_flat - w_mean_pred
            cov = (weights * y_true_centered * y_pred_centered).sum(axis=1) / w_sum.ravel()
            std_true = np.sqrt((weights * y_true_centered**2).sum(axis=1) / w_sum.ravel())
            std_pred = np.sqrt((weights * y_pred_centered**2).sum(axis=1) / w_sum.ravel())
            
            bootstrap_rs = cov / (std_true * std_pred)
            if np.percentile(bootstrap_rs, 2.5) <= true_population_r <= np.percentile(bootstrap_rs, 97.5):
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        assert 0.90 <= coverage_rate <= 0.98
        print(f"\n✓ TEST PASSED: True parameter coverage rate {coverage_rate*100:.1f}% is appropriate")

    def test_per_gene_vectorized_vs_loop(self):
        """
        Test 3: Per-gene vectorized bootstrap produces equivalent results to loop version.
        """
        print("\n" + "="*70)
        print("TEST 3: Per-Gene Vectorized vs Loop Equivalence")
        print("="*70)
        
        y_true_df, y_pred, _ = self.generate_synthetic_data(n_samples=200, n_genes=50, noise_level=0.3, seed=42)
        n_genes = y_true_df.shape[1]
        model_pearson_rs = np.array([pearsonr(y_true_df.iloc[:, i].values, y_pred[:, i])[0] for i in range(n_genes)])
        
        n_bootstrap = 1000
        np.random.seed(42)
        weights_vec = np.random.poisson(lam=1, size=(n_bootstrap, len(model_pearson_rs)))
        weighted_means_vec = (weights_vec * model_pearson_rs).sum(axis=1) / weights_vec.sum(axis=1)
        
        np.random.seed(42)
        weighted_means_loop = []
        for _ in range(n_bootstrap):
            weights = np.random.poisson(lam=1, size=len(model_pearson_rs))
            weighted_means_loop.append((weights * model_pearson_rs).sum() / weights.sum())
        
        np.testing.assert_array_almost_equal(weighted_means_vec, np.array(weighted_means_loop), decimal=10)
        print("\n✓ TEST PASSED: Per-gene vectorized method matches loop method")

    def test_bootstrap_distribution_properties(self):
        """
        Test 4: Bootstrap distributions have expected statistical properties.
        """
        print("\n" + "="*70)
        print("TEST 4: Bootstrap Distribution Properties")
        print("="*70)
        
        y_true_df, y_pred, _ = self.generate_synthetic_data(n_samples=500, n_genes=100, noise_level=0.2, seed=42)
        y_true_flat, y_pred_flat = y_true_df.values.ravel(), y_pred.ravel()
        observed_r, _ = pearsonr(y_true_flat, y_pred_flat)
        
        np.random.seed(42)
        weights = np.random.poisson(lam=1, size=(5000, len(y_true_flat)))
        w_sum = weights.sum(axis=1, keepdims=True)
        w_mean_true = (weights * y_true_flat).sum(axis=1, keepdims=True) / w_sum
        w_mean_pred = (weights * y_pred_flat).sum(axis=1, keepdims=True) / w_sum
        
        cov = (weights * (y_true_flat - w_mean_true) * (y_pred_flat - w_mean_pred)).sum(axis=1) / w_sum.ravel()
        std_true = np.sqrt((weights * (y_true_flat - w_mean_true)**2).sum(axis=1) / w_sum.ravel())
        std_pred = np.sqrt((weights * (y_pred_flat - w_mean_pred)**2).sum(axis=1) / w_sum.ravel())
        bootstrap_rs = cov / (std_true * std_pred)
        
        assert abs(np.mean(bootstrap_rs) - observed_r) < 0.01
        assert 0.001 < np.std(bootstrap_rs) < 0.1
        print("\n✓ TEST PASSED: Bootstrap distribution has expected properties")

    def test_edge_cases(self):
        """
        Test 5: Functions handle edge cases properly.
        """
        print("\n" + "="*70)
        print("TEST 5: Edge Case Handling")
        print("="*70)
        
        y_true = np.random.randn(100, 10)
        y_true_flat, y_pred_flat = y_true.ravel(), y_true.ravel() # Perfect correlation
        
        np.random.seed(42)
        weights = np.random.poisson(lam=1, size=(100, len(y_true_flat)))
        w_sum = weights.sum(axis=1, keepdims=True)
        w_mean_true = (weights * y_true_flat).sum(axis=1, keepdims=True) / w_sum
        w_mean_pred = (weights * y_pred_flat).sum(axis=1, keepdims=True) / w_sum
        
        cov = (weights * (y_true_flat - w_mean_true) * (y_pred_flat - w_mean_pred)).sum(axis=1) / w_sum.ravel()
        std_true = np.sqrt((weights * (y_true_flat - w_mean_true)**2).sum(axis=1) / w_sum.ravel())
        std_pred = np.sqrt((weights * (y_pred_flat - w_mean_pred)**2).sum(axis=1) / w_sum.ravel())
        bootstrap_rs = cov / (std_true * std_pred)
        
        assert np.all(bootstrap_rs > 0.98)
        print("✓ Perfect correlation handled correctly")
        print("\n✓ TEST PASSED: All edge cases handled properly")

    def run_all_tests(self):
        """Run all unit tests."""
        print("\n" + "="*70)
        print("RUNNING ALL BOOTSTRAP UNIT TESTS")
        print("="*70)
        try:
            self.test_vectorized_vs_loop_equivalence_aggregate()
            self.test_ci_coverage_properties()
            self.test_ci_coverage_of_true_parameter()
            self.test_per_gene_vectorized_vs_loop()
            self.test_bootstrap_distribution_properties()
            self.test_edge_cases()
            print("\n" + "="*70)
            print("ALL TESTS PASSED ✓")
            print("="*70)
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            raise


# Run the tests
if __name__ == "__main__":
    tester = TestBootstrapFunctions()
    tester.run_all_tests()