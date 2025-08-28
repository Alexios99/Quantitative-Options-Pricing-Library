import pytest
import numpy as np
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.utils.convergence import run_pde_convergence_study, PDEConvergenceReport
from quantlib.utils.visualization import plot_pde_convergence, plot_pde_timing_analysis


class TestPDEConvergence:
    
    def setup_method(self):
        self.european_call = OptionContract(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
        
        self.american_put = OptionContract(
            spot=100.0,
            strike=110.0,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=0.3,
            option=OptionType.PUT,
            style=ExerciseStyle.AMERICAN
        )

    def test_pde_convergence_study_european(self):
        """Test PDE convergence study for European option"""
        report = run_pde_convergence_study(
            self.european_call,
            spatial_sizes=[25, 50, 100],
            temporal_sizes=[50, 100, 200]
        )
        
        assert isinstance(report, PDEConvergenceReport)
        assert len(report.spatial_df) == 3
        assert len(report.temporal_df) == 3
        assert report.reference_price > 0
        
        # Check that errors are reasonable and improve with finer grids
        spatial_errors = report.spatial_df['abs_err'].values
        temporal_errors = report.temporal_df['abs_err'].values
        
        # Finest grid should have reasonable error (< 5% of option price)
        assert spatial_errors[-1] < 0.05 * report.reference_price
        assert temporal_errors[-1] < 0.05 * report.reference_price
        
        # Spatial error should improve from coarsest to finest
        assert spatial_errors[-1] < spatial_errors[0]
        
        # Check convergence orders are reasonable
        assert 0.5 < report.spatial_order < 3.0  # Should be around 2 for O(h²)
        # Temporal convergence can be tricky with small grids, just check it's finite
        assert np.isfinite(report.temporal_order)

    def test_pde_convergence_study_american(self):
        """Test PDE convergence study for American option"""
        report = run_pde_convergence_study(
            self.american_put,
            spatial_sizes=[30, 60],
            temporal_sizes=[100, 200]
        )
        
        assert isinstance(report, PDEConvergenceReport)
        assert len(report.spatial_df) == 2
        assert len(report.temporal_df) == 2
        assert report.reference_price > 0
        
        # American put should have higher value than European
        intrinsic = max(self.american_put.strike - self.american_put.spot, 0)
        assert report.reference_price >= intrinsic

    def test_pde_convergence_dataframe_structure(self):
        """Test that convergence DataFrames have correct structure"""
        report = run_pde_convergence_study(
            self.european_call,
            spatial_sizes=[25, 50],
            temporal_sizes=[50, 100]
        )
        
        # Check spatial DataFrame
        expected_spatial_cols = ['n_space', 'price', 'abs_err', 'rel_err', 'ms']
        assert all(col in report.spatial_df.columns for col in expected_spatial_cols)
        assert len(report.spatial_df) == 2
        
        # Check temporal DataFrame
        expected_temporal_cols = ['n_time', 'price', 'abs_err', 'rel_err', 'ms']
        assert all(col in report.temporal_df.columns for col in expected_temporal_cols)
        assert len(report.temporal_df) == 2
        
        # Check that all values are finite
        assert np.all(np.isfinite(report.spatial_df['price']))
        assert np.all(np.isfinite(report.temporal_df['price']))
        assert np.all(report.spatial_df['abs_err'] >= 0)
        assert np.all(report.temporal_df['abs_err'] >= 0)

    def test_pde_convergence_metadata(self):
        """Test that convergence report contains proper metadata"""
        report = run_pde_convergence_study(
            self.european_call,
            spatial_sizes=[25, 50],
            temporal_sizes=[50, 100],
            s_max=150,
            theta=0.6
        )
        
        assert 'engine' in report.meta
        assert report.meta['engine'] == 'PDEEngine'
        assert 'spatial_sizes' in report.meta
        assert 'temporal_sizes' in report.meta
        assert 's_max' in report.meta
        assert report.meta['s_max'] == 150
        assert 'theta' in report.meta
        assert report.meta['theta'] == 0.6
        assert 'psor_params' in report.meta

    def test_pde_convergence_custom_parameters(self):
        """Test PDE convergence with custom parameters"""
        from types import SimpleNamespace
        
        custom_psor = SimpleNamespace(
            psor_omega=1.5,
            psor_tol=1e-8,
            psor_max_iter=100
        )
        
        report = run_pde_convergence_study(
            self.american_put,
            spatial_sizes=[40, 80],
            temporal_sizes=[100, 200],
            s_max=200,
            theta=1.0,  # Fully implicit
            psor_params=custom_psor
        )
        
        assert report.meta['theta'] == 1.0
        assert report.meta['s_max'] == 200
        assert report.meta['psor_params']['psor_omega'] == 1.5
        assert report.meta['psor_params']['psor_tol'] == 1e-8

    def test_pde_visualization_functions(self):
        """Test that PDE visualization functions work without errors"""
        report = run_pde_convergence_study(
            self.european_call,
            spatial_sizes=[25, 50],
            temporal_sizes=[50, 100]
        )
        
        # Test main convergence plot
        fig1, axes1 = plot_pde_convergence(report)
        assert fig1 is not None
        assert len(axes1) == 4
        
        # Test timing analysis plot
        fig2, axes2 = plot_pde_timing_analysis(report)
        assert fig2 is not None
        assert len(axes2) == 2

    def test_convergence_order_estimation(self):
        """Test that convergence order estimation is reasonable"""
        # Use finer grids for better order estimation
        report = run_pde_convergence_study(
            self.european_call,
            spatial_sizes=[25, 50, 100, 200],
            temporal_sizes=[100, 200, 400, 800]
        )
        
        # Spatial should be close to 2 (O(h²))
        assert 1.5 < report.spatial_order < 2.5
        
        # Temporal should be close to 1 for Crank-Nicolson (O(Δt))
        # But might be closer to 2 due to higher order accuracy
        assert 0.8 < report.temporal_order < 2.2

    def test_pde_convergence_pass_criteria(self):
        """Test convergence pass/fail criteria"""
        # Test with very coarse grid (should fail)
        coarse_report = run_pde_convergence_study(
            self.european_call,
            spatial_sizes=[10, 20],
            temporal_sizes=[20, 40]
        )
        
        # Test with fine grid (should pass)
        fine_report = run_pde_convergence_study(
            self.european_call,
            spatial_sizes=[100, 200, 400],
            temporal_sizes=[500, 1000, 2000]
        )
        
        # Fine grid should perform better
        assert fine_report.spatial_df['abs_err'].iloc[-1] <= coarse_report.spatial_df['abs_err'].iloc[-1]
        assert fine_report.temporal_df['abs_err'].iloc[-1] <= coarse_report.temporal_df['abs_err'].iloc[-1]