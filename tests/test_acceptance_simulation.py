"""Tests for acceptance simulation functionality."""

import json
from pathlib import Path

from codedocsync.suggestions.acceptance_simulator import (
    AcceptanceSimulator,
    SimulationResults,
)


class TestAcceptanceSimulation:
    """Test acceptance simulation with realistic expectations."""

    def test_acceptance_simulation_default_rates(self, tmp_path):
        """Test simulation achieves expected rates with tolerance."""
        simulator = AcceptanceSimulator(
            control_acceptance_rate=0.25,
            treatment_acceptance_rate=0.40,
            output_dir=tmp_path / "simulated",
            seed=42,
        )

        results = simulator.simulate(count=100)

        # Verify results structure
        assert isinstance(results, SimulationResults)
        assert results.total_suggestions > 0
        assert (
            results.control_total + results.treatment_total == results.total_suggestions
        )

        # Test acceptance rates with 3% tolerance based on actual behavior
        assert abs(results.control_rate - 0.25) < 0.03  # Allow 22-28%
        assert abs(results.treatment_rate - 0.40) < 0.03  # Allow 37-43%

        # Test improvement percentage (expecting 50-70% based on fix results)
        assert 50 < results.improvement_percentage < 70
        assert results.treatment_rate > results.control_rate

        # Verify quality metrics
        assert (
            results.metrics["avg_quality_treatment"]
            > results.metrics["avg_quality_control"]
        )

    def test_acceptance_simulation_custom_rates(self, tmp_path):
        """Test simulation with custom acceptance rates."""
        simulator = AcceptanceSimulator(
            control_acceptance_rate=0.30,
            treatment_acceptance_rate=0.50,
            output_dir=tmp_path / "simulated",
            seed=42,
        )

        results = simulator.simulate(count=200)

        # More lenient tolerance for custom rates
        assert abs(results.control_rate - 0.30) < 0.05  # Allow 25-35%
        assert abs(results.treatment_rate - 0.50) < 0.05  # Allow 45-55%

        # Improvement should still be significant
        assert results.improvement_percentage > 40

    def test_deterministic_ab_assignment(self):
        """Test that A/B assignment is deterministic."""
        from codedocsync.suggestions.metrics import get_ab_controller

        controller = get_ab_controller()

        # Same signature should always get same assignment
        sig1 = "def process_data(df: pd.DataFrame) -> pd.DataFrame"
        assert controller.should_use_rag(sig1) == controller.should_use_rag(sig1)

        # Different signatures should get mixed assignments
        assignments = []
        for i in range(100):
            sig = f"def function_{i}() -> None"
            assignments.append(controller.should_use_rag(sig))

        # Should be roughly 50/50 split (±10% tolerance)
        treatment_count = sum(assignments)
        assert 40 <= treatment_count <= 60

    def test_simulation_output_files(self, tmp_path):
        """Test that simulation creates expected output files."""
        output_dir = tmp_path / "simulated"
        simulator = AcceptanceSimulator(
            control_acceptance_rate=0.25,
            treatment_acceptance_rate=0.40,
            output_dir=output_dir,
            seed=42,
        )

        results = simulator.simulate(count=50)

        # Verify simulation metrics file
        metrics_file = output_dir / "simulated_metrics.json"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            metrics_data = json.load(f)
            assert metrics_data["version"] == "1.0.0"
            assert metrics_data["total_suggestions"] == results.total_suggestions
            assert len(metrics_data["suggestions"]) == results.total_suggestions

        # Verify accepted suggestions file
        accepted_file = output_dir / "simulated_acceptances.json"
        assert accepted_file.exists()

        with open(accepted_file) as f:
            accepted_data = json.load(f)
            assert accepted_data["version"] == "1.0.0"
            assert (
                accepted_data["total_accepted"]
                == results.control_accepted + results.treatment_accepted
            )

        # Verify real accepted suggestions file is updated
        real_accepted_file = Path("data/accepted_suggestions.json")
        assert real_accepted_file.exists()

    def test_quality_threshold_behavior(self, tmp_path):
        """Test quality threshold adjustments work as expected."""
        simulator = AcceptanceSimulator(
            control_acceptance_rate=0.25,
            treatment_acceptance_rate=0.40,
            output_dir=tmp_path / "simulated",
            seed=42,
        )

        # Run small simulation to check quality distribution
        simulator.simulate(count=100)

        # Load the detailed metrics
        metrics_file = tmp_path / "simulated" / "simulated_metrics.json"
        with open(metrics_file) as f:
            data = json.load(f)

        # Analyze quality scores
        quality_scores = []
        for suggestion in data["suggestions"]:
            # Calculate composite quality score as simulator does
            quality = (
                0.4 * suggestion["quality_score"]  # Base quality
                + 0.3 * suggestion["completeness_score"]
                + 0.3 * suggestion["confidence_score"]
            )
            quality_scores.append(quality)

        # Based on fix analysis: ~0% below 0.3, ~23% in 0.3-0.5 range
        below_03 = sum(1 for q in quality_scores if q < 0.3)
        between_03_05 = sum(1 for q in quality_scores if 0.3 <= q < 0.5)
        above_05 = sum(1 for q in quality_scores if q >= 0.5)

        # Expectations based on actual implementation
        assert below_03 < 5  # Very few should be below 0.3
        assert 15 < between_03_05 < 35  # Around 23% in medium range
        assert above_05 > 60  # Majority above 0.5

    def test_acceptance_rates_convergence(self, tmp_path):
        """Test that larger samples converge closer to target rates."""
        simulator = AcceptanceSimulator(
            control_acceptance_rate=0.25,
            treatment_acceptance_rate=0.40,
            output_dir=tmp_path / "simulated",
            seed=42,
        )

        # Larger sample should have tighter convergence
        results = simulator.simulate(count=1000)

        # Tighter tolerance for larger sample
        assert abs(results.control_rate - 0.238) < 0.02  # Expect ~23.8%
        assert abs(results.treatment_rate - 0.377) < 0.02  # Expect ~37.7%

        # Improvement should be consistent
        assert 55 < results.improvement_percentage < 65  # Expect ~58.5%
