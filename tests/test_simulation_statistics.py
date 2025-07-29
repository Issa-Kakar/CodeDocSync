"""Statistical validation of simulation results."""

import numpy as np
from scipy import stats

from codedocsync.suggestions.acceptance_simulator import AcceptanceSimulator


def test_statistical_significance():
    """Test that improvement is statistically significant."""
    simulator = AcceptanceSimulator(
        control_acceptance_rate=0.25, treatment_acceptance_rate=0.40, seed=42
    )

    results = simulator.simulate(count=1000)

    # Perform chi-square test
    control_success = results.control_accepted
    control_failure = results.control_total - results.control_accepted
    treatment_success = results.treatment_accepted
    treatment_failure = results.treatment_total - results.treatment_accepted

    observed = np.array(
        [[control_success, control_failure], [treatment_success, treatment_failure]]
    )

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Should be highly significant
    assert p_value < 0.001

    # Effect size (Cramér's V)
    n = observed.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))
    assert cramers_v > 0.1  # Medium to large effect size
