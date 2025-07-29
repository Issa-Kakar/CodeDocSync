"""Integration tests for acceptance simulation with real components."""

from codedocsync.suggestions.acceptance_simulator import AcceptanceSimulator
from codedocsync.suggestions.rag_corpus import RAGCorpusManager


class TestSimulationIntegration:
    """Test simulation integration with RAG corpus."""

    def test_simulation_updates_rag_corpus(self, tmp_path):
        """Test that accepted suggestions are added to RAG corpus."""
        # Get initial corpus size
        rag_manager = RAGCorpusManager()
        initial_size = len(rag_manager.memory_corpus)

        # Run simulation
        simulator = AcceptanceSimulator(
            control_acceptance_rate=0.25,
            treatment_acceptance_rate=0.40,
            output_dir=tmp_path / "simulated",
            seed=42,
        )

        results = simulator.simulate(count=50)

        # Verify corpus grew by accepted count
        final_size = len(rag_manager.memory_corpus)
        total_accepted = results.control_accepted + results.treatment_accepted

        # Account for potential duplicates or filtering
        assert final_size >= initial_size + (total_accepted * 0.8)  # At least 80% added

    def test_simulation_metrics_tracking(self, tmp_path):
        """Test that simulation properly tracks all metrics."""
        from codedocsync.suggestions.metrics import get_metrics_collector

        collector = get_metrics_collector()
        initial_count = len(collector.current_session)

        simulator = AcceptanceSimulator(
            control_acceptance_rate=0.25,
            treatment_acceptance_rate=0.40,
            output_dir=tmp_path / "simulated",
            seed=42,
        )

        results = simulator.simulate(count=20)

        # Verify metrics were tracked
        final_count = len(collector.current_session)
        assert final_count > initial_count

        # Check lifecycle events
        accepted_count = sum(
            1
            for m in collector.current_session
            if any(e["type"] == "accepted" for e in m.events)
        )
        rejected_count = sum(
            1
            for m in collector.current_session
            if any(e["type"] == "rejected" for e in m.events)
        )

        assert accepted_count == results.control_accepted + results.treatment_accepted
        assert accepted_count + rejected_count == results.total_suggestions
