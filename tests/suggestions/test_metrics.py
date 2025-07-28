"""Tests for suggestion metrics tracking and A/B testing."""

import time
import uuid

from codedocsync.suggestions.metrics import (
    ABTestController,
    ImprovementCalculator,
    MetricsCollector,
    SuggestionMetrics,
)


class TestSuggestionMetrics:
    """Test SuggestionMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating suggestion metrics."""
        metric = SuggestionMetrics(
            suggestion_id=str(uuid.uuid4()),
            timestamp=time.time(),
            function_signature="def test_func(x: int) -> str",
            file_path="test.py",
            issue_type="missing_docstring",
            generator_used="BehaviorSuggestionGenerator",
            rag_enhanced=True,
            examples_used=3,
            similarity_scores=[0.8, 0.7, 0.6],
            suggestion_length=150,
            completeness_score=0.85,
            confidence_score=0.9,
            ab_group="treatment",
        )

        assert metric.shown_to_user is False
        assert metric.accepted is False
        assert len(metric.events) == 0

    def test_add_event(self):
        """Test adding lifecycle events."""
        metric = SuggestionMetrics(
            suggestion_id=str(uuid.uuid4()),
            timestamp=time.time(),
            function_signature="def test()",
            file_path="test.py",
            issue_type="test",
            generator_used="test",
            rag_enhanced=False,
            examples_used=0,
            suggestion_length=100,
            completeness_score=0.5,
            confidence_score=0.7,
            ab_group="control",
        )

        # Add display event
        metric.add_event("displayed")
        assert metric.shown_to_user is True
        assert len(metric.events) == 1
        assert metric.events[0]["type"] == "displayed"

        # Add acceptance event
        time.sleep(0.1)  # Ensure time difference
        metric.add_event("accepted")
        assert metric.accepted is True
        assert metric.time_to_decision is not None
        assert metric.time_to_decision > 0


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def test_track_suggestion(self, tmp_path):
        """Test tracking a new suggestion."""
        collector = MetricsCollector(tmp_path)

        suggestion_id = collector.track_suggestion(
            function_signature="def process_data(items: list[str]) -> dict",
            file_path="processor.py",
            issue_type="missing_returns",
            generator_name="ReturnSuggestionGenerator",
            suggestion_text="Process items and return results dict.",
            confidence=0.85,
            rag_examples=[{"similarity": 0.9}, {"similarity": 0.8}],
        )

        assert suggestion_id is not None
        assert len(collector.current_session) == 1

        metric = collector.current_session[0]
        assert metric.suggestion_id == suggestion_id
        assert metric.rag_enhanced is True
        assert metric.examples_used == 2
        assert metric.ab_group == "treatment"

    def test_mark_displayed(self, tmp_path):
        """Test marking suggestion as displayed."""
        collector = MetricsCollector(tmp_path)

        # Track suggestion
        suggestion_id = collector.track_suggestion(
            function_signature="def test()",
            file_path="test.py",
            issue_type="test",
            generator_name="TestGenerator",
            suggestion_text="Test",
            confidence=0.5,
        )

        # Mark as displayed
        collector.mark_displayed(suggestion_id)

        metric = collector.current_session[0]
        assert metric.shown_to_user is True
        assert any(e["type"] == "displayed" for e in metric.events)

    def test_save_and_load_metrics(self, tmp_path):
        """Test saving and loading metrics."""
        collector1 = MetricsCollector(tmp_path)

        # Track some suggestions
        id1 = collector1.track_suggestion(
            function_signature="def func1()",
            file_path="test1.py",
            issue_type="test",
            generator_name="Gen1",
            suggestion_text="Text1",
            confidence=0.8,
        )
        collector1.mark_displayed(id1)
        collector1.mark_accepted(id1)

        # Save metrics
        collector1.save_session_metrics()

        # Create new collector and verify it loads
        collector2 = MetricsCollector(tmp_path)
        assert len(collector2.historical_metrics) == 1
        assert collector2.historical_metrics[0].suggestion_id == id1
        assert collector2.historical_metrics[0].accepted is True


class TestABTestController:
    """Test A/B testing controller."""

    def test_deterministic_assignment(self):
        """Test that assignments are deterministic."""
        controller1 = ABTestController(treatment_percentage=0.5)
        controller2 = ABTestController(treatment_percentage=0.5)

        # Same function should get same assignment
        func_sig = "def test_function(x: int) -> str"
        assert controller1.get_assignment(func_sig) == controller2.get_assignment(
            func_sig
        )

    def test_assignment_distribution(self):
        """Test that assignments follow expected distribution."""
        controller = ABTestController(treatment_percentage=0.5)

        # Generate many assignments
        assignments = []
        for i in range(1000):
            sig = f"def func_{i}()"
            assignments.append(controller.get_assignment(sig))

        # Check distribution (should be close to 50/50)
        treatment_count = assignments.count("treatment")
        assert 400 < treatment_count < 600  # Allow some variance


class TestImprovementCalculator:
    """Test improvement metrics calculation."""

    def test_calculate_improvement(self):
        """Test calculating improvement percentage."""
        # Create test metrics
        metrics = []

        # Control group: 2/5 accepted (40%)
        for i in range(5):
            m = SuggestionMetrics(
                suggestion_id=str(uuid.uuid4()),
                timestamp=time.time(),
                function_signature=f"def control_{i}()",
                file_path="test.py",
                issue_type="test",
                generator_used="TestGen",
                rag_enhanced=False,
                examples_used=0,
                suggestion_length=100,
                completeness_score=0.5,
                confidence_score=0.7,
                ab_group="control",
            )
            m.shown_to_user = True
            m.accepted = i < 2
            metrics.append(m)

        # Treatment group: 4/5 accepted (80%)
        for i in range(5):
            m = SuggestionMetrics(
                suggestion_id=str(uuid.uuid4()),
                timestamp=time.time(),
                function_signature=f"def treatment_{i}()",
                file_path="test.py",
                issue_type="test",
                generator_used="TestGen",
                rag_enhanced=True,
                examples_used=3,
                suggestion_length=150,
                completeness_score=0.8,
                confidence_score=0.9,
                ab_group="treatment",
            )
            m.shown_to_user = True
            m.accepted = i < 4
            metrics.append(m)

        calculator = ImprovementCalculator(metrics)

        # Check acceptance rates
        rates = calculator.calculate_acceptance_rates()
        assert rates["control"] == 0.4
        assert rates["treatment"] == 0.8

        # Check improvement: (0.8 - 0.4) / 0.4 * 100 = 100%
        improvement = calculator.calculate_improvement_percentage()
        assert improvement == 100.0

    def test_generate_report(self):
        """Test report generation."""
        # Create minimal test data
        metrics = [
            SuggestionMetrics(
                suggestion_id=str(uuid.uuid4()),
                timestamp=time.time(),
                function_signature="def test()",
                file_path="test.py",
                issue_type="test",
                generator_used="TestGen",
                rag_enhanced=False,
                examples_used=0,
                suggestion_length=100,
                completeness_score=0.5,
                confidence_score=0.7,
                ab_group="control",
                shown_to_user=True,
                accepted=True,
            )
        ]

        calculator = ImprovementCalculator(metrics)
        report = calculator.generate_report()

        assert "RAG Enhancement Improvement Report" in report
        assert "Total suggestions analyzed: 1" in report
