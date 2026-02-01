"""
Tests for Evaluation Metrics
Tests precision, recall, and latency tracking.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from src.evaluation.metrics import (
    PrecisionEvaluator,
    RecallEvaluator,
    LatencyTracker,
    EvaluationResult,
    format_evaluation_report,
)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            metric_name="Precision@5",
            average_score=0.85,
            per_query_scores={"q1": 0.8, "q2": 0.9},
            num_queries=2,
            target_met=True,
            target_value=0.8,
            details={"k": 5},
        )

        assert result.metric_name == "Precision@5"
        assert result.average_score == 0.85
        assert result.num_queries == 2
        assert result.target_met is True

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = EvaluationResult(
            metric_name="Test",
            average_score=0.9,
            per_query_scores={},
            num_queries=1,
            target_met=True,
            target_value=0.8,
        )

        result_dict = result.to_dict()

        assert result_dict["metric_name"] == "Test"
        assert result_dict["average_score"] == 0.9
        assert result_dict["target_met"] is True


class TestPrecisionEvaluator:
    """Tests for PrecisionEvaluator."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        evaluator = PrecisionEvaluator()

        assert evaluator.k == 5
        assert evaluator.target_precision == 0.8

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        evaluator = PrecisionEvaluator(k=10, target_precision=0.9)

        assert evaluator.k == 10
        assert evaluator.target_precision == 0.9

    def test_calculate_precision_perfect(self):
        """Test precision calculation with all relevant results."""
        evaluator = PrecisionEvaluator(k=5)

        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        relevant = {"chunk1", "chunk2", "chunk3", "chunk4", "chunk5"}

        precision = evaluator.calculate_precision_at_k(retrieved, relevant)

        assert precision == 1.0

    def test_calculate_precision_half(self):
        """Test precision calculation with half relevant."""
        evaluator = PrecisionEvaluator(k=4)

        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4"]
        relevant = {"chunk1", "chunk3"}

        precision = evaluator.calculate_precision_at_k(retrieved, relevant)

        assert precision == 0.5

    def test_calculate_precision_none(self):
        """Test precision when no results are relevant."""
        evaluator = PrecisionEvaluator(k=5)

        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk4", "chunk5"}

        precision = evaluator.calculate_precision_at_k(retrieved, relevant)

        assert precision == 0.0

    def test_calculate_precision_empty_retrieved(self):
        """Test precision with empty retrieved list."""
        evaluator = PrecisionEvaluator(k=5)

        precision = evaluator.calculate_precision_at_k([], {"chunk1"})

        assert precision == 0.0

    def test_calculate_precision_respects_k(self):
        """Test that only top-k results are considered."""
        evaluator = PrecisionEvaluator(k=3)

        # First 3 are relevant, next 2 are not
        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        relevant = {"chunk1", "chunk2", "chunk3"}

        precision = evaluator.calculate_precision_at_k(retrieved, relevant)

        # Should be 3/3 = 1.0, ignoring chunks 4 and 5
        assert precision == 1.0

    def test_evaluate_single_query(self):
        """Test evaluation with a single query."""
        evaluator = PrecisionEvaluator(k=5)

        # Mock retriever
        mock_retriever = MagicMock()
        mock_result = Mock()
        mock_result.chunk_id = "chunk1"
        mock_retriever.retrieve.return_value = [mock_result]

        test_set = [{"query": "test query", "relevant_chunks": {"chunk1"}}]

        result = evaluator.evaluate(test_set, mock_retriever)

        assert result.metric_name == "Precision@5"
        assert result.average_score == 0.2  # 1 relevant out of 1 retrieved, k=5
        assert result.num_queries == 1

    def test_evaluate_multiple_queries(self):
        """Test evaluation with multiple queries."""
        evaluator = PrecisionEvaluator(k=5)

        # Mock retriever that returns 3 chunks, 2 relevant
        mock_retriever = MagicMock()

        def mock_retrieve(query, top_k):
            results = []
            for i in range(3):
                mock_result = Mock()
                mock_result.chunk_id = f"chunk{i}"
                results.append(mock_result)
            return results

        mock_retriever.retrieve.side_effect = mock_retrieve

        test_set = [
            {"query": "query1", "relevant_chunks": {"chunk0", "chunk1"}},
            {"query": "query2", "relevant_chunks": {"chunk1", "chunk2"}},
        ]

        result = evaluator.evaluate(test_set, mock_retriever)

        # Query 1: 2/3 relevant, Query 2: 2/3 relevant
        # Average: (2/3 + 2/3) / 2 = 2/3 ≈ 0.667
        assert result.average_score == pytest.approx(0.4, abs=0.01)  # 2/5 each
        assert result.num_queries == 2

    def test_evaluate_target_met(self):
        """Test that target_met is calculated correctly."""
        evaluator = PrecisionEvaluator(k=2, target_precision=0.5)

        mock_retriever = MagicMock()
        mock_result = Mock()
        mock_result.chunk_id = "chunk1"
        mock_retriever.retrieve.return_value = [mock_result]

        test_set = [{"query": "test", "relevant_chunks": {"chunk1"}}]

        result = evaluator.evaluate(test_set, mock_retriever)

        # Precision is 1/2 = 0.5, meets target
        assert result.target_met is True


class TestRecallEvaluator:
    """Tests for RecallEvaluator."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        evaluator = RecallEvaluator()

        assert evaluator.k_values == [5, 10, 20]
        assert evaluator.target_recall == 0.85

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        evaluator = RecallEvaluator(k_values=[3, 5], target_recall=0.9)

        assert evaluator.k_values == [3, 5]
        assert evaluator.target_recall == 0.9

    def test_calculate_recall_perfect(self):
        """Test recall calculation with all relevant retrieved."""
        evaluator = RecallEvaluator()

        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk2", "chunk3"}

        recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=5)

        assert recall == 1.0

    def test_calculate_recall_half(self):
        """Test recall calculation with half relevant retrieved."""
        evaluator = RecallEvaluator()

        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4"]
        relevant = {"chunk1", "chunk2", "chunk5", "chunk6"}

        recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=10)

        assert recall == 0.5  # Retrieved 2 out of 4 relevant

    def test_calculate_recall_none(self):
        """Test recall when no relevant docs retrieved."""
        evaluator = RecallEvaluator()

        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk3", "chunk4"}

        recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=5)

        assert recall == 0.0

    def test_calculate_recall_empty_relevant(self):
        """Test recall with empty relevant set."""
        evaluator = RecallEvaluator()

        recall = evaluator.calculate_recall_at_k(["chunk1"], set(), k=5)

        assert recall == 0.0

    def test_calculate_recall_respects_k(self):
        """Test that only top-k results are considered."""
        evaluator = RecallEvaluator()

        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        relevant = {"chunk1", "chunk2", "chunk6"}  # 3 relevant total

        # At k=2, only first 2 retrieved, so 2/3 recall
        recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=2)
        assert recall == pytest.approx(2 / 3, abs=0.01)

        # At k=5, first 5 retrieved, still only 2 relevant, so 2/3 recall
        recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=5)
        assert recall == pytest.approx(2 / 3, abs=0.01)

    def test_evaluate_multiple_k(self):
        """Test evaluation with multiple k values."""
        evaluator = RecallEvaluator(k_values=[2, 5])

        # Mock retriever
        mock_retriever = MagicMock()

        def mock_retrieve(query, top_k):
            results = []
            for i in range(min(top_k, 5)):
                mock_result = Mock()
                mock_result.chunk_id = f"chunk{i}"
                results.append(mock_result)
            return results

        mock_retriever.retrieve.side_effect = mock_retrieve

        test_set = [
            {"query": "test", "relevant_chunks": {"chunk0", "chunk1", "chunk2"}}
        ]

        results = evaluator.evaluate(test_set, mock_retriever)

        assert 2 in results
        assert 5 in results
        assert results[2].metric_name == "Recall@2"
        assert results[5].metric_name == "Recall@5"


class TestLatencyTracker:
    """Tests for LatencyTracker."""

    def test_init(self):
        """Test initialization."""
        tracker = LatencyTracker(target_p95=2.0)

        assert tracker.target_p95 == 2.0
        assert tracker.latencies == []

    def test_measure_query(self):
        """Test measuring query latency."""
        tracker = LatencyTracker()

        def slow_function():
            time.sleep(0.01)
            return "result"

        result, latency = tracker.measure_query(slow_function)

        assert result == "result"
        assert latency >= 0.01
        assert len(tracker.latencies) == 1

    def test_measure_query_with_args(self):
        """Test measuring query with arguments."""
        tracker = LatencyTracker()

        def add(a, b):
            return a + b

        result, latency = tracker.measure_query(add, 2, 3)

        assert result == 5
        assert latency >= 0  # May be 0 for very fast operations

    def test_calculate_percentiles_empty(self):
        """Test percentile calculation with no data."""
        tracker = LatencyTracker()

        percentiles = tracker.calculate_percentiles()

        assert percentiles["p50"] == 0.0
        assert percentiles["p95"] == 0.0
        assert percentiles["p99"] == 0.0

    def test_calculate_percentiles(self):
        """Test percentile calculation with data."""
        tracker = LatencyTracker()
        tracker.latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        percentiles = tracker.calculate_percentiles()

        assert percentiles["p50"] == pytest.approx(0.5, abs=0.1)
        assert percentiles["p95"] == pytest.approx(0.95, abs=0.1)
        assert percentiles["p99"] == pytest.approx(0.99, abs=0.1)
        assert percentiles["mean"] == 0.55

    def test_evaluate(self):
        """Test latency evaluation."""
        tracker = LatencyTracker(target_p95=1.0)

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {"answer": "test"}

        test_set = [
            {"query": "test1"},
            {"query": "test2"},
        ]

        result = tracker.evaluate(test_set, mock_pipeline)

        assert result.metric_name == "Latency"
        assert result.num_queries == 2
        assert len(result.per_query_scores) == 2

    def test_reset(self):
        """Test resetting latencies."""
        tracker = LatencyTracker()
        tracker.latencies = [1.0, 2.0, 3.0]

        tracker.reset()

        assert tracker.latencies == []


class TestFormatEvaluationReport:
    """Tests for format_evaluation_report function."""

    def test_format_single_result(self):
        """Test formatting a single result."""
        result = EvaluationResult(
            metric_name="Precision@5",
            average_score=0.85,
            per_query_scores={},
            num_queries=10,
            target_met=True,
            target_value=0.8,
            details={"k": 5},
        )

        report = format_evaluation_report([result])

        assert "EVALUATION REPORT" in report
        assert "Precision@5" in report
        assert "0.8500" in report
        assert "✓ YES" in report

    def test_format_multiple_results(self):
        """Test formatting multiple results."""
        results = [
            EvaluationResult(
                metric_name="Precision@5",
                average_score=0.85,
                per_query_scores={},
                num_queries=10,
                target_met=True,
                target_value=0.8,
            ),
            EvaluationResult(
                metric_name="Recall@10",
                average_score=0.75,
                per_query_scores={},
                num_queries=10,
                target_met=False,
                target_value=0.85,
            ),
        ]

        report = format_evaluation_report(results)

        assert "Precision@5" in report
        assert "Recall@10" in report
        assert "✓ YES" in report
        assert "✗ NO" in report


def test_integration():
    """Test that evaluation components work together."""
    # Create evaluators
    precision_eval = PrecisionEvaluator(k=5)
    recall_eval = RecallEvaluator(k_values=[5])
    latency_tracker = LatencyTracker()

    # Verify they have required methods
    assert hasattr(precision_eval, "evaluate")
    assert hasattr(recall_eval, "evaluate")
    assert hasattr(latency_tracker, "evaluate")
