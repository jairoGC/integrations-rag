"""
Evaluation Metrics for RAG Pipeline
Implements precision, recall, and latency tracking.
"""

import logging
import time
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""

    metric_name: str
    average_score: float
    per_query_scores: Dict[str, float]
    num_queries: int
    target_met: bool
    target_value: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "average_score": self.average_score,
            "per_query_scores": self.per_query_scores,
            "num_queries": self.num_queries,
            "target_met": self.target_met,
            "target_value": self.target_value,
            "details": self.details,
        }


class PrecisionEvaluator:
    """
    Evaluates retrieval precision.

    Precision@k = (relevant docs in top-k) / k
    """

    def __init__(self, k: int = 5, target_precision: float = 0.8) -> None:
        """
        Initialize precision evaluator.

        Args:
            k: Number of top results to evaluate (default: 5)
            target_precision: Target average precision (default: 0.8)
        """
        self.k = k
        self.target_precision = target_precision
        logger.info("Initialized PrecisionEvaluator with k=%d, target=%.2f", k, target_precision)

    def calculate_precision_at_k(
        self, retrieved_ids: List[str], relevant_ids: Set[str]
    ) -> float:
        """
        Calculate precision@k for a single query.

        Args:
            retrieved_ids: List of retrieved chunk IDs (in order)
            relevant_ids: Set of ground truth relevant chunk IDs

        Returns:
            Precision score (0.0 to 1.0)
        """
        if not retrieved_ids:
            return 0.0

        # Take top k results
        top_k = retrieved_ids[: self.k]

        # Count how many are relevant
        relevant_count = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)

        # Calculate precision@k: divide by k, not by actual retrieved count
        precision = relevant_count / self.k

        return precision

    def evaluate(
        self, test_set: List[Dict[str, Any]], retriever: Any
    ) -> EvaluationResult:
        """
        Evaluate precision across a test set.

        Args:
            test_set: List of test queries with ground truth
                Each item should have:
                - 'query': query string
                - 'relevant_chunks': set of relevant chunk IDs
            retriever: Retriever instance to evaluate

        Returns:
            EvaluationResult with precision scores
        """
        logger.info("Starting precision evaluation on %d queries", len(test_set))

        per_query_scores = {}

        for item in test_set:
            query = item["query"]
            relevant_ids = item["relevant_chunks"]

            # Retrieve results
            results = retriever.retrieve(query, top_k=self.k)

            # Extract chunk IDs
            retrieved_ids = [r.chunk_id for r in results]

            # Calculate precision
            precision = self.calculate_precision_at_k(retrieved_ids, relevant_ids)
            per_query_scores[query] = precision

            logger.debug(
                "Query: %s | Precision@%d: %.3f",
                query[:50],
                self.k,
                precision,
            )

        # Calculate average
        average_precision = (
            statistics.mean(per_query_scores.values())
            if per_query_scores
            else 0.0
        )

        target_met = average_precision >= self.target_precision

        logger.info(
            "Precision evaluation complete: avg=%.3f (target: %.2f, met: %s)",
            average_precision,
            self.target_precision,
            target_met,
        )

        return EvaluationResult(
            metric_name=f"Precision@{self.k}",
            average_score=average_precision,
            per_query_scores=per_query_scores,
            num_queries=len(test_set),
            target_met=target_met,
            target_value=self.target_precision,
            details={"k": self.k},
        )


class RecallEvaluator:
    """
    Evaluates retrieval recall.

    Recall@k = (relevant docs in top-k) / (total relevant docs)
    """

    def __init__(
        self, k_values: Optional[List[int]] = None, target_recall: float = 0.85
    ) -> None:
        """
        Initialize recall evaluator.

        Args:
            k_values: List of k values to evaluate (default: [5, 10, 20])
            target_recall: Target recall@10 (default: 0.85)
        """
        self.k_values = k_values or [5, 10, 20]
        self.target_recall = target_recall
        logger.info(
            "Initialized RecallEvaluator with k_values=%s, target=%.2f",
            self.k_values,
            target_recall,
        )

    def calculate_recall_at_k(
        self, retrieved_ids: List[str], relevant_ids: Set[str], k: int
    ) -> float:
        """
        Calculate recall@k for a single query.

        Args:
            retrieved_ids: List of retrieved chunk IDs (in order)
            relevant_ids: Set of ground truth relevant chunk IDs
            k: Number of top results to consider

        Returns:
            Recall score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0

        # Take top k results
        top_k = retrieved_ids[:k]

        # Count how many relevant docs were retrieved
        retrieved_relevant = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)

        # Calculate recall
        recall = retrieved_relevant / len(relevant_ids)

        return recall

    def evaluate(
        self, test_set: List[Dict[str, Any]], retriever: Any
    ) -> Dict[int, EvaluationResult]:
        """
        Evaluate recall across a test set for multiple k values.

        Args:
            test_set: List of test queries with ground truth
            retriever: Retriever instance to evaluate

        Returns:
            Dictionary mapping k values to EvaluationResults
        """
        logger.info(
            "Starting recall evaluation on %d queries for k=%s",
            len(test_set),
            self.k_values,
        )

        # Store results for each k
        results_by_k: Dict[int, Dict[str, float]] = {
            k: {} for k in self.k_values
        }

        for item in test_set:
            query = item["query"]
            relevant_ids = item["relevant_chunks"]

            # Retrieve results (use max k value)
            max_k = max(self.k_values)
            results = retriever.retrieve(query, top_k=max_k)

            # Extract chunk IDs
            retrieved_ids = [r.chunk_id for r in results]

            # Calculate recall for each k
            for k in self.k_values:
                recall = self.calculate_recall_at_k(retrieved_ids, relevant_ids, k)
                results_by_k[k][query] = recall

        # Create EvaluationResults for each k
        evaluation_results = {}

        for k in self.k_values:
            scores = results_by_k[k]
            average_recall = statistics.mean(scores.values()) if scores else 0.0

            # Check target for k=10
            target_met = (
                average_recall >= self.target_recall if k == 10 else None
            )

            logger.info("Recall@%d: avg=%.3f", k, average_recall)

            evaluation_results[k] = EvaluationResult(
                metric_name=f"Recall@{k}",
                average_score=average_recall,
                per_query_scores=scores,
                num_queries=len(test_set),
                target_met=target_met if target_met is not None else False,
                target_value=self.target_recall if k == 10 else 0.0,
                details={"k": k},
            )

        logger.info("Recall evaluation complete")

        return evaluation_results


class LatencyTracker:
    """
    Tracks query latency and calculates percentiles.
    """

    def __init__(self, target_p95: float = 3.0) -> None:
        """
        Initialize latency tracker.

        Args:
            target_p95: Target p95 latency in seconds (default: 3.0)
        """
        self.target_p95 = target_p95
        self.latencies: List[float] = []
        logger.info("Initialized LatencyTracker with target_p95=%.2fs", target_p95)

    def measure_query(self, query_fn: Any, *args: Any, **kwargs: Any) -> tuple:
        """
        Measure latency of a query function.

        Args:
            query_fn: Function to measure
            *args: Positional arguments for query_fn
            **kwargs: Keyword arguments for query_fn

        Returns:
            Tuple of (result, latency_seconds)
        """
        start_time = time.time()
        result = query_fn(*args, **kwargs)
        latency = time.time() - start_time

        self.latencies.append(latency)

        return result, latency

    def calculate_percentiles(self) -> Dict[str, float]:
        """
        Calculate latency percentiles.

        Returns:
            Dictionary with p50, p95, p99 latencies
        """
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        sorted_latencies = sorted(self.latencies)

        def percentile(data: List[float], p: float) -> float:
            """Calculate percentile."""
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[-1]
            d0 = data[f] * (c - k)
            d1 = data[c] * (k - f)
            return d0 + d1

        return {
            "p50": percentile(sorted_latencies, 0.50),
            "p95": percentile(sorted_latencies, 0.95),
            "p99": percentile(sorted_latencies, 0.99),
            "mean": statistics.mean(self.latencies),
        }

    def evaluate(self, test_set: List[Dict[str, Any]], pipeline: Any) -> EvaluationResult:
        """
        Evaluate latency across a test set.

        Args:
            test_set: List of test queries
            pipeline: RAG pipeline to evaluate

        Returns:
            EvaluationResult with latency statistics
        """
        logger.info("Starting latency evaluation on %d queries", len(test_set))

        # Reset latencies
        self.latencies = []

        per_query_latencies = {}

        for item in test_set:
            query = item["query"]

            # Measure query latency
            _, latency = self.measure_query(pipeline.query, query)
            per_query_latencies[query] = latency

            logger.debug("Query: %s | Latency: %.3fs", query[:50], latency)

        # Calculate percentiles
        percentiles = self.calculate_percentiles()

        # Check if p95 meets target
        target_met = percentiles["p95"] <= self.target_p95

        logger.info(
            "Latency evaluation complete: p50=%.3fs, p95=%.3fs, p99=%.3fs (target: %.2fs, met: %s)",
            percentiles["p50"],
            percentiles["p95"],
            percentiles["p99"],
            self.target_p95,
            target_met,
        )

        return EvaluationResult(
            metric_name="Latency",
            average_score=percentiles["mean"],
            per_query_scores=per_query_latencies,
            num_queries=len(test_set),
            target_met=target_met,
            target_value=self.target_p95,
            details=percentiles,
        )

    def reset(self) -> None:
        """Reset tracked latencies."""
        self.latencies = []


def format_evaluation_report(results: List[EvaluationResult]) -> str:
    """
    Format evaluation results into a readable report.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    for result in results:
        lines.append(f"Metric: {result.metric_name}")
        lines.append("-" * 70)
        lines.append(f"Average Score: {result.average_score:.4f}")
        lines.append(f"Target: {result.target_value:.4f}")
        lines.append(f"Target Met: {'✓ YES' if result.target_met else '✗ NO'}")
        lines.append(f"Number of Queries: {result.num_queries}")

        # Add details if available
        if result.details:
            lines.append("\nDetails:")
            for key, value in result.details.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
