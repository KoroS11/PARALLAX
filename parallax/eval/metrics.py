"""Evaluation metrics computation for PARALLAX.

Includes:
- Per-class precision, recall, F1, and macro/weighted F1
- Confusion matrix calculation
- Binary metrics (ROC-AUC, PR-AUC)
- Recall at fixed FPR (e.g., recall @ FPR <= 0.01, recall @ FPR <= 0.05)
- Expected Calibration Error (ECE) with equal-width binning
- Deterministic bootstrap confidence intervals with fixed random seed
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ClassMetrics:
    """Per-class classification metrics."""

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfusionMatrixResult:
    """Confusion matrix with associated class labels."""

    labels: list[str]
    matrix: list[list[int]]

    def to_numpy(self) -> np.ndarray:
        """Return the confusion matrix as a 2D numpy array."""
        return np.array(self.matrix, dtype=int)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationReport:
    """Multi-class classification report."""

    per_class: dict[str, ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    accuracy: float
    total_support: int
    confusion_matrix: ConfusionMatrixResult | None = None

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        return res


@dataclass(frozen=True)
class FixedFPRResult:
    """Operating point metrics at a constrained False Positive Rate."""

    target_fpr: float
    achieved_fpr: float
    recall: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapCI:
    """Deterministic bootstrap confidence interval estimate."""

    point_estimate: float
    lower: float
    upper: float
    mean: float
    std: float
    alpha: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BinaryMetrics:
    """Comprehensive binary classification metrics."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    support_positive: int
    support_negative: int
    roc_auc: float | None = None
    pr_auc: float | None = None
    ece: float | None = None
    recall_at_1pct_fpr: FixedFPRResult | None = None
    recall_at_5pct_fpr: FixedFPRResult | None = None
    confusion_matrix: ConfusionMatrixResult | None = None
    bootstrap_f1_ci: BootstrapCI | None = None
    bootstrap_recall_ci: BootstrapCI | None = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[str] | None = None,
) -> ConfusionMatrixResult:
    """Compute confusion matrix for arbitrary class labels.

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        labels: Optional explicit ordering of class labels.

    Returns:
        ConfusionMatrixResult with labels list and 2D matrix (row = true, col = pred).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    str_y_true = [str(y) for y in y_true]
    str_y_pred = [str(y) for y in y_pred]

    if labels is not None:
        sorted_labels = [str(lbl) for lbl in labels]
    else:
        unique_labels = set(str_y_true) | set(str_y_pred)
        sorted_labels = sorted(unique_labels)

    label_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}
    n_classes = len(sorted_labels)
    mat = [[0] * n_classes for _ in range(n_classes)]

    for yt, yp in zip(str_y_true, str_y_pred, strict=True):
        if yt in label_to_idx and yp in label_to_idx:
            mat[label_to_idx[yt]][label_to_idx[yp]] += 1

    return ConfusionMatrixResult(labels=sorted_labels, matrix=mat)


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[str] | None = None,
) -> ClassificationReport:
    """Compute per-class precision, recall, F1, support, macro-F1, and weighted F1.

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        labels: Optional explicit ordering of class labels.

    Returns:
        ClassificationReport containing per-class and aggregated macro/weighted metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    if len(y_true) == 0:
        return ClassificationReport(
            per_class={},
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            accuracy=0.0,
            total_support=0,
            confusion_matrix=ConfusionMatrixResult(labels=[], matrix=[]),
        )

    str_y_true = [str(y) for y in y_true]
    str_y_pred = [str(y) for y in y_pred]

    if labels is not None:
        sorted_labels = [str(lbl) for lbl in labels]
    else:
        unique_labels = set(str_y_true) | set(str_y_pred)
        sorted_labels = sorted(unique_labels)

    cm = compute_confusion_matrix(str_y_true, str_y_pred, labels=sorted_labels)
    mat = cm.matrix
    n_classes = len(sorted_labels)

    per_class: dict[str, ClassMetrics] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

    correct_predictions = 0

    for i, lbl in enumerate(sorted_labels):
        tp = mat[i][i]
        support = sum(mat[i])  # row sum
        pred_count = sum(mat[r][i] for r in range(n_classes))  # col sum

        prec = float(tp / pred_count) if pred_count > 0 else 0.0
        rec = float(tp / support) if support > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[lbl] = ClassMetrics(
            precision=prec,
            recall=rec,
            f1=f1,
            support=support,
        )

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        correct_predictions += tp

    total_support = len(str_y_true)
    accuracy = float(correct_predictions / total_support) if total_support > 0 else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    if total_support > 0:
        weighted_f1 = float(
            sum(f * s for f, s in zip(f1s, supports, strict=True)) / total_support
        )
    else:
        weighted_f1 = 0.0

    return ClassificationReport(
        per_class=per_class,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=accuracy,
        total_support=total_support,
        confusion_matrix=cm,
    )


def compute_roc_auc(
    y_true: Sequence[bool | int],
    y_score: Sequence[float],
) -> float | None:
    """Compute Receiver Operating Characteristic Area Under Curve (ROC-AUC).

    Args:
        y_true: Ground truth binary labels (True/1 or False/0).
        y_score: Continuous prediction scores / probabilities in [0.0, 1.0].

    Returns:
        ROC-AUC score as float, or None if undefined (e.g. only one class present).
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_score ({len(y_score)})"
        )
    if len(y_true) == 0:
        return None

    y_arr = np.asarray(y_true, dtype=int)
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        return None

    try:
        return float(roc_auc_score(y_arr, y_score))
    except (ValueError, TypeError):
        return None


def compute_pr_auc(
    y_true: Sequence[bool | int],
    y_score: Sequence[float],
) -> float | None:
    """Compute Precision-Recall Area Under Curve (PR-AUC).

    Args:
        y_true: Ground truth binary labels (True/1 or False/0).
        y_score: Continuous prediction scores / probabilities in [0.0, 1.0].

    Returns:
        PR-AUC score as float, or None if undefined (e.g. only one class present).
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_score ({len(y_score)})"
        )
    if len(y_true) == 0:
        return None

    y_arr = np.asarray(y_true, dtype=int)
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        return None

    try:
        prec, rec, _ = precision_recall_curve(y_arr, y_score)
        return float(auc(rec, prec))
    except (ValueError, TypeError):
        return None


def compute_recall_at_fpr(
    y_true: Sequence[bool | int],
    y_score: Sequence[float],
    max_fpr: float = 0.01,
) -> FixedFPRResult:
    """Compute maximum recall (TPR) achievable while keeping FPR <= max_fpr.

    In prompt-injection defense, maintaining a very low False Positive Rate
    (e.g., <= 1% or <= 5%) on benign user inputs is a hard operational requirement.

    Args:
        y_true: Binary ground truth (True/1 for attack, False/0 for benign).
        y_score: Continuous attack probability scores.
        max_fpr: Maximum acceptable false positive rate (e.g. 0.01 or 0.05).

    Returns:
        FixedFPRResult with achieved_fpr, recall (TPR), and score threshold.
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_score ({len(y_score)})"
        )
    if len(y_true) == 0:
        return FixedFPRResult(
            target_fpr=max_fpr, achieved_fpr=0.0, recall=0.0, threshold=1.0
        )

    y_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(y_score, dtype=float)

    pos_count = int(np.sum(y_arr == 1))
    neg_count = int(np.sum(y_arr == 0))

    if pos_count == 0 or neg_count == 0:
        return FixedFPRResult(
            target_fpr=max_fpr, achieved_fpr=0.0, recall=0.0, threshold=1.0
        )

    # Compute ROC curve thresholds
    fprs, tprs, thresholds = roc_curve(
        y_arr, score_arr, pos_label=1, drop_intermediate=False
    )

    # Find operating points where FPR <= max_fpr
    valid_indices = np.where(fprs <= max_fpr)[0]
    if len(valid_indices) == 0:
        return FixedFPRResult(
            target_fpr=max_fpr, achieved_fpr=0.0, recall=0.0, threshold=1.0
        )

    # Choose operating point maximizing TPR (recall)
    best_idx = valid_indices[0]
    best_tpr = -1.0
    lowest_fpr = 2.0

    for idx in valid_indices:
        tpr = float(tprs[idx])
        fpr = float(fprs[idx])
        if tpr > best_tpr or (tpr == best_tpr and fpr < lowest_fpr):
            best_tpr = tpr
            lowest_fpr = fpr
            best_idx = idx

    threshold_val = float(thresholds[best_idx])
    # Handle infinite threshold from roc_curve when starting at index 0
    if np.isinf(threshold_val):
        threshold_val = float(np.max(score_arr) + 1e-6)

    return FixedFPRResult(
        target_fpr=max_fpr,
        achieved_fpr=float(fprs[best_idx]),
        recall=float(tprs[best_idx]),
        threshold=threshold_val,
    )


def compute_ece(
    y_true: Sequence[bool | int],
    y_score: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width binning.

    ECE measures the difference between predicted probabilities and empirical
    accuracy across binned confidence intervals:
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Binary ground truth labels (True/1 or False/0).
        y_score: Predicted continuous probability scores in [0.0, 1.0].
        n_bins: Number of equal-width bins (default 10).

    Returns:
        Expected Calibration Error in [0.0, 1.0].
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_score ({len(y_score)})"
        )
    if len(y_true) == 0:
        return 0.0
    if n_bins <= 0:
        raise ValueError("n_bins must be greater than 0")

    y_arr = np.asarray(y_true, dtype=float)
    score_arr = np.asarray(y_score, dtype=float)
    n_samples = len(y_arr)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (score_arr >= low) & (score_arr <= high)
        else:
            in_bin = (score_arr >= low) & (score_arr < high)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_arr[in_bin])
            bin_conf = np.mean(score_arr[in_bin])
            weight = bin_count / n_samples
            ece += weight * abs(bin_acc - bin_conf)

    return float(ece)


def compute_bootstrap_ci(
    metric_fn: Callable[[list[Any], list[Any]], float],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Compute deterministic bootstrap confidence interval for any evaluation metric.

    Uses a fixed random seed (default 42) with numpy's Generator for exact reproducibility.

    Args:
        metric_fn: Callable accepting (resampled_y_true, resampled_y_pred) and returning float.
        y_true: Ground truth labels or values.
        y_pred: Predicted labels or scores.
        n_bootstraps: Number of bootstrap iterations (default 1000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for deterministic reproducibility.

    Returns:
        BootstrapCI containing point estimate, lower and upper percentiles, mean, std.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )
    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("Cannot compute bootstrap CI on empty sequences")

    point_estimate = float(metric_fn(list(y_true), list(y_pred)))

    rng = np.random.default_rng(seed)
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)

    boot_values: list[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        sample_true = true_arr[boot_idx].tolist()
        sample_pred = pred_arr[boot_idx].tolist()
        try:
            val = float(metric_fn(sample_true, sample_pred))
            if not np.isnan(val) and not np.isinf(val):
                boot_values.append(val)
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    if not boot_values:
        return BootstrapCI(
            point_estimate=point_estimate,
            lower=point_estimate,
            upper=point_estimate,
            mean=point_estimate,
            std=0.0,
            alpha=alpha,
            seed=seed,
        )

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower = float(np.percentile(boot_values, lower_pct))
    upper = float(np.percentile(boot_values, upper_pct))
    mean_val = float(np.mean(boot_values))
    std_val = float(np.std(boot_values))

    return BootstrapCI(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        mean=mean_val,
        std=std_val,
        alpha=alpha,
        seed=seed,
    )


def compute_wilson_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute exact Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of positive outcomes (e.g. true positives or false positives).
    total : int
        Total trials (e.g. total positives or total negatives).
    confidence : float, default=0.95
        Confidence level for two-sided interval.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) clamped to [0.0, 1.0].
    """
    if total <= 0:
        return (0.0, 0.0)

    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    else:
        z = 1.959963984540054

    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (
        z * np.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    ) / denom
    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return (round(lower, 4), round(upper, 4))


def compute_binary_metrics(
    y_true: Sequence[bool | int],
    y_pred: Sequence[bool | int],
    y_score: Sequence[float] | None = None,
    threshold: float = 0.5,
    compute_cis: bool = False,
    ci_bootstraps: int = 500,
    ci_seed: int = 42,
) -> BinaryMetrics:
    """Compute full suite of binary classification metrics for prompt injection.

    Args:
        y_true: Ground truth binary labels (True/1 = attack, False/0 = benign).
        y_pred: Predicted binary labels. If None and y_score provided, thresholded at threshold.
        y_score: Optional continuous attack probability scores.
        threshold: Score threshold if y_pred is derived from y_score.
        compute_cis: Whether to compute bootstrap confidence intervals for F1 and Recall.
        ci_bootstraps: Number of bootstrap iterations for CI.
        ci_seed: Fixed random seed for bootstrap.

    Returns:
        BinaryMetrics dataclass containing all standard metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    int_true = [1 if bool(yt) else 0 for yt in y_true]
    int_pred = [1 if bool(yp) else 0 for yp in y_pred]

    cm = compute_confusion_matrix(int_true, int_pred, labels=["0", "1"])
    # mat[row=true][col=pred]:
    # mat[0][0] = TN, mat[0][1] = FP
    # mat[1][0] = FN, mat[1][1] = TP
    tn = cm.matrix[0][0]
    fp = cm.matrix[0][1]
    fn = cm.matrix[1][0]
    tp = cm.matrix[1][1]

    support_pos = tp + fn
    support_neg = tn + fp
    total = len(int_true)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    roc_auc = None
    pr_auc = None
    ece = None
    recall_1pct = None
    recall_5pct = None

    if y_score is not None:
        roc_auc = compute_roc_auc(int_true, y_score)
        pr_auc = compute_pr_auc(int_true, y_score)
        ece = compute_ece(int_true, y_score)
        recall_1pct = compute_recall_at_fpr(int_true, y_score, max_fpr=0.01)
        recall_5pct = compute_recall_at_fpr(int_true, y_score, max_fpr=0.05)

    f1_ci = None
    rec_ci = None
    if compute_cis and total > 0:

        def f1_metric(yt: list[Any], yp: list[Any]) -> float:
            sub_true = [1 if bool(x) else 0 for x in yt]
            sub_pred = [1 if bool(x) else 0 for x in yp]
            sub_tp = sum(
                1 for t, p in zip(sub_true, sub_pred, strict=True) if t == 1 and p == 1
            )
            sub_fp = sum(
                1 for t, p in zip(sub_true, sub_pred, strict=True) if t == 0 and p == 1
            )
            sub_fn = sum(
                1 for t, p in zip(sub_true, sub_pred, strict=True) if t == 1 and p == 0
            )
            p = sub_tp / (sub_tp + sub_fp) if (sub_tp + sub_fp) > 0 else 0.0
            r = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        def recall_metric(yt: list[Any], yp: list[Any]) -> float:
            sub_true = [1 if bool(x) else 0 for x in yt]
            sub_pred = [1 if bool(x) else 0 for x in yp]
            sub_tp = sum(
                1 for t, p in zip(sub_true, sub_pred, strict=True) if t == 1 and p == 1
            )
            sub_fn = sum(
                1 for t, p in zip(sub_true, sub_pred, strict=True) if t == 1 and p == 0
            )
            return sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 0.0

        f1_ci = compute_bootstrap_ci(
            f1_metric, int_true, int_pred, n_bootstraps=ci_bootstraps, seed=ci_seed
        )
        rec_ci = compute_bootstrap_ci(
            recall_metric, int_true, int_pred, n_bootstraps=ci_bootstraps, seed=ci_seed
        )

    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        support_positive=support_pos,
        support_negative=support_neg,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        ece=ece,
        recall_at_1pct_fpr=recall_1pct,
        recall_at_5pct_fpr=recall_5pct,
        confusion_matrix=cm,
        bootstrap_f1_ci=f1_ci,
        bootstrap_recall_ci=rec_ci,
    )
