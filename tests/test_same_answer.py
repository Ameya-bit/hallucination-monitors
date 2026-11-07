"""Tests for Dr. Miller's same-answer test on synthetic data where the answer is known."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pilot"))
from same_answer import SameAnswerGroup, centered_cosine_matrix, permutation_test, same_answer_groups  # noqa: E402

DIM = 64


def test_groups_pair_each_hallucination_with_correct_rows_giving_the_same_answer():
    answers = np.array(["france", "france", "france", "italy", "spain", "france"])
    is_hallucination = np.array([True, False, False, True, True, False])
    is_correct = np.array([False, True, True, False, False, False])  # row 5 is wrong but not in the hard set
    groups = same_answer_groups(answers, is_hallucination, is_correct, min_correct=1)
    assert [(g.hallucination, g.correct) for g in groups] == [(0, (1, 2))]


def test_groups_with_too_few_correct_rows_are_left_out():
    answers = np.array(["red", "red", "blue", "blue", "blue"])
    is_hallucination = np.array([True, False, True, False, False])
    is_correct = ~is_hallucination
    groups = same_answer_groups(answers, is_hallucination, is_correct, min_correct=2)
    assert [g.hallucination for g in groups] == [2]


def test_centered_cosine_matrix_removes_the_shared_offset():
    # Two vectors that share a big common offset look alike raw, but are orthogonal once it is removed.
    offset = np.full(DIM, 10.0)
    a, b = offset.copy(), offset.copy()
    a[0] += 1.0
    b[1] += 1.0
    cos = centered_cosine_matrix(np.stack([a, b]), center=offset)
    assert cos[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert np.diag(cos) == pytest.approx([1.0, 1.0])


def synthetic_groups(n_groups, group_size, rng, hallucination_shares_direction):
    """Each group's correct rows share a direction; the hallucination does or does not."""
    rows, groups = [], []
    for _ in range(n_groups):
        direction = rng.normal(size=DIM)
        start = len(rows)
        h = direction if hallucination_shares_direction else rng.normal(size=DIM)  # same scale, unrelated direction
        rows.append(h + rng.normal(scale=0.5, size=DIM))
        rows.extend(direction + rng.normal(scale=0.5, size=DIM) for _ in range(group_size))
        groups.append((start, tuple(range(start + 1, start + 1 + group_size))))
    return np.array(rows), groups


def to_groups(pairs):
    return [SameAnswerGroup(hallucination=h, correct=c) for h, c in pairs]


def test_permutation_test_detects_a_hallucination_off_its_groups_direction():
    rng = np.random.default_rng(0)
    x, pairs = synthetic_groups(20, 3, rng, hallucination_shares_direction=False)
    result = permutation_test(centered_cosine_matrix(x, center=np.zeros(DIM)), to_groups(pairs), n_perm=2000, seed=0)
    assert result.hallucination_cos < result.correct_cos
    assert result.p_value < 0.01


def test_permutation_test_does_not_flag_a_hallucination_like_its_group():
    rng = np.random.default_rng(1)
    x, pairs = synthetic_groups(20, 3, rng, hallucination_shares_direction=True)
    result = permutation_test(centered_cosine_matrix(x, center=np.zeros(DIM)), to_groups(pairs), n_perm=2000, seed=0)
    assert result.p_value > 0.05


def test_permutation_test_needs_groups_with_at_least_two_correct_rows():
    # With one correct row, swapping labels leaves cos(h, c) unchanged, so the group carries no information.
    cos = centered_cosine_matrix(np.random.default_rng(2).normal(size=(4, DIM)), center=np.zeros(DIM))
    with pytest.raises(ValueError, match="at least two"):
        permutation_test(cos, to_groups([(0, (1,)), (2, (3,))]), n_perm=100, seed=0)
