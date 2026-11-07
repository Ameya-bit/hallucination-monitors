"""Dr. Miller's same-answer test (strong form of applying CSO to hallucinations).

Take a hallucinated prompt p_H with response R_H, and correct prompts p_1..p_L whose (correct)
response is the same R_H. Miller's claim: at some layer, g(p_H) is orthogonal to the g(p_i).

Raw activations all share a large common offset, so raw cosines are high for any pair. Cosines are
therefore taken after subtracting the mean correct-answer activation. The test asks whether the
hallucination is *less* aligned with its same-answer correct prompts than those prompts are with
each other. Under the null, the hallucination is just another member of its group, so which member
carries the "hallucination" label is exchangeable; the permutation test reassigns that label at random
within each group. A group needs at least two correct prompts, or every reassignment gives the same
cosine.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SameAnswerGroup:
    hallucination: int         # row index of the hallucinated answer
    correct: tuple[int, ...]   # row indices of correct answers with the same (normalized) response


@dataclass(frozen=True)
class PermutationResult:
    hallucination_cos: float   # mean over groups of mean cos(hallucination, its correct prompts)
    correct_cos: float         # mean over groups of mean cos(correct prompt, the group's other correct prompts)
    n_groups: int
    p_value: float             # one-sided: hallucination less aligned than a random group member would be


def same_answer_groups(answers: np.ndarray, is_hallucination: np.ndarray, is_correct: np.ndarray,
                       min_correct: int) -> list[SameAnswerGroup]:
    """One group per hallucination with at least `min_correct` correct rows giving the same answer."""
    correct_by_answer: dict[str, list[int]] = {}
    for i in np.flatnonzero(is_correct):
        correct_by_answer.setdefault(answers[i], []).append(int(i))
    return [SameAnswerGroup(hallucination=int(i), correct=tuple(correct_by_answer[answers[i]]))
            for i in np.flatnonzero(is_hallucination)
            if len(correct_by_answer.get(answers[i], ())) >= min_correct]


def centered_cosine_matrix(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) - center
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("a row equals the center, so its cosine is undefined")
    u = z / norms
    return u @ u.T


def _mean_cos_to_rest(cos: np.ndarray, members: np.ndarray) -> np.ndarray:
    """For each member, its mean cosine to the other members."""
    block = cos[np.ix_(members, members)]
    return (block.sum(1) - np.diag(block)) / (len(members) - 1)


def permutation_test(cos: np.ndarray, groups: list[SameAnswerGroup], n_perm: int, seed: int) -> PermutationResult:
    groups = [g for g in groups if len(g.correct) >= 2]
    if not groups:
        raise ValueError("need at least two correct prompts in some group")
    # per group: member 0 is the hallucination, the rest are its correct prompts
    per_member = [_mean_cos_to_rest(cos, np.array((g.hallucination, *g.correct))) for g in groups]
    observed = float(np.mean([m[0] for m in per_member]))
    correct_cos = float(np.mean([_mean_cos_to_rest(cos, np.array(g.correct)).mean() for g in groups]))
    rng = np.random.default_rng(seed)
    null = np.mean([m[rng.integers(0, len(m), n_perm)] for m in per_member], axis=0)
    p_value = (1 + int((null <= observed).sum())) / (n_perm + 1)
    return PermutationResult(hallucination_cos=observed, correct_cos=correct_cos, n_groups=len(groups), p_value=p_value)
