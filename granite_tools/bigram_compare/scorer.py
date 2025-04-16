from __future__ import annotations

import pickle
import random
import typing
from collections import Counter
from typing import TypedDict

import numpy as np
from scipy.stats import norm  # type: ignore

from granite_tools.app_types import KeySeq
from granite_tools.bigram_compare.comparisons import keyseqs_to_comparisons
from granite_tools.bigram_compare.fitting import get_scores

if typing.TYPE_CHECKING:
    from granite_tools.bigram_compare.compare_app import KeySequenceCompareApp


class CompareDataDict(TypedDict):

    comparisons_all: list[tuple[KeySeq, KeySeq]]
    initial_order: list[KeySeq]
    current_order: list[KeySeq]
    processed_key_sequences: list[KeySeq]
    current_scores: dict[KeySeq, float]
    p95_fraction: float
    pairs_per_sequence: int


def load_compare_pickle(file_path: str) -> CompareDataDict:
    """Loads data from .compare.pickle file"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return typing.cast(CompareDataDict, data)


def save_compare_pickle(file_path: str, d: CompareDataDict) -> None:
    """Loads data into a .compare.pickle file"""
    with open(file_path, "wb") as f:
        pickle.dump(d, f)


def get_distribution_sigma(n: int, p95_fraction=1 / 5) -> float:
    """Gets the normal distribution sigma (STD), which is chosen so that the normal
    distribution central 95% inter-percentile would have a width of n*p95_fraction.
    In other words, if n=10 and p95_fraction=0.2, and mean would be set to 5 there would
    be a 95% chance that the a randomly selected value would be within the range from 4
    to 6 (width of 2 which is n*p95_fraction).
    """
    width = n * p95_fraction
    sigma = width / (2 * norm.ppf(0.975))
    return float(sigma)


def unique_numbers_from_normal_dist(
    num_samples, current_index: int, lower_bound: int, upper_bound: int, sigma: float
) -> list[int]:
    """Generates a list of unique integers from a normal distribution centered around
    current_index. The output may contain lower_bound and upper_bound, but not anything
    outside them or the current_index. The sigma is used as the standard deviation in the
    normal distribution."""

    if upper_bound - lower_bound < num_samples:
        raise ValueError(
            "Range is too small to generate the required number of unique samples."
        )
    if current_index < lower_bound or current_index > upper_bound:
        raise ValueError("Current index is out of bounds.")

    mean = current_index

    counter: Counter[int] = Counter()

    while len(counter) < num_samples:
        # Generate more samples than needed to improve efficiency
        samples = np.random.normal(loc=mean, scale=sigma, size=(num_samples * 10))

        rounded_samples = [
            int(round(x)) for x in samples if lower_bound <= round(x) <= upper_bound
        ]
        counter.update(rounded_samples)

        if current_index in counter:
            counter.pop(current_index)

        if len(counter) >= num_samples:
            break

    return [x[0] for x in counter.most_common(num_samples)]


class RoundFinishedError(RuntimeError):
    """The comparison round is already finished."""


class RoundNotFinishedError(RuntimeError): ...


class NoCurrentComparisonPair(RuntimeError): ...


class ComparisonBasedScorer:
    """A class for the granite-bigram-compare app."""

    comparisons_all: list[tuple[KeySeq, KeySeq]]
    """The most important piece of information. The first N-1 pairs are the pairs created
    based on the initial_order (lenght N). The rest of the pairs are added in batches
    in the end of each round during fitting. Each item in the list is a tuple of two
    key sequences. The left side of the tuple is the one with higher score (more effort).

    This is used at the end of each round to calculate scores for the key sequences,
    which is then used to create the key sequence order / ranking.
    """

    comparisons_new: list[tuple[KeySeq, KeySeq]]
    """The comparisons that are being made with the current key sequence. These have not
    yet been fitted. These will be added to `comparisons_all` when the round is
    finished."""

    pair_index: int
    """The index of the current comparison pair (in `current_comparison_pairs`)."""

    current_scores: dict[KeySeq, float]
    """The current scores of the key sequences.."""

    sigma: float
    """The standard deviation of the normal distribution used to select the key
    sequences for comparison."""

    def __init__(
        self,
        initial_order: list[KeySeq],
        current_order: list[KeySeq] | None = None,
        p95_fraction: float = 1 / 4,
        pairs_per_sequence: int = 10,
        processed_key_sequences: list[KeySeq] | None = None,
        app: KeySequenceCompareApp | None = None,
    ) -> None:
        """
        Parameters
        ---------
        initial_order : list[KeySeq]
            The initial order for key sequences.
        current_order : list[KeySeq], optional
            The current order. If not given, the initial order is used. This should be
            used when loading a saved state from a file.
        p95_fraction: float, optional
            Controls how close or far away from the "current key sequence" the random
            pairs for comparison are selected. The 1/4 means that 95% of the pairs will
            be chosen within an range corresponding 1/4 of all the key sequences (1/8
            to each direction)
        pairs_per_sequence : int, optional
            The number of pairs to compare with each key sequence.
        processed_key_sequences : list[KeySeq], optional
            The key sequences that have already been processed.
        """
        self.current_order = (
            current_order.copy() if current_order is not None else initial_order.copy()
        )
        self.initial_order = initial_order.copy()
        self.n_ngrams = len(self.initial_order)
        self.processed_key_sequences: list[KeySeq] = (
            processed_key_sequences.copy()
            if processed_key_sequences is not None
            else []
        )
        self.current_key_sequence: KeySeq | None = None
        self.current_scores: dict[KeySeq, float] = dict()
        self.pairs_per_sequence = pairs_per_sequence
        self.p95_fraction = p95_fraction
        self.app = app
        self.sigma = get_distribution_sigma(
            n=len(self.initial_order),
            p95_fraction=self.p95_fraction,
        )
        self.comparisons_all = keyseqs_to_comparisons(self.initial_order)
        self.pair_index = 0
        self.comparisons_new: list[tuple[KeySeq, KeySeq]] = []
        if not self.is_finished():
            self.current_comparison_pairs = self._select_new_comparison_pairs()
        else:
            self.current_comparison_pairs = []

    def _select_new_comparison_pairs(self) -> list[tuple[KeySeq, KeySeq]]:
        """Selects random key sequence pairs for comparison.

        Rules:
        1) The left side of each pair is the same in each pair
        2) There will be `pairs_per_sequence` pairs
        3) The right side of each pair is selected using a normal distribution.
           The right side cannot be the same as the left side. In addition, if the left
           side is a unigram or a repeat, the right side can only be a (non-repeat)
           bigram.

        Returns
        -------
        list[tuple[KeySeq, KeySeq]]
            A list of key sequence pairs for comparison. All the the key sequences
            are "new" (not seen before).
        """
        available_key_sequences = self._get_available_key_sequences()
        self.current_key_sequence = self._get_next_key_sequence(available_key_sequences)
        pair_candidates = filter_candidates(
            self.current_order, self.current_key_sequence
        )

        current_idx = pair_candidates.index(self.current_key_sequence)

        indices = unique_numbers_from_normal_dist(
            num_samples=self.pairs_per_sequence,
            current_index=current_idx,
            lower_bound=0,
            upper_bound=len(pair_candidates) - 1,
            sigma=self.sigma,
        )

        pairs = []
        for idx in indices:
            pairs.append((self.current_key_sequence, pair_candidates[idx]))

        return pairs

    def _get_available_key_sequences(self) -> list[KeySeq]:
        return list(set(self.initial_order) - set(self.processed_key_sequences))

    @staticmethod
    def _get_next_key_sequence(available_key_sequences: list[KeySeq]) -> KeySeq:
        if not available_key_sequences:
            raise ValueError("No more key sequences to compare.")
        return random.choice(available_key_sequences)

    def handle_select_left(self):
        if self.is_current_round_finished():
            raise RoundFinishedError("The current round is already finished.")

        self.comparisons_new.append(
            tuple(reversed(self.current_comparison_pairs[self.pair_index]))
        )
        self.pair_index += 1
        self.refresh()

    def handle_select_right(self):
        if self.is_current_round_finished():
            raise RoundFinishedError("The current round is already finished.")

        self.comparisons_new.append(self.current_comparison_pairs[self.pair_index])
        self.pair_index += 1
        self.refresh()

    def handle_goto_previous(self):
        if not self.comparisons_new:
            return
        self.comparisons_new.pop()
        self.pair_index -= 1
        self.refresh()

    def fit(self, allow_unfinished: bool = False):
        if not allow_unfinished and not self.is_current_round_finished():
            raise RoundNotFinishedError("The current round is not yet finished.")
        self.comparisons_all += self.comparisons_new
        self.comparisons_new = []
        self.current_scores = get_scores(self.comparisons_all)
        self.current_order = sorted(
            self.initial_order, key=lambda x: self.current_scores[x]
        )
        if self.current_key_sequence is None:
            raise RuntimeError(
                "current_key_sequence is None after fit(). This should never happen."
            )
        self.processed_key_sequences.append(self.current_key_sequence)

    def select_next_key_sequence(self):
        if not self.is_current_round_finished():
            raise RoundNotFinishedError("The current round is not yet finished.")
        self.current_comparison_pairs = self._select_new_comparison_pairs()
        self.pair_index = 0
        self.refresh()

    @property
    def current_comparison_pair(self):
        try:
            return self.current_comparison_pairs[self.pair_index]
        except IndexError:
            raise NoCurrentComparisonPair(
                "No current comparison pair. The round is finished."
            )

    def is_current_round_finished(self) -> bool:
        return self.pair_index == len(self.current_comparison_pairs)

    def is_finished(self) -> bool:
        return len(self.processed_key_sequences) >= len(self.initial_order)

    def to_dict(self) -> CompareDataDict:
        return {
            "processed_key_sequences": self.processed_key_sequences.copy(),
            "comparisons_all": self.comparisons_all.copy(),
            "initial_order": self.initial_order.copy(),
            "current_order": self.current_order.copy(),
            "current_scores": self.current_scores.copy(),
            # Settings type of things:
            "p95_fraction": self.p95_fraction,
            "pairs_per_sequence": self.pairs_per_sequence,
        }

    def save_to_file(self, file_path: str):
        d = self.to_dict()
        if not self._verify_data(d):
            return False
        save_compare_pickle(file_path, d)
        return True

    @classmethod
    def load_from_file(
        cls, file_path: str, app: KeySequenceCompareApp | None = None
    ) -> ComparisonBasedScorer:

        data = load_compare_pickle(file_path)
        instance = cls(
            data["initial_order"],
            current_order=data["current_order"],
            p95_fraction=data["p95_fraction"],
            pairs_per_sequence=data["pairs_per_sequence"],
            processed_key_sequences=data["processed_key_sequences"],
            app=app,
        )

        instance.comparisons_all = data["comparisons_all"]
        instance.current_scores = data["current_scores"]
        instance._verify_data(data)
        return instance

    def refresh(self):
        if self.app is None:
            return

        try:
            self.app.do_refresh()
        except NoCurrentComparisonPair:
            pass

        self.app.set_progress(len(self.processed_key_sequences))

    def _verify_data(self, data) -> bool:
        """Verifies the data.

        Returns
        -------
        True if the data is valid, otherwise False.
        """
        try:
            verify_data(data)
            return True
        except Exception as exc:
            msg = f"Error in data validation! Please check the data before proceeding! Error message: {str(exc)}"
            if self.app is not None:
                self.app.write_log(msg)
            else:
                print(msg)
            return False


class DataValidityError(RuntimeError): ...


def verify_data(data: CompareDataDict):
    initial_pairs = len(data["initial_order"]) - 1
    p = data["pairs_per_sequence"]
    n_sequences = len(data["processed_key_sequences"])
    expected_comparisons = initial_pairs + p * n_sequences
    n_comparisons = len(data["comparisons_all"])
    if n_comparisons != expected_comparisons:
        raise DataValidityError(
            f"Number of comparisons ({n_comparisons}) does not match the expected number of comparisons ({expected_comparisons}). Debug info: {initial_pairs=}, pairs_per_sequence={p}, n_processed_sequences={n_sequences}"
        )

    expected_first = keyseqs_to_comparisons(data["initial_order"])
    if data["comparisons_all"][:initial_pairs] != expected_first:
        raise DataValidityError(
            "The first part of the comparisons_all list does not match the pairs created from initial_order."
        )

    for i, seq in enumerate(data["processed_key_sequences"]):
        pairs_for_this_keyseq = set()
        for j in range(p):
            idx_offset = i * p + j
            idx = initial_pairs + idx_offset
            pair = data["comparisons_all"][idx]
            if seq not in pair:
                raise DataValidityError(
                    f"Key sequence {seq} is not in the comparison pair {pair} at index {idx} of comparisons_all!"
                )
            other = pair[0] if pair[1] == seq else pair[1]
            if other in pairs_for_this_keyseq:
                raise DataValidityError(
                    f"Key sequence {seq} is paired with {other} more than once!"
                )
            pairs_for_this_keyseq.add(other)

    counts = Counter(data["processed_key_sequences"])
    most_common = counts.most_common(1)[0]
    if most_common[1] > 1:
        raise DataValidityError(
            f"Duplicate key sequence {most_common[0]} in processed_key_sequences!"
        )

    set_initial = set(data["initial_order"])
    set_current = set(data["current_order"])
    set_processed = set(data["processed_key_sequences"])
    if len(data["processed_key_sequences"]) == len(data["initial_order"]):
        if set_initial != set_processed:
            missing = set_initial - set_processed
            extra = set_processed - set_initial
            raise DataValidityError(
                f"Key sequences in processed_key_sequences do not match the ones in initial_order!: {missing=} {extra=}"
            )

    if set_initial != set_current:
        raise DataValidityError(
            "The initial_order does not contain the same key sequences as the current_order!"
        )

    # All processed key sequences must be in the initial order and current order
    for pair in data["comparisons_all"]:
        for keyseq in pair:
            if keyseq not in set_initial:
                raise DataValidityError(
                    f"comparisons_all contains pair {pair} which has key sequence {keyseq} that is not part of initial_order"
                )
            if keyseq not in set_current:
                raise DataValidityError(
                    f"comparisons_all contains pair {pair} which has key sequence {keyseq} that is not part of current_order"
                )

    # All processed pairs must be in the initial order and in the current order
    for keyseq in data["processed_key_sequences"]:
        if keyseq not in set_initial:
            raise DataValidityError(
                f"processed_key_sequences contains key sequence {keyseq} that is not part of initial_order"
            )
        if keyseq not in set_current:
            raise DataValidityError(
                f"processed_key_sequences contains key sequence {keyseq} that is not part of current_order"
            )


def filter_candidates(
    all_key_sequences: list[KeySeq], current_key_sequence: KeySeq
) -> list[KeySeq]:
    """Filters pairing logic a bit: Bigrams can be paired with any key sequence, but
    any other type of key sequence can only be paired with bigrams."""
    if is_unigram(current_key_sequence):
        return [
            seq
            for seq in all_key_sequences
            if is_bigram(seq) or seq == current_key_sequence
        ]
    elif is_repeat(current_key_sequence):
        return [
            seq
            for seq in all_key_sequences
            if is_bigram(seq) or seq == current_key_sequence
        ]
    return all_key_sequences


def is_unigram(key_sequence: KeySeq):
    return len(key_sequence) == 1


def is_repeat(key_sequence: KeySeq):
    return len(key_sequence) >= 2 and len(set(key_sequence)) == 1


def is_bigram(key_sequence: KeySeq):
    """Checks if keysequence is a non-repeating bigram"""
    if len(key_sequence) != 2:
        return False
    return key_sequence[0] != key_sequence[1]
