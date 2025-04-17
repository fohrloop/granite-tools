"""The Key Sequence Comparer App; granite-bigram-compare

Used to compare bigrams to create a new order based on the user's
preferences. The user is presented with two bigrams and selects the one that they think
is easier or "nicer" to type. The program then uses this data to create a new order.

Development
------------
Run in one terminal: (this shows logs and prints)
    uv run textual console
Run in another terminal: (this runs the app)
    uv run textual run --dev granite_tools/bigram_compare/compare_app.py foo examples/config_numbers_mini.yml
"""

from __future__ import annotations

import datetime as dt
import sys
import typing
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TypedDict

from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Container, Horizontal, Vertical
from textual.widgets import Footer, Label, Log

from granite_tools.bigram_compare.scorer import ComparisonBasedScorer
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.config import Config, read_config
from granite_tools.hands import get_hands_data
from granite_tools.textual_widgets.exit_modal import ExitModal
from granite_tools.textual_widgets.progress import Progress

if typing.TYPE_CHECKING:

    from typing import Any

    from granite_tools.hands import Hands

    KeySeq = tuple[int, ...]

STARTING_INSTRUCTIONS = """INSTRUCTIONS:

In this program, you will compare each ngram 10 times with other random ngrams.
The pairs are randomly selected from a normal distribution centered to the current
ngram. After each round of 10 comparisons, the program will fit the ngram comparison
data to form a new order. The new order will be used to select the next 10 pairs of
ngrams for comparison.

Your task is to always select the ngram which you think is easier to type (which you
want to type more often).

You cannot go back to previous rounds (only within a single round). Remember to save
your progress often."""


@dataclass
class NgramShowParams:
    left: KeySeq | None
    right: KeySeq | None
    hands: Hands


class GetSymbolTextKwargs(TypedDict):
    key_seq: KeySeq | None
    fallback: str
    center: int


class NgramCard(Vertical):

    def __init__(self, identifier: str, hands: Hands, keyseq: KeySeq | None) -> None:
        super().__init__()
        self.identifier = identifier
        self.hands = hands
        self.keyseq = keyseq
        self.text = Label(self.to_content(self.keyseq), classes="ngram-card")

    def compose(self) -> ComposeResult:
        yield self.text

    def update(self, keyseq: KeySeq | None) -> None:
        self.keyseq = keyseq
        self.text.update(self.to_content(self.keyseq))
        self.text.classes = "ngram-card hidden" if keyseq is None else "ngram-card"

    def get_style(self) -> str:
        return "#2beaff"

    def to_content(self, key_seq: KeySeq | None) -> Align:
        style = self.get_style()

        kwargs = GetSymbolTextKwargs(key_seq=key_seq, fallback="   ", center=3)
        symbols_left = self.hands.get_symbols_text("Left", **kwargs)
        symbols_right = self.hands.get_symbols_text("Right", **kwargs)
        text = symbols_left + "\n" + symbols_right
        panel = Panel(text, padding=(1, 5), border_style=style)
        return Align(
            panel,
            align="center",  # Align the panel to the center
            vertical="middle",
            pad=False,
        )


class NgramSelectionColumn(Horizontal, Center):

    def __init__(self, params: NgramShowParams) -> None:
        super().__init__()
        self.params = params
        self.card_left = NgramCard("left", self.params.hands, self.params.left)
        self.card_right = NgramCard("right", self.params.hands, self.params.right)

    def compose(self) -> ComposeResult:
        yield self.card_left
        yield self.card_right

    def update(
        self,
        left: KeySeq | None,
        right: KeySeq | None,
    ) -> None:
        self.card_left.update(left)
        self.card_right.update(right)


class MainArea(Vertical):

    default_text = Text(
        "Select the ngram that you want to use more in your layout (least effort/cost) using LEFT and RIGHT arrow keys.",
    )

    def __init__(
        self,
        ngram_params: NgramShowParams,
    ) -> None:
        self.total_sequences: int | None = None
        self.ngram_params = ngram_params
        self.compare_col = NgramSelectionColumn(self.ngram_params)
        self.text = Label(self.default_text, id="keyseq_instructions", classes="box")
        self.text_additional = Label("", id="additional_text")
        self.log_component = Log()
        self.log_component.can_focus = False
        self.log_component.auto_scroll = True
        super().__init__()

    def set_total_sequences(self, total_sequences: int) -> None:
        self.total_sequences = total_sequences
        self.progress.update_total(self.total_sequences)

    def compose(self) -> ComposeResult:
        yield Progress(total_sequences=self.total_sequences)
        with Container(
            classes="centered-full-width-container", id="instructions-text-container"
        ):
            yield self.text
        with Container(
            classes="centered-full-width-container", id="ngram-cards-container"
        ):
            yield self.compare_col
        with Container(id="additional-text-container"):
            yield self.text_additional
        with Container(classes="centered-full-width-container", id="log-container"):
            yield self.log_component

    def set_progress(self, progress: int) -> None:
        self.progress.update(total=self.total_sequences or 0, progress=progress)

    @cached_property
    def text(self) -> Label:
        return typing.cast(Label, self.query_one("#keyseq_instructions"))

    @cached_property
    def progress(self) -> Progress:
        return self.query_one(Progress)

    def update(
        self,
        left: KeySeq | None,
        right: KeySeq | None,
        additional_text: Text | str = "",
        is_finished: bool = False,
    ) -> None:
        if is_finished:
            self.text.update(
                "🎉 All ngrams processed! Save the results (Ctrl-S) and quit (Ctrl-C)."
            )
            self.text_additional.update("")
        else:
            self.text.update(self.default_text)
            self.text_additional.update(additional_text)
        self.compare_col.update(left, right)

    def write_log(self, message: str) -> None:
        self.log_component.write_line(message)


class KeySequenceCompareApp(App):
    TITLE = "Ngram Compare"
    CSS_PATH = "compare_app.tcss"
    BINDINGS = [
        Binding("ctrl+c", "exit", "Exit"),
        Binding("ctrl+s", "save", "Save"),
        Binding("left", "select_left", "Left"),
        Binding("right", "select_right", "Right"),
        Binding("enter", "press_enter", "Next ngram (new pairs)"),
        Binding("backspace", "previous_paír", "Previous pair"),
    ]

    def __init__(self, file: Path | str, config: Config) -> None:
        """
        Parameters
        ----------
        file:
            Either initial order file (created with bigram_ranking_initial and/or viewer)
            or pickle file (saved with compare_app).
        """
        self._file = file
        self.pickle_file: Path | None = None
        self.scorer: ComparisonBasedScorer | None = None
        self.config = config
        self.hands = get_hands_data(self.config)
        self.main_area = MainArea(
            ngram_params=NgramShowParams(None, None, hands=self.hands),
        )

        super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app. This happens before on_mount."""
        with Horizontal():
            yield self.main_area
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.scorer, self.pickle_file = self.load_from_file(self._file)
        self.main_area.set_total_sequences(self.scorer.n_ngrams)
        self.set_progress(len(self.scorer.processed_key_sequences))
        self.write_log("Session started.")
        self.write_log(STARTING_INSTRUCTIONS)
        self.do_refresh()

    def action_exit(self) -> None:
        self.push_screen(ExitModal(), self.conditional_exit)

    def action_select_left(self) -> None:
        if not self.scorer or self.scorer.is_finished():
            return
        if not self.scorer.is_current_round_finished():
            self.scorer.handle_select_left()
        self._after_pressing_left_or_right()

    def action_select_right(self) -> None:
        if not self.scorer or self.scorer.is_finished():
            return
        if not self.scorer.is_current_round_finished():
            self.scorer.handle_select_right()
        self._after_pressing_left_or_right()

    def _after_pressing_left_or_right(self) -> None:
        if not self.scorer:
            return
        if self.scorer.is_current_round_finished():
            self.write_log("Round finished. Press Enter to continue.")

    def action_press_enter(self) -> None:
        if (
            not self.scorer
            or not self.scorer.is_current_round_finished()
            or self.scorer.is_finished()
        ):
            return

        if self.scorer.current_key_sequence is None:
            self.write_log("No current_key_sequence.")
            return

        self.handle_fit_start(
            self.scorer.current_key_sequence,
            len(self.scorer.processed_key_sequences),
            len(self.scorer.initial_order),
        )
        self.scorer.fit()
        if self.scorer.is_finished():
            self.scorer.refresh()
            self.main_area.update(
                None,
                None,
                is_finished=True,
            )
            self.write_log("All ngrams processed. Save the results and exit.")
            return
        self.scorer.select_next_key_sequence()
        self.write_log("Fitting ready. You should save the progress now.")

    def action_previous_paír(self) -> None:
        if not self.scorer:
            return
        self.scorer.handle_goto_previous()

    def action_save(self) -> None:
        if not self.scorer:
            return
        if self.scorer.save_to_file(str(self.pickle_file)):
            self.write_log(f"Saved state to {self.pickle_file}.")
        else:
            self.write_log("Data validation failed. Not saving the data to file.")

    def do_refresh(self) -> None:
        if not self.scorer:
            return
        is_finished = self.scorer.is_finished()

        comparison_pair: tuple[KeySeq | None, KeySeq | None]
        if is_finished:
            comparison_pair = (None, None)
        else:
            comparison_pair = self.scorer.current_comparison_pair

        self.main_area.update(*comparison_pair, is_finished=is_finished)

    def conditional_exit(self, condition: Any) -> None:
        if not isinstance(condition, bool):
            warnings.warn("Exit condition is not a boolean.")

        if condition:
            self.exit()

    def write_log(self, message: str) -> None:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self.main_area.write_log(line)

    def load_from_file(self, file: str | Path) -> tuple[ComparisonBasedScorer, Path]:
        initial_order_file, pickle_file = get_files(file)

        if initial_order_file is not None:
            self.write_log(f"Loading initial ngram order from {initial_order_file}")
            key_seqs = load_bigram_rankings(initial_order_file)
            scorer = ComparisonBasedScorer(key_seqs, app=self)
        else:
            self.write_log(f"Loading ngram order and saved state from {pickle_file}")
            scorer = ComparisonBasedScorer.load_from_file(str(pickle_file), app=self)
        return scorer, pickle_file

    def set_progress(self, progress: int) -> None:
        self.main_area.set_progress(progress)

    def handle_fit_start(
        self, key_sequence: KeySeq, processed: int, total: int
    ) -> None:
        left = self.hands.left.get_symbols_visualization(key_sequence)
        right = self.hands.right.get_symbols_visualization(key_sequence)
        self.write_log(
            f"Fitting {left} {right} {key_sequence} -- {processed+1}/{total}"
        )


PICKLE_FILE_SUFFIX = ".compare.pickle"


def get_files(file: Path | str) -> tuple[Path | None, Path]:
    file = Path(file)
    if str(file).endswith(PICKLE_FILE_SUFFIX):
        pickle_file = file
        initial_order_file = None
    else:
        pickle_file = file.with_suffix(PICKLE_FILE_SUFFIX)
        initial_order_file = file
    return initial_order_file, pickle_file


def main() -> None:
    try:
        app = KeySequenceCompareApp(sys.argv[1], config=read_config(sys.argv[2]))
    except IndexError:
        docs = """Usage:\ngranite-bigram-compare <ngram-ranking-file|saved-pickle-file> <config-file-yml>"""
        print(docs)
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
