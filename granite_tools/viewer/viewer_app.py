"""The Key Sequence Sorter App.

Development
------------
Run in one terminal: (this shows logs and prints)
    uv run textual console
Run in another terminal: (this runs the app)
    uv run textual run --dev app/viewer/viewer_app.py foo examples/keyseq_effort_numbers_mini.yml
"""

from __future__ import annotations

import datetime as dt
import typing
from functools import cached_property
from pathlib import Path

from rich.panel import Panel
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Label, Log, Static

from granite_tools.config import Config, read_config
from granite_tools.effort import create_permutations, get_hands_data
from granite_tools.exit_modal import ExitModal
from granite_tools.progress import Progress
from granite_tools.viewer.keyseq_table import GotoModal, KeySequenceTable

if typing.TYPE_CHECKING:
    from rich.text import Text

INSTRUCTIONS_TEXT = """
You may browse the table using the up/down arrow keys. It's possible to reorder to table (using space and up/down).

In all of the following, these short notations are used:
p: pinky, r: ring, m: middle, i: index, t: thumb

Repeats (least to most effort):
-------------------------------
REP: Repeated key (bigram/trigram)
SFS: Same Finger Skipgram (2nd key same hand)
SFB: Same Finger Bigram
RSFT: Same Finger Trigram with a repeat (two different keys )
SFT: Same Finger Trigram

rowdiff
-------
= Uncomfortable vertical movements. Depends on the row difference between two consecutive key presses. Diff is only marked if it's meaningful (depends on fingers used). From least to most effort (approx):

2u: 2 row difference
mi2u: Middle below index finger (2u)
mp1u: Middle below pinky (1u)
ip2u: Index below pinky (2u)
mr2u: Middle below ring finger (2u)
pr2u: Pinky below ring finger (2u)
mp2u: Middle below pinky (2u)
rp1u: Ring below pinky (1u)
rp2u: Ring below pinky (2u)

direction
---------
Directions track the some unfomfortable horizontal movements. From bigrams, they catch few outward and inwards rolls (involving pinky but no index). From trigrams, they additionally catch different types of redirects.

    Here are all trigrams with three different fingers used:

        p  r  m  i
       ------------
    1:  1  2  3        in (pr)
    2:  1  2     3     in (pr)
    3:  1  3  2        redir4
    4:  1  3     2     redir2
    5:  1     2  3     in (pm)
    6:  1     3  2     redir2
    7:  2  1  3        redir4
    8:  2  1     3     redir3
    9:  2  3  1        redir4
    10: 2  3     1     redir3
    11: 2     1  3     redir3
    12: 2     3  1     redir3
    13: 3  1  2        redir4
    14: 3  1     2     redir2
    15: 3  2  1        out (rp)
    16: 3  2     1     out (rp)
    17: 3     1  2     redir2
    18: 3     2  1     out (mp)
    19:    1  2  3
    20:    1  3  2     redir1
    21:    2  1  3     redir1
    22:    2  3  1     redir1
    23:    3  1  2     redir1
    24     3  2  1

From least to most effort (approximation for effort scale, arbitrary units):

redir1:   Redirect, level 1. No pinky invoved (easiest).   (0.4)
in (pm):  Inwards from pinky to middle                     (0.8)
in (pr):  Inwards from pinky to ring                       (1.5)
out (mp): Outwards from middle to pinky                    (2.5)
redir2:   Redirect, level 2. Index in the middle.          (3.1)
out (rp): Outwards from ring to pinky                        (7)
redir3:   Redirect, level 3. Index not in the middle.       (10)
redir4:   Redirect, level 4. Index not included (harderst)  (25)

"""


INSTRUCTIONS = Panel(
    INSTRUCTIONS_TEXT,
    title="Instructions",
)


class MainArea(Vertical):
    def __init__(self, total_sequences: int = 0) -> None:
        self.total_sequences = total_sequences
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Progress(total_sequences=self.total_sequences)
        with VerticalScroll():
            yield Label(INSTRUCTIONS, id="keyseq_instructions")
        log = Log()
        log.can_focus = False
        log.styles.margin = (1, 0, 0, 0)
        yield log

    def set_text(self, text: str) -> None:
        self.static.update(Panel(text))

    @cached_property
    def static(self) -> Static:
        return typing.cast(Static, self.query_one("#keyseq_instructions"))

    @cached_property
    def progress(self) -> Progress:
        return typing.cast(Progress, self.query_one(Progress))


class NgramTableViewerApp(App):
    TITLE = "Ngram Table Viewer"
    CSS_PATH = "viewer_app.tcss"
    BINDINGS = [
        Binding("h", "toggle_help", "Show/Hide Help"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+c", "exit", "Exit"),
    ]

    def __init__(self, file_out: Path | str, config: Config) -> None:
        self.file_out = Path(file_out)
        self.config = config
        self.hands = get_hands_data(self.config)
        self.permutations: list[tuple[int, ...]] = create_permutations(
            self.hands.left, self.hands.right, sequence_lengths=(1, 2)
        )
        super().__init__()
        self.help_panel_visible: bool = False
        self._current_index = 0

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Horizontal():
            yield Vertical(KeySequenceTable(cursor_type="row", hands=self.hands))
            yield MainArea(total_sequences=len(self.permutations))
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.initialize_table()

    def initialize_table(self) -> None:
        """Initializes the table. If the out file is empty (not saved progress),
        will only add a single row. Otherwise, will load the data from the file.
        """
        if self.file_out.exists():
            self.write_log(
                f"Loading data from: {self.file_out} (Note that if this file was created with different configuration file, the results may be anything.)"
            )
            with self.batch_update():
                self.table.load(str(self.file_out), self.hands)

            self.main.progress.update(
                total=len(self.permutations), progress=len(self.table)
            )
        else:
            self.write_log(
                f"Error loading data from: {self.file_out} (File does not exist)"
            )

    def action_toggle_help(self):
        """Toggle the help panel."""
        self.help_panel_visible = not self.help_panel_visible
        if self.help_panel_visible:
            self.action_show_help_panel()
        else:
            self.action_hide_help_panel()

    def action_exit(self):
        self.push_screen(ExitModal(), self.conditional_exit)

    def conditional_exit(self, condition: bool):
        if condition:
            self.exit()

    def act_all_key_sequences_placed(self):
        self.write_log("All key sequences have been placed.")
        self.table.change_to_moving_cursor()

    def write_log(self, text: Text | str):
        datetime = dt.datetime.now().strftime("%H:%M:%S")
        self.logwidget.write_line(f"[{datetime}] {text}")

    def action_save(self):
        """Save the table to a file."""
        self.table.save(self.file_out)
        self.write_log(f"Saved to: {self.file_out}")

    @cached_property
    def logwidget(self) -> Log:
        return self.query_one(Log)

    @cached_property
    def table(self) -> KeySequenceTable:
        return self.query_one(KeySequenceTable)

    @cached_property
    def main(self) -> MainArea:
        return self.query_one(MainArea)

    def on_key_sequence_table_placed(self, message: KeySequenceTable.Placed) -> None:
        """Called when a key sequence is placed."""

        self.write_log(
            f"Placed: left={message.left}, right={message.right}, row={message.row_label}"
        )

    def on_key_sequence_table_go_to_requested(
        self, _: KeySequenceTable.GoToRequested
    ) -> None:
        self.push_screen(GotoModal(), self.goto_row)  # type:ignore

    def goto_row(self, ngram: str) -> None:
        self.table.goto_row(ngram)

    def on_key_sequence_table_write_log(
        self, message: KeySequenceTable.WriteLog
    ) -> None:
        self.write_log(message.text)


if __name__ == "__main__":
    import sys

    app = NgramTableViewerApp(sys.argv[1], config=read_config(sys.argv[2]))
    app.run()
