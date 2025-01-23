import plotext._utility as ut  # type: ignore
from rich.text import Text
from textual.widget import Widget
from textual.widgets import Label


class PositionBar(Widget):

    def __init__(
        self, left_outer: float, left: float, right: float, right_outer: float
    ) -> None:
        self.label = Label(
            self.create_text(left_outer, left, right, right_outer),
            classes="centered-full-width-container",
        )
        super().__init__()

    def compose(self):
        yield self.label

    def update(
        self, left_outer: float, left: float, right: float, right_outer: float
    ) -> None:
        self.label.update(self.create_text(left_outer, left, right, right_outer))

    @staticmethod
    def create_text(
        left_outer: float, left: float, right: float, right_outer: float
    ) -> Text:
        return Text.from_ansi(
            get_bar(left_outer, left, right, right_outer, width=100, colors=colors)
        )


def get_bar(
    left_outer: float,
    left: float,
    right: float,
    right_outer: float,
    width: int = 100,
    colors=None,
) -> str:
    # Modified (pruned) version of simple_stacked_bar from plotext/_global.py
    bar_widths = left_outer, left, right, right_outer
    bar_data_args = ([""], [[w] for w in bar_widths])
    *_, Y, width = ut.bar_data(*bar_data_args, width=width)
    marker = ut.correct_marker(None)
    n_bars = len(Y)
    stacked_bars = len(Y[0])

    colors_ok1 = (
        isinstance(colors, list)
        and isinstance(colors[0], list)
        and ut.matrix_size(colors) == [n_bars, stacked_bars]
    )
    colors_ok2 = isinstance(colors, list) and len(colors) == stacked_bars
    colors = (
        ut.transpose(colors)
        if colors_ok1
        else (
            [colors] * n_bars
            if colors_ok2
            else [ut.color_sequence[:stacked_bars]] * n_bars
        )
    )

    x_vals = list(Y[0])
    # hack: make sure at least something is shown. The bar width is not super accurate
    # anyway (the printed width in pixels varies), so this is okay.
    if x_vals[1] == 0:
        x_vals[1] = 1
    if x_vals[2] == 0:
        x_vals[2] = 1
    return single_bar(x_vals, marker, colors[0])


def single_bar(x, marker, colors) -> str:
    # modified (pruned) version of single_bar from plotext/_utility.py
    lenofx = len(x)
    lc = len(colors)
    bar = [marker * el for el in x]
    bar = [ut.apply_ansi(bar[i], colors[i % lc], 1) for i in range(lenofx)]
    return "".join(bar)


colors = [
    (175, 255, 135),  # left outer: pale_green1 from rich
    (135, 215, 0),  # left: chartreuse2 from rich
    (95, 95, 255),  # right: royal_blue1 from rich
    (135, 175, 255),  # right outer: sky_blue2 from rich
]
