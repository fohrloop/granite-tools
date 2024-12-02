from __future__ import annotations

from textual import on
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ExitModal(ModalScreen):

    def compose(self):
        with Vertical():
            yield Static(
                "Are you sure you want to exit? You will lose your progress if you have not saved it."
            )
            with Horizontal():
                btn_yes = Button.success(r"\[y]es", id="exit-yes")
                btn_no = Button.error(r"\[n]o", id="exit-no")
                btn_yes.can_focus = False
                btn_no.can_focus = False
                yield btn_yes
                yield btn_no

    @on(Button.Pressed)
    def exit_modal(self, event):
        if event.button.id == "exit-yes":
            self.action_press_button_yes()
        elif event.button.id == "exit-no":
            self.action_press_button_no()

    def on_mount(self):
        self._bindings.bind("y", "press_button_yes", "Press Yes", show=False)
        self._bindings.bind("n", "press_button_no", "Press No", show=False)

    def action_press_button_yes(self):
        self.dismiss(True)

    def action_press_button_no(self):
        self.dismiss(False)
