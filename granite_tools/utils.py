from pathlib import Path

this_file = Path(__file__).expanduser().resolve()
ROOT_FOLDER = this_file.parent.parent
assert (
    ROOT_FOLDER / "pyproject.toml"
).exists(), (
    f"Expecting to get to the root folder of the project. Check location of {__file__}"
)
DATA_FOLDER = this_file.parent / "data"
