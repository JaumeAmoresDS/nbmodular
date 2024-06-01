

# %% auto 0
__all__ = ['set_log_level', 'get_repo_root_folder', 'cd_root', 'make_nb_from_cell_list', 'markdown_cell', 'code_cell']

# %% ../../nbs/utils.ipynb 2
import logging
import os
import nbdev
from pathlib import Path
from typing import List

from sklearn.utils import Bunch


# %% ../../nbs/utils.ipynb 3
def set_log_level(logger, log_level):
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)


def get_repo_root_folder(
    file_to_look_for_in_root_folder="settings.ini",
    max_parent_levels_to_traverse=10,
):
    """Gets root folder of repo where notebook is.

    It assumes that the root folder has a file called `file_to_look_for_in_root_folder`, which
    by default is `settings.ini`.
    """
    list_contents = os.listdir(".")
    traversed_parent_levels = 0
    while (
        file_to_look_for_in_root_folder not in list_contents
        and traversed_parent_levels < max_parent_levels_to_traverse
    ):
        traversed_parent_levels += 1
        os.chdir("..")
        list_contents = os.listdir(".")
    repo_root_folder = Path(".").resolve()
    return repo_root_folder


def _cd_root_nbdev_impl():
    config = nbdev.config.get_config()
    os.chdir(config.config_path)


def cd_root():
    repo_root_path = get_repo_root_folder()
    os.chdir(repo_root_path)


# %% ../../nbs/utils.ipynb 6
def make_nb_from_cell_list(cell_list: List[Bunch]):
    nb = Bunch(
        metadata=Bunch(
            kernelspec=Bunch(
                display_name="Python 3 (ipykernel)",
                language="python",
                name="python3",
            )
        ),
        cells=cell_list,
    )
    return nb


# %% ../../nbs/utils.ipynb 8
def markdown_cell(text: str):
    cell = Bunch(cell_type="markdown", metadata={}, source=text.splitlines())
    if len(cell.source[0].strip()) == 0:
        cell.source = cell.source[1:]
    cell.source = "\n".join(cell.source)
    return cell


# %% ../../nbs/utils.ipynb 10
def code_cell(text: str):
    cell = Bunch(
        cell_type="code",
        execution_count=None,
        metadata={},
        outputs=[],
        source=text.splitlines(),
    )
    if len(cell.source[0].strip()) == 0:
        cell.source = cell.source[1:]
    cell.source = "\n".join(cell.source)
    return cell
