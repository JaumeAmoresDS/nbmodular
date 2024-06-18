

# %% auto 0
__all__ = ['imported_jupytext', 'set_log_level', 'get_repo_root_folder', 'cd_root', 'get_config']

# %% ../../nbs/utils.ipynb 2
# standard
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple
from configparser import ConfigParser
import re

# 3rd party
import nbdev
from sklearn.utils import Bunch
from execnb.nbio import new_nb, write_nb

imported_jupytext = False
try:
    import jupytext as jp

    imported_jupytext = True
except ImportError:
    pass

# %% ../../nbs/utils.ipynb 4
def set_log_level(logger, log_level):
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)

# %% ../../nbs/utils.ipynb 6
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

# %% ../../nbs/utils.ipynb 7
def _cd_root_nbdev_impl():
    config = nbdev.config.get_config()
    os.chdir(config.config_path)

# %% ../../nbs/utils.ipynb 8
def cd_root():
    repo_root_path = get_repo_root_folder()
    os.chdir(repo_root_path)

# %% ../../nbs/utils.ipynb 10
def get_config(path: str = "settings.ini"):
    config = ConfigParser(delimiters=["="])
    config.read(path, encoding="utf-8")
    cfg = config["DEFAULT"]
    cfg.config_path = Path(path).resolve()
    return cfg
