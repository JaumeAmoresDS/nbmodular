# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Utils
#
# > Exporting to python module

# %%
# | default_exp core.utils

# %%
# |export
# standard
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
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


# %% [markdown]
# ## Logging

# %% [markdown]
# ### set_log_level


# %%
# |export
def set_handle_and_log_level(
    logger,
    log_level,
    handler: logging.StreamHandler | logging.FileHandler = logging.StreamHandler(),
):
    logger.setLevel(log_level)
    handler.setLevel(log_level)
    logger.addHandler(handler)


# %% [markdown]
# ### set_log_level


# %%
# |export
def set_log_level(logger, log_level):
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)


# %% [markdown]
# ### set_logger


# %%
# |export
def create_or_get_logger(
    name: str = "nbmodular",
    log_level: Optional[str] = None,
    log_path: Optional[str] = "logs",
    file_name: str = "log.log",
    to_file=True,
):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        set_handle_and_log_level(
            logger,
            log_level,
        )
        if to_file:
            full_log_path = (
                Path(log_path) / file_name if log_path is not None else Path(file_name)
            )
            full_log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(full_log_path)
            set_handle_and_log_level(logger, log_level, handler=file_handler)
    elif log_level is not None:
        set_log_level(logger, log_level)
    return logger


# %% [markdown]
# ## cd_root


# %%
# | export
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


# %%
# | export
def _cd_root_nbdev_impl():
    config = nbdev.config.get_config()
    os.chdir(config.config_path)


# %%
# | export
def cd_root():
    repo_root_path = get_repo_root_folder()
    os.chdir(repo_root_path)


# %% [markdown]
# ## get_config


# %%
# | export
def get_config(path: str = "settings.ini"):
    config = ConfigParser(delimiters=["="])
    config.read(path, encoding="utf-8")
    cfg = config["DEFAULT"]
    cfg.config_path = Path(path).resolve()
    return cfg


# %% [markdown]
# ### Example usage

# %%
cd_root()

cfg = get_config()

print(cfg.config_path)
