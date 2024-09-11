# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Jupytext to nbmodular
#
# > Transforming Jupytext notebooks to nbmodular notebooks, nbs documentation, python modules and tests

# %%
# | default_exp jupynbm

# %%
# |export
# Standard
import argparse
import sys
import shutil
from pathlib import Path
from typing import List
import os

# 3rd party
from jupytext.cli import jupytext

# nbmodular
from nbmodular.export import nbm_export_all_paths, nbm_update_all_paths
from .utils import create_or_get_logger, get_config


# %% [markdown]
# ## Export Jupytext modules to nbmodular notebooks and library python modules


# %%
# | export
def export_jupytext_modules(jupytext_path: str, nbm_path: str) -> None:
    """
    Export jupytext modules to nbmodular notebooks
    """
    current_path = os.getcwd()
    os.chdir(jupytext_path)

    jupytext("--set-formats ipynb,py *.py".split())
    os.chdir(current_path)
    shutil.move(f"{jupytext_path}/*.ipynb", nbm_path)
    nbm_export_all_paths(nbm_path)


def parse_argv_and_run_jupynbm(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Udpdate python modules from their corresponding notebooks."
    )

    parser.add_argument(
        "--jupytext",
        type=str,
        default=None,
        help="Path to jupytext modules",
    )
    parser.add_argument(
        "--nbm",
        type=str,
        default=None,
        help="Path to nbmodular notebooks",
    )
    args = parser.parse_args(argv)
    logger = create_or_get_logger()
    if args.jupytext is None:
        args.jupytext = str(Path(get_config()["jupy_path"]).resolve())
    if args.nbm is None:
        args.nbm = str(Path(get_config()["nbm_path"]).resolve())
    logger.info(
        f"Exporting jupytext python modules from {args.jupytext}, to {args.nbm}"
    )
    export_jupytext_modules(args.jupytext, args.nbm)


def jupynbm_export_cli():
    parse_argv_and_run_jupynbm(sys.argv)


# %% [markdown]
# ## Import library python modules to nbmodular notebooks and jupytext modules


# %%
# | export
def import_jupytext_modules(nbm_path: str, jupytext_path: str) -> None:
    """
    Export jupytext modules to nbmodular notebooks
    """
    current_path = os.getcwd()
    nbm_update_all_paths(nbm_path)
    shutil.copy(f"{nbm_path}/*.ipynb", jupytext_path)
    os.chdir(jupytext_path)
    jupytext("--set-formats ipynb,py *.ipynb".split())
    os.remove("*.ipynb")
    os.chdir(current_path)


def parse_argv_and_run_nbmjupy(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Udpdate python modules from their corresponding notebooks."
    )

    parser.add_argument(
        "--nbm",
        type=str,
        default=None,
        help="Path to nbmodular notebooks",
    )
    parser.add_argument(
        "--jupytext",
        type=str,
        default=None,
        help="Path to jupytext modules",
    )
    args = parser.parse_args(argv)
    logger = create_or_get_logger()
    if args.jupytext is None:
        args.jupytext = str(Path(get_config()["jupy_path"]).resolve())
    if args.nbm is None:
        args.nbm = str(Path(get_config()["nbm_path"]).resolve())
    logger.info(f"Exporting nbm notebooks from {args.nbmjupytext} to {args.nbm}")
    import_jupytext_modules(args.nbm, args.jupytext)


def jupynbm_import_cli():
    parse_argv_and_run_nbmjupy(sys.argv)
