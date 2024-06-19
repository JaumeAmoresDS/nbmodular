# ---
# jupyter:
#   jupytext:
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
# # Test Utils
#
# > Utilities for writting tests.

# %%
# | default_exp testing.utils

# %%
# |export
# standard
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import re

# 3rd party
from execnb.nbio import new_nb, write_nb, mk_cell, read_nb

# ours
from nbmodular.core.utils import cd_root

# %% [markdown]
# ## Notebook examples
#
# > Example notebooks used for testing

# %% [markdown]
# ### Simple example 1

# %%
# | export
nb1 = """
[markdown]
# First notebook

[code]
%%function hello
print ('hello')

[code]
%%function one_plus_one --test
a=1+1
print (a)
"""

# %% [markdown]
# ### Simple example 2

# %%
nb2 = """
[markdown]
# Second notebook

[code]
%%function bye
print ('bye')

[markdown]
%%function two_plus_two --test
a=2+2
print (a)
"""

# %% [markdown]
# ### Mixed Cells Example

# %%
# | export
mixed_nb1 = """
[code]
%%function
def first():
    pass

[markdown]
comment
    
[code]
%%function --test
def second ():
    pass
"""

# %% [markdown]
# ## Notebook structure
#
# > Utilities for building a dictionary with notebook structure. Useful for testing purposes.

# %% [markdown]
# ### convert_nested_nb_cells_to_dicts


# %%
# | export
def convert_nested_nb_cells_to_dicts(dict_like_with_nbcells: dict) -> dict:
    """Convert nested NbCells to dicts.

    Parameters
    ----------
    dict_like_with_nbcells : dict
        dict-like object with embedded NbCell cells

    Returns
    -------
    dict
        dict object without embedded NbCell cells
    """
    new_dict = {k: v for k, v in dict_like_with_nbcells.items()}
    new_dict["cells"] = [dict(**cell) for cell in new_dict["cells"]]
    return new_dict


# %% [markdown]
# ### parse_nb_sections


# %%
# | export
def parse_nb_sections(nb):
    # Define the regex pattern to match sections
    pattern = "\[(markdown|code)\](.*?)((?=\[markdown\])|(?=\[code\])|$)"

    # Find all matches using re.findall which returns a list of tuples
    matches = re.findall(pattern, nb, re.DOTALL)

    # Transform the matches to the required format
    result = [(match[0], match[1].strip()) for match in matches]

    return result


# %% [markdown]
# #### Example usage

# %%
nb_text = parse_nb_sections(nb1)
assert nb_text == [
    ("markdown", "# First notebook"),
    ("code", "%%function hello\nprint ('hello')"),
    ("code", "%%function one_plus_one --test\na=1+1\nprint (a)"),
]


# %% [markdown]
# ### text2nb


# %%
# | export
def text2nb(nb: str):
    cells = [
        mk_cell(text, cell_type=cell_type) for cell_type, text in parse_nb_sections(nb)
    ]
    return new_nb(cells)


# %% [markdown]
# #### Example usage

# %%
nb_text = text2nb(nb1)

# %% [markdown]
# #### checks

# %%
expected = {
    "cells": [
        {
            "cell_type": "markdown",
            "source": "# First notebook",
            "directives_": {},
            "metadata": {},
            "idx_": 0,
        },
        {
            "cell_type": "code",
            "source": "%%function hello\nprint ('hello')",
            "directives_": {},
            "metadata": {},
            "idx_": 1,
        },
        {
            "cell_type": "code",
            "source": "%%function one_plus_one --test\na=1+1\nprint (a)",
            "directives_": {},
            "metadata": {},
            "idx_": 2,
        },
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}
actual = convert_nested_nb_cells_to_dicts(
    nb_text
)  # just for comparison purposes, we convert nested NbCells to dicts
assert actual == expected


# %% [markdown]
# ### texts2nbs


# %%
# | export
def texts2nbs(nbs: List[str] | str) -> List[dict]:
    if not isinstance(nbs, list):
        nbs = [nbs]
    return [text2nb(nb) for nb in nbs]


# %% [markdown]
# ### nb2text


# %%
# | export
def nb2text(nb: dict) -> str:
    return "\n\n".join(
        [f"[{cell['cell_type']}]\n{cell['source']}" for cell in nb["cells"]]
    )


def nbs2text(nbs: List[dict]) -> List[str]:
    return [nb2text(nb) for nb in (nbs if isinstance(nbs, list) else [nbs])]


# %% [markdown]
# #### Usage example

# %%
nb_text = text2nb(nb1)
nb_text = nb2text(nb_text)
assert (
    nb_text
    == """[markdown]
# First notebook

[code]
%%function hello
print ('hello')

[code]
%%function one_plus_one --test
a=1+1
print (a)"""
)

# %%
nb_text

# %%
"""[markdown]
# First notebook

[code]
%%function hello
print ('hello')

[code]
%%function one_plus_one --test
a=1+1
print (a)"""


# %% [markdown]
# ### printnb


# %%
# | export
def printnb(
    nb_text: str | dict | List[str] | List[dict], no_newlines: bool = False, titles=None
) -> None:
    if isinstance(nb_text, list):
        assert titles is None or len(titles) == len(nb_text)
        titles = (
            ["\n"] * len(nb_text)
            if titles is None
            else ["\n" + title for title in titles]
        )
        for nb_text, title in zip(nb_text, titles):
            print(title)
            print(f"{'-'*50}")
            printnb(nb_text, no_newlines=no_newlines)
    else:
        if isinstance(nb_text, dict):
            nb_text = nb2text(nb_text)
        print(f'''"""{nb_text}"""''' if no_newlines else f'''"""\n{nb_text}\n"""''')


# %% [markdown]
# #### Usage example

# %%
print("-" * 50)
print("with new lines at beginning and end:")
printnb(nb1)
print()
print("-" * 50)
print("without new lines at beginning and end:")
printnb(nb1, no_newlines=True)

# %%
printnb([nb1, nb1], titles=["Number 1", "Number 2"], no_newlines=True)


# %% [markdown]
# ## Check utilities

# %% [markdown]
# ### strip_nb


# %%
# | export
def strip_nb(nb: str) -> str:
    return nb2text(text2nb(nb))


# %% [markdown]
# ### check_test_repo_content


# %%
# | export
def check_test_repo_content(
    current_root: str,
    new_root: str,
    nb_folder: str,
    nb_paths: List[str],  # type: ignore
    nbs: Optional[List[str]] = None,
    show_content: bool = False,
    clean: bool = False,
):
    """_summary_

    Parameters
    ----------
    current_root : Path
        _description_
    nb_paths : List[str]
        _description_
    content : Optional[List[str]], optional
        _description_, by default None
    show_content : bool, optional
        _description_, by default False
    clean : bool, optional
        _description_, by default False
    """
    assert Path(current_root).name == "nbmodular"
    new_wd = os.getcwd()

    assert Path(new_wd).resolve() == Path(f"{current_root}/{new_root}").resolve()
    os.chdir(current_root)
    assert (Path(new_root) / "settings.ini").exists()
    nb_paths: List[Path] = [
        Path(f"{new_root}/{nb_folder}/{nb_path}") for nb_path in nb_paths
    ]

    all_files = []
    for nb_path in nb_paths:
        all_files += os.listdir(nb_path.parent)
    assert all_files == [nb_path_i.name for nb_path_i in nb_paths]

    nbs_in_disk = []
    for nb_path in nb_paths:
        assert nb_path.exists()
        nbs_in_disk.append(read_nb(nb_path))

    if nbs is not None:
        assert [nb2text(nb) for nb in nbs_in_disk] == [strip_nb(nb) for nb in nbs]
    if show_content:
        printnb(nbs_in_disk, no_newlines=True)
    if clean:
        shutil.rmtree(new_root)


# %% [markdown]
# ##### Example usage

# %% [markdown]
# See checks after example usage for `create_test_content`

# %% [markdown]
# ## Create tests

# %% [markdown]
# ### create_and_cd_to_new_root_folder


# %%
# | export
def create_and_cd_to_new_root_folder(
    root_folder: str | Path,
    config_path: str | Path = "settings.ini",
) -> Path:
    """Creates `root_folder`, cds to it, and makes it act as *new root* (see below).

    In order to make it the new root, it copies the file `settings.ini`, which
    allows cd_root () find it and cd to it, and also allows some modules to load
    the global root's config from it.

    It assumes that

    Parameters
    ----------
    root_folder : str or Path
        Path to new root.
    config_path : str or Path, optional
        path to roo'ts config file, by default "settings.ini"

    Returns
    -------
    Path
        Absolute path to root_folder, as Path object.
    """
    config_path = Path(config_path)
    root_folder = Path(root_folder).absolute()
    root_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, root_folder / config_path.name)
    os.chdir(root_folder)

    return root_folder


# %% [markdown]
# ### create_test_content


# %%
# | export
def create_test_content(
    nbs: List[str] | str,
    nb_paths: Optional[List[str] | List[Path] | str | Path] = None,
    nb_folder: str = "nbm",
    new_root: str = "new_test",
    config_path: str = "settings.ini",
) -> Tuple[str, List[str]]:
    """
    Create test content for notebooks.

    Parameters:
        nbs (List[str] | str): List of notebook texts or a single notebook text.
        nb_paths (Optional[List[str] | List[Path] | str | Path]): List of notebook paths or a single notebook path.
            If None, automatically generates notebook paths based on the number of notebooks.
        nb_folder (str): Name of the notebook folder.
        new_root (str): Name of the new root folder.
        config_path (str): Path to the configuration file.

    Returns:
        Tuple[str, List[str]]: A tuple containing the current root folder path and the list of notebook paths.
    """

    # we start from the root folder of our repo
    cd_root()
    current_root = os.getcwd()

    # Convert input texts into corresponding dicts with notebook structure
    nbs = texts2nbs(nbs)

    # Generate list of nb_paths if None
    if nb_paths is None:
        nb_paths = [f"f{idx}" for idx in range(len(nbs))]
    else:
        if not isinstance(nb_paths, list):
            nb_paths = [nb_paths]
        if len(nb_paths) != len(nbs):
            raise ValueError("nb_paths must have same number of items as nbs")

    for nb, nb_path in zip(nbs, nb_paths):
        full_nb_path = Path(new_root) / nb_folder / nb_path
        full_nb_path.parent.mkdir(parents=True, exist_ok=True)
        write_nb(nb, full_nb_path)

    # Copy settings.ini in new root folder, so that this file
    # can be read later on by our export / import functions.
    # Also, cd to new root folder.
    _ = create_and_cd_to_new_root_folder(new_root, config_path)

    return current_root, nb_paths


# %% [markdown]
# #### Example usage

# %%
# just for checking later
cwd = os.getcwd()

# usage
new_root = "test_create_test_content"
nb_folder = "nbm"
current_root, nb_paths = create_test_content(
    nbs=[nb1, nb2],
    nb_paths=["first_folder/first.ipynb", "second_folder/second.ipynb"],
    nb_folder=nb_folder,
    new_root=new_root,
)

# %% [markdown]
# #### checks and cleaning

# %%
check_test_repo_content(
    current_root,
    new_root,
    nb_folder,
    nb_paths,
    nbs=[nb1, nb2],
    show_content=True,
    clean=True,
)
