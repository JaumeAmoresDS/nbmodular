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
from typing import List, Tuple
import re

# 3rd party
from execnb.nbio import new_nb, write_nb, mk_cell

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
## First notebook

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
## Second notebook

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
nb = parse_nb_sections(nb1)
assert nb == [
    ("markdown", "## First notebook"),
    ("code", "%%function hello\nprint ('hello')"),
    ("code", "%%function one_plus_one --test\na=1+1\nprint (a)"),
]


# %% [markdown]
# ### create_notebook


# %%
# | export
def create_notebook(nb: str):
    cells = [
        mk_cell(text, cell_type=cell_type) for cell_type, text in parse_nb_sections(nb)
    ]
    return new_nb(cells)


# %% [markdown]
# #### Example usage

# %%
nb = create_notebook(nb1)
assert nb == {
    "cells": [
        {
            "cell_type": "markdown",
            "directives_": {},
            "idx_": 0,
            "metadata": {},
            "source": "## First notebook",
        },
        {
            "cell_type": "code",
            "directives_": {},
            "idx_": 1,
            "metadata": {},
            "source": "%%function hello\nprint ('hello')",
        },
        {
            "cell_type": "code",
            "directives_": {},
            "idx_": 2,
            "metadata": {},
            "source": "%%function one_plus_one --test\na=1+1\nprint (a)",
        },
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}

# %% [markdown]
# ### nb_to_text


# %%
# | export
def nb_to_text(nb: dict) -> str:
    return "\n\n".join(
        [f"[{cell['cell_type']}]\n{cell['source']}" for cell in nb["cells"]]
    )


# %% [markdown]
# #### Usage example
# nb = create_notebook(nb1)
# nb_text = nb_to_text(nb)
# assert (
#     nb_text
#     == """[markdown]
# # First notebook
#
# [code]
# %%function hello
# print ('hello')
#
# [code]
# %%function one_plus_one --test
# a=1+1
# print (a)"""
# )

# %% [markdown]
# ### printnb


# %%
# | export
def printnb(nb_text: str, no_newlines: bool = False) -> None:
    print(f'''"""{nb_text}"""''' if no_newlines else f'''"""\n{nb_text}\n"""''')


# %% [markdown]
# #### Usage example
# print("-" * 50)
# print("with new lines at beginning and end:")
# printnb(nb1)
# print()
# print("-" * 50)
# print("without new lines at beginning and end:")
# printnb(nb1, no_newlines=True)

# %% [markdown]
# ### create_notebooks


# %%
# | export
def create_notebooks(nbs: List[str] | str) -> List[dict]:
    if not isinstance(nbs, list):
        nbs = [nbs]
    return [create_notebook(nb) for nb in nbs]


# %% [markdown]
# ## create_and_cd_to_new_root_folder


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
# ## create_test_content


# %%
# | export
def create_test_content(
    nbs: List[str] | str,
    nb_paths: Optional[List[str] | str] = None,
    nb_folder="nbm",
    new_root="new_test",
    config_path="settings.ini",
) -> Tuple[str, List[str]]:
    # we start from the root folder of our repo
    cd_root()
    current_root = os.getcwd()

    # Convert input texts into corresponding dicts with notebook structure
    nbs = create_notebooks(nbs)

    # Generate list of nb_paths if None
    if nb_paths is None:
        nb_paths = [f"f{idx}" for idx in range(len(nbs))]
    elif not isinstance(nb_paths, list):
        nb_paths = [nb_paths]
    if not len(nb_paths) == len(nbs):
        raise ValueError("nb_paths have same number of elements as nbs")
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
cwd = os.getcwd()

# %%
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
assert Path(current_root).name == "nbmodular"
os.chdir(current_root)
new_wd = os.getcwd()
assert Path(new_wd).resolve() == Path(f"{current_root}/{new_root}").resolve()
nb_paths = [Path(f"{new_root}/{nb_folder}/{nb_path}") for nb_path in nb_paths]
assert (Path(new_root) / "settings.ini").exists()
all_files = []
for nb_path in nb_paths:
    all_files += os.listdir(nb_path.parent)
assert all_files == ["first.ipynb", "second.ipynb"]
nbs_in_disk = []
for nb_path in nb_paths:
    assert nb_path.exists()
    nbs_in_disk.append(read_nb(nb_path))

content = [[c["source"] for c in nb.cells] for nb in nbs_in_disk]
assert content == [
    [
        "## First notebook",
        "%%function hello\nprint ('hello')",
        "%%function one_plus_one --test\na=1+1\nprint (a)",
    ],
    [
        "## Second notebook",
        "%%function bye\nprint ('bye')",
        "%%function two_plus_two --test\na=2+2\nprint (a)",
    ],
]

shutil.rmtree(new_root)
