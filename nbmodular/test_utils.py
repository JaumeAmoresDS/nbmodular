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
# # Test Utils
#
# > Utilities for writting tests.

# %%
# | default_exp test_utils

# %%
# |export
# standard
from code import interact
import logging
from math import log
import os
import shutil
from pathlib import Path
from token import OP
from typing import List, Tuple, Optional, Union, Dict
import re

# 3rd party
from execnb.nbio import new_nb, write_nb, mk_cell, read_nb
from plum import Val
from requests import post
from fastcore.basics import AttrDict
import joblib

# ours
from nbmodular.utils import cd_root

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
# ## Python module examples
#
# > Example python modules used here

# %% [markdown]
# ### Example 1

# %%
# | export
py1 = """
def hello ():
    print ('hello')

def one_plus_one ():
    a=1+1
    print (a)
"""

# %% [markdown]
# ### Simple example 2

# %%
py2 = """
def bye ():
    print ('bye')

def two_plus_two ():
    a=2+2
    print (a)
"""


# %% [markdown]
# ### Updated example 1

# %%
exported_nbs = [
    # nbm/mixed/mixed_cells.ipynb
    """
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
""",
    # nbs/mixed/mixed_cells.ipynb
    """
[code]
#|export
def first():
    pass

[markdown]
comment

[code]
pass
""",
    # .nbs/mixed/mixed_cells.ipynb
    """
[code]
#|default_exp mixed.mixed_cells

[code]
#|export
#@@function
def first():
    pass
""",
    # .nbs/mixed/test_mixed_cells.ipynb
    """
[code]
#|default_exp tests.mixed.test_mixed_cells

[code]
#|export
#@@function --test
def second():
    pass
""",
]

exported_nb_paths = [
    "nbm/mixed/mixed_cells.ipynb",
    "nbs/mixed/mixed_cells.ipynb",
    ".nbs/mixed/mixed_cells.ipynb",
    ".nbs/mixed/test_mixed_cells.ipynb",
]

updated_py_modules = [
    # nbmodular/mixed/mixed_cells.py
    """
# @%% auto 0
__all__ = ['first']

# @%% ../../nbs/mixed/mixed_cells.ipynb 1
#@@function
def first():
    x = 3 + 1
""",
    # nbmodular/tests/mixed/test_mixed_cells.py
    """
# @%% auto 0
__all__ = ['second']

# @%% ../../../nbs/mixed/test_mixed_cells.ipynb 1
#@@function --test
def second():
    print("hello")
""",
]
updated_py_paths = [
    "nbmodular/mixed/mixed_cells.py",
    "nbmodular/tests/mixed/test_mixed_cells.py",
]
updated_cell_types_lists = [
    ["code", "original", "test"],
]
updated_cell_types_paths = [Path(".nbmodular/mixed/cell_types_mixed_cells.pk")]

# %% [markdown]
# ## Multiple updated examples

# %%
multiple_exported_nbs = [
    # nbm/first_folder/first.ipynb
    """
[markdown]
# First notebook

[code]
%%function hello
print ('hello')

[code]
%%function one_plus_one --test
a=1+1
print (a)
""",
    # nbs/first_folder/first.ipynb
    """
[markdown]
# First notebook

[code]
#|export
def hello():
    print ('hello')

[code]
a=1+1
print (a)
""",
    # nbs/first_folder/test_first.ipynb
    """
[code]
#|default_exp first_folder.first

[code]
#|export
#@@function hello
def hello():
    print ('hello')
""",
    # .nbs/first_folder/first.ipynb
    """
[code]
#|default_exp tests.first_folder.test_first

[code]
#|export
#@@function one_plus_one --test
def one_plus_one():
    a=1+1
    print (a)
""",
    # .nbs/first_folder/test_first.ipynb
    """
[markdown]
# Second notebook

[code]
%%function bye
print ('bye')

[markdown]
%%function two_plus_two --test
a=2+2
print (a)
""",
    # nbm/second_folder/second.ipynb
    """
[markdown]
# Second notebook

[code]
#|export
def bye():
    print ('bye')

[markdown]
%%function two_plus_two --test
a=2+2
print (a)
""",
    # nbs/second_folder/second.ipynb
    """
[code]
#|default_exp second_folder.second

[code]
#|export
#@@function bye
def bye():
    print ('bye')
""",
]

multiple_exported_nb_paths = [
    "nbm/first_folder/first.ipynb",
    "nbs/first_folder/first.ipynb",
    "nbs/first_folder/test_first.ipynb",
    ".nbs/first_folder/first.ipynb",
    ".nbs/first_folder/test_first.ipynb",
    "nbm/second_folder/second.ipynb",
    "nbs/second_folder/second.ipynb",
]

multiple_updated_py_modules = [
    # nbmodular/first_folder/first.py
    f"""


# @%% auto 0
__all__ = ['hello']

# @%% ../../nbs/first_folder/first.ipynb 1
#@@function hello
def hello(name):
    print ('hello', name)

""",
    # nbmodular/tests/first_folder/test_first.py
    f"""


# @%% auto 0
__all__ = ['one_plus_one']

# @%% ../../../nbs/first_folder/test_first.ipynb 1
#@@function x_plus_y --test
def x_plus_y (x, y):
    a=x+y
    print (x, '+', y, '=', a)


""",
    # nbmodular/second_folder/second.py
    """


# @%% auto 0
__all__ = ['bye']

# @%% ../../nbs/second_folder/second.ipynb 1
#@@function bye
def bye(name):
    print ('bye', name)


""",
]

multiple_updated_py_paths = [
    "nbmodular/first_folder/first.py",
    "nbmodular/tests/first_folder/test_first.py",
    "nbmodular/second_folder/second.py",
]

multiple_updated_cell_types_lists = [
    ["original", "code", "test"],
    ["original", "code", "original"],
]

multiple_updated_cell_types_paths = [
    Path(".nbmodular/first_folder/cell_types_first.pk"),
    Path(".nbmodular/second_folder/cell_types_second.pk"),
]


# %%


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
def text2nb(nb: str) -> dict | AttrDict:
    cells = [
        mk_cell(text, cell_type=cell_type) for cell_type, text in parse_nb_sections(nb)
    ]
    return new_nb(cells)  # type: ignore


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
# ### read_nbs_in_repo


# %%
# | export
def read_nbs_in_repo(
    nb_paths: List[str],  # type: ignore
    new_root: str = "new_test",
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
    print_as_list: bool = False,
    print: bool = False,
    logger: logging.Logger = None,
    previous_text: str = "",
    posterior_text: str = "",
):
    """
    Read notebooks in a repository.

    Parameters
    ----------
    nb_paths : List[str]
        List of notebook paths.
    new_root : str, optional
        New root directory, by default "new_test".
    nbm_folder : str, optional
        Folder name for nbm, by default "nbm".
    tmp_folder : str, optional
        Temporary folder name, by default ".nbs".
    nbs_folder : str, optional
        Folder name for nbs, by default "nbs".
    print_as_list : bool, optional
        Whether to print the files as a list, by default False.
    print : bool, optional
        Whether to print the files, by default False.
    previous_text : str, optional
        Text to print before the files, by default "".
    posterior_text : str, optional
        Text to print after the files, by default "".

    Returns
    -------
    content : dict
        Dictionary containing the content of the notebooks.
    """
    nb_paths = derive_nb_paths(
        nb_paths,
        new_root,
        nbm_folder=nbm_folder,
        tmp_folder=tmp_folder,
        nbs_folder=nbs_folder,
    )
    if logger is not None:
        logger.debug(f"Reading notebooks in {nb_paths}")
    content = read_nbs(nb_paths)
    if print:
        print_files(
            content,
            print_as_list=print_as_list,
            paths=nb_paths,
            previous_text=previous_text,
            posterior_text=posterior_text,
        )
    return content


# %% [markdown]
# ### read_pymodules_in_repo


# %%
# | export
def read_pymodules_in_repo(
    nb_paths: List[str],  # type: ignore
    new_root: str = "new_test",
    lib_folder: str = "nbmodular",
    print_as_list: bool = False,
    print: bool = False,
    previous_text: str = "",
    posterior_text: str = "",
    interactive_notebook: bool = True,
):
    """
    Read Python modules in a repository.

    Parameters:
    ----------
    nb_paths : List[str]
        List of paths to Jupyter notebooks.
    new_root : str, optional
        New root directory for the notebooks, by default "new_test".
    lib_folder : str, optional
        Name of the library folder, by default "nbmodular".
    print_as_list : bool, optional
        Whether to print the files as a list, by default False.
    print : bool, optional
        Whether to print the files, by default False.
    interactive_notebook : bool, optional
        Whether the notebook is run in VSC interactive mode, by default True.

    Returns:
    -------
    content : str
        Content of the Python modules.

    """
    py_paths = derive_py_paths(nb_paths, new_root, lib_folder=lib_folder)
    content = read_text_files(py_paths)
    if interactive_notebook:
        content = [x.replace("%%", "@%%") for x in content]
    if print:
        print_files(
            content,
            print_as_list=print_as_list,
            paths=py_paths,
            previous_text=previous_text,
            posterior_text=posterior_text,
        )
    return content


# %% [markdown]
# ### read_cell_types_lists


# %%
# | export
def read_cell_types_lists(
    paths: List[str | Path], must_exist: Dict[str | Path, bool] = {}
) -> List[str]:
    """
    Read the contents of Python modules from the given paths.

    Parameters
    ----------
    paths : List[str]
        A list of file paths to Python modules.

    Returns
    -------
    List[str]
        A list of strings containing the contents of the Python modules.

    Raises
    ------
    AssertionError
        If a file path does not exist.

    """
    cell_types_lists = []
    paths = [Path(path) for path in paths]
    for path in paths:
        # Check that file exists. useful for being called inside a test utility
        # to see where it fails.
        if path.exists():
            cell_types_lists.append(joblib.load(path))
        elif must_exist.get(path, False):
            raise FileNotFoundError(f"File {path} does not exist")

    return cell_types_lists


# %% [markdown]
# ### read_cell_types_lists_in_repo


# %%
# | export
def read_cell_types_lists_in_repo(
    nb_paths: List[str],  # type: ignore
    new_root: str = "new_test",
    cell_types_folder: str = ".nbmodular",
    print_as_list: bool = False,
    tab_size: int = 4,
) -> List[List[str]]:
    """
    Read Python modules in a repository.

    Parameters:
    ----------
    nb_paths : List[str]
        List of paths to Jupyter notebooks.
    new_root : str, optional
        New root directory for the notebooks, by default "new_test".
    lib_folder : str, optional
        Name of the library folder, by default "nbmodular".
    print_as_list : bool, optional
        Whether to print the files as a list, by default False.
    print : bool, optional
        Whether to print the files, by default False.
    interactive_notebook : bool, optional
        Whether the notebook is run in VSC interactive mode, by default True.

    Returns:
    -------
    content : str
        Content of the Python modules.

    """
    cell_types_paths = derive_cell_types_paths(
        nb_paths, new_root, cell_types_folder=cell_types_folder
    )
    cell_types_lists = read_cell_types_lists(cell_types_paths)

    if print_as_list:
        print("cell_types_lists = [")
        for x in cell_types_lists:
            print(f"{' '*tab_size}{x},")
        print("]")
        print(f"cell_types_paths={cell_types_paths}")
    return cell_types_lists


# %% [markdown]
# ### read_content_in_repo


# %%
# | export
def read_content_in_repo(
    nb_paths: List[str],
    new_root: Union[str, Path],
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
    lib_folder: Optional[str] = "nbmodular",
    cell_types_folder: Optional[str] = ".nbmodular",
    print_as_list: bool = False,
    print: bool = True,
    interactive_notebook: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Read the content in a repository.

    Parameters:
    ----------
    nb_paths : List[str]
        List of notebook paths.
    new_root : Union[str, Path]
        New root directory.
    nbm_folder : str, optional
        Folder name for nbm files. Defaults to "nbm".
    tmp_folder : str, optional
        Temporary folder name. Defaults to ".nbs".
    nbs_folder : str, optional
        Folder name for nbs files. Defaults to "nbs".
    lib_folder : str, optional
        Folder name for nbmodular files. Defaults to "nbmodular".
    print_as_list : bool, optional
        Whether to print the content as a list. Defaults to False.
    print : bool, optional
        Whether to print the content. Defaults to True.

    Returns:
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists - the nbs content and the py_modules content.
    """
    if interactive_notebook and not print_as_list:
        raise ValueError(
            "interactive_notebook can only be True if print_as_list is True"
        )
    if print_as_list:
        previous_text = "expected_nbs = "
    nbs = read_nbs_in_repo(
        nb_paths,
        new_root,
        nbm_folder,
        tmp_folder,
        nbs_folder,
        print_as_list,
        print,
        previous_text=previous_text,
    )
    if print_as_list:
        previous_text = "expected_py_modules = "
    py_modules = (
        read_pymodules_in_repo(
            nb_paths,
            new_root,
            lib_folder,
            print_as_list,
            print,
            previous_text=previous_text,
            interactive_notebook=interactive_notebook,
        )
        if lib_folder is not None
        else []
    )

    cell_types_lists = (
        read_cell_types_lists_in_repo(
            nb_paths,
            new_root,
            cell_types_folder,
            print_as_list,
        )
        if cell_types_folder is not None
        else []
    )

    if cell_types_folder is not None:
        return nbs, py_modules, cell_types_lists
    else:
        return nbs, py_modules


# %% [markdown]
# ### check_nbs


# %% [markdown]
# ### derive_nb_paths


# %%
# | export
from typing import List, Optional
from pathlib import Path


def derive_nb_paths(
    nb_paths: List[str],
    new_root: str | Path,
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
) -> List[Path]:
    """
    Derives the paths of notebooks based on the given parameters.

    Parameters
    ----------
    nb_paths : List[str]
        A list of notebook paths.
    new_root : str | Path
        The new root directory where the notebooks will be located.
    nbm_folder : Optional[str], optional
        The folder name for nbm files. Defaults to "nbm".
    tmp_folder : Optional[str], optional
        The temporary folder name. Defaults to ".nbs".
    nbs_folder : Optional[str], optional
        The folder name for nbs files. Defaults to "nbs".

    Returns
    -------
    List[Path]
        A list of derived notebook paths.
    """
    all_nb_paths = []
    for nb_path in nb_paths:
        if nbm_folder is not None:
            all_nb_paths.append(Path(new_root) / nbm_folder / nb_path)
        if nbs_folder is not None:
            nb_code_path = Path(new_root) / nbs_folder / nb_path
            all_nb_paths.append(nb_code_path)
            nb_test_path = nb_code_path.parent / f"test_{nb_code_path.name}"
            all_nb_paths.append(nb_test_path)
        if tmp_folder is not None:
            tmp_nb = Path(new_root) / tmp_folder / nb_path
            all_nb_paths.append(tmp_nb)
            tmp_test_nb = tmp_nb.parent / f"test_{tmp_nb.name}"
            all_nb_paths.append(tmp_test_nb)

    return all_nb_paths


# %% [markdown]
# ### derive_py_paths


# %%
# | export
def derive_py_paths(
    nb_paths: List[str],
    new_root: str | Path,
    lib_folder: str = "nbmodular",
):
    py_paths = []
    for nb_path in nb_paths:
        original_nb_path = Path(nb_path)
        py_paths.append(
            Path(new_root)
            / lib_folder
            / original_nb_path.parent
            / f"{original_nb_path.stem}.py"
        )
        py_paths.append(
            Path(new_root)
            / lib_folder
            / "tests"
            / original_nb_path.parent
            / f"test_{original_nb_path.stem}.py"
        )
    return py_paths


# %% [markdown]
# ### derive_cell_types_paths


# %%
# | export
def derive_cell_types_paths(
    nb_paths: List[str],
    new_root: str | Path,
    cell_types_folder: str = ".nbmodular",
):
    cell_types_paths = []
    for nb_path in nb_paths:
        original_nb_path = Path(nb_path)
        cell_types_paths.append(
            Path(new_root)
            / cell_types_folder
            / original_nb_path.parent
            / f"cell_types_{original_nb_path.stem}.pk"
        )
    return cell_types_paths


# %% [markdown]
# ### derive_all_paths


# %%
# | export
def derive_all_paths(
    nb_paths: List[str],
    new_root: str | Path,
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
    lib_folder: Optional[str] = "nbmodular",
    cell_types_folder: Optional[str] = ".nbmodular",
):
    all_nb_paths = derive_nb_paths(
        nb_paths,
        new_root,
        nbm_folder=nbm_folder,
        tmp_folder=tmp_folder,
        nbs_folder=nbs_folder,
    )
    py_paths = (
        derive_py_paths(nb_paths, new_root, lib_folder=lib_folder)
        if lib_folder is not None
        else []
    )
    cell_types_paths = (
        derive_cell_types_paths(nb_paths, new_root, cell_types_folder=cell_types_folder)
        if cell_types_folder is not None
        else []
    )
    return all_nb_paths, py_paths, cell_types_paths


# %% [markdown]
# #### Example usage

# %%
nb_paths, py_paths, _ = derive_all_paths(
    nb_paths=["folder_A/nb_A.ipynb", "folder_B/nb_B.ipynb"], new_root="tmp_repo"
)
assert nb_paths == [
    Path("tmp_repo/nbm/folder_A/nb_A.ipynb"),
    Path("tmp_repo/nbs/folder_A/nb_A.ipynb"),
    Path("tmp_repo/nbs/folder_A/test_nb_A.ipynb"),
    Path("tmp_repo/.nbs/folder_A/nb_A.ipynb"),
    Path("tmp_repo/.nbs/folder_A/test_nb_A.ipynb"),
    Path("tmp_repo/nbm/folder_B/nb_B.ipynb"),
    Path("tmp_repo/nbs/folder_B/nb_B.ipynb"),
    Path("tmp_repo/nbs/folder_B/test_nb_B.ipynb"),
    Path("tmp_repo/.nbs/folder_B/nb_B.ipynb"),
    Path("tmp_repo/.nbs/folder_B/test_nb_B.ipynb"),
]
assert py_paths == [
    Path("tmp_repo/nbmodular/folder_A/nb_A.py"),
    Path("tmp_repo/nbmodular/tests/folder_A/test_nb_A.py"),
    Path("tmp_repo/nbmodular/folder_B/nb_B.py"),
    Path("tmp_repo/nbmodular/tests/folder_B/test_nb_B.py"),
]

# %% [markdown]
# ### read_nbs


# %%
# | export
def read_nbs(
    paths: List[str] | List[Path], must_exist: dict = {}, as_text: bool = True
) -> List[str] | List[dict]:
    """
    Read notebooks from disk.

    Parameters:
        paths (List[str]): A list of paths to the notebooks.
        as_text (bool, optional): If True, the notebooks will be returned as text.
            If False, the notebooks will be returned as dictionaries.
            Defaults to True.

    Returns:
        List[str] | List[dict]: A list of notebooks. If `as_text` is True, the notebooks
            will be returned as text. If `as_text` is False, the notebooks will be
            returned as dictionaries.
    """
    nbs_in_disk = []
    paths = [Path(path) for path in paths]
    for path in paths:
        # Check that file exists. useful for being called inside a test utility
        # to see where it fails.
        if path.exists():
            nbs_in_disk.append(read_nb(path))
        elif must_exist.get(path, False):
            raise FileNotFoundError(f"File {path} does not exist")

    return [strip_nb(nb2text(nb)) for nb in nbs_in_disk] if as_text else nbs_in_disk


# %% [markdown]
# ### write_nbs


# %%
def write_nbs(nbs: List[str], nb_paths: List[str]) -> None:
    for nb, path in zip(nbs, nb_paths):
        write_nb(text2nb(nb), path)


# %% [markdown]
# ### compare_nbs


# %%
# | export
def compare_nb(nb1: str, nb2: str) -> bool:
    return strip_nb(nb1) == strip_nb(nb2)


def compare_nbs(nbs1: List[str], nbs2: List[str]) -> bool:
    return all(map(compare_nb, nbs1, nbs2))


# %% [markdown]
# #### Example usage

# %%
nbs = [nb1, nb2]
nb_paths = ["first.ipynb", "second.ipynb"]
write_nbs(nbs, nb_paths)
nbs_in_disk = read_nbs(nb_paths)
assert compare_nbs(nbs_in_disk, nbs)
for nb_path in nb_paths:
    Path(nb_path).unlink()

# %% [markdown]
# ### read_text_files


# %%
# | export
def read_text_files(
    paths: List[str | Path], must_exist: Dict[str | Path, bool] = {}
) -> List[str]:
    """
    Read the contents of Python modules from the given paths.

    Parameters
    ----------
    paths : List[str]
        A list of file paths to Python modules.

    Returns
    -------
    List[str]
        A list of strings containing the contents of the Python modules.

    Raises
    ------
    AssertionError
        If a file path does not exist.

    """
    text_files = []
    paths = [Path(path) for path in paths]
    for path in paths:
        # Check that file exists. useful for being called inside a test utility
        # to see where it fails.
        if path.exists():
            text_files.append(path.read_text())
        elif must_exist.get(path, False):
            raise FileNotFoundError(f"File {path} does not exist")

    return text_files


# %% [markdown]
# ### write_text_files


# %%
# | export
def write_text_files(texts: List[str], paths: List[str]) -> None:
    for text, path in zip(texts, paths):
        with open(path, "wt") as file:
            file.write(text)


# %% [markdown]
# ### compare_texts


# %%
# | export
def compare_texts(texts1: List[str], texts2: List[str]) -> bool:
    return all(map(lambda x, y: x.strip() == y.strip(), texts1, texts2))


# %% [markdown]
# #### Example usage

# %%
texts = [py1, py2]
paths = ["first.py", "second.py"]
write_text_files(texts, paths)
texts_in_disk = read_text_files(paths)
assert compare_texts(texts_in_disk, texts)

# clean
for path in paths:
    Path(path).unlink()

# %% [markdown]
# ### read_and_print

# %% [markdown]
# ### print_files


# %%
# | export
def print_files(
    files: List[str],
    print_as_list: bool = False,
    paths: Optional[List[str] | List[Path]] = None,
    previous_text: str = "",
    posterior_text: str = "",
) -> None:
    print(previous_text, end="")
    if print_as_list:
        print("[")
    suffix_path = "# " if print_as_list else ""
    for idx, file in enumerate(files):
        if not print_as_list:
            print(f"{'-'*50}")
        if paths is not None:
            print(f"{suffix_path}{paths[idx]}")
        print('"""')
        print(file)
        print('"""', end="")
        if print_as_list and idx < (len(files) - 1):
            print(",")
        else:
            print()
    if print_as_list:
        print("]")
    print(posterior_text, end="")


# %%
# | export
def read_and_print(
    paths: List[str], file_type: str, print_as_list: bool = False
) -> None:
    if file_type == "notebook":
        files = read_nbs(paths)
    elif file_type == "text":
        files = read_text_files(paths)
    else:
        raise ValueError(f"file_type {file_type} not recognized")

    print_files(
        files, print_as_list=print_as_list, paths=None if not print_as_list else paths
    )


# %% [markdown]
# ## check generated notebooks and python modules

# %% [markdown]
# ### check_py_modules


# %%
# | export
def check_nbs(
    nb_paths: List[str],  # type: ignore
    expected: List[str],
    new_root: str,  # type: ignore
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
):
    """
    Check if the notebooks in the given paths match the expected notebooks.

    Parameters
    ----------
    nb_paths : List[str]
        List of paths to the notebooks to be checked.
    expected : List[str]
        List of paths to the expected notebooks.
    new_root : str
        The new root directory for the notebooks.
    nbm_folder : str, optional
        The folder name for the notebook metadata (default is "nbm").
    tmp_folder : str, optional
        The temporary folder name for storing intermediate files (default is ".nbs").
    nbs_folder : str, optional
        The folder name for the processed notebooks (default is "nbs").

    Raises
    ------
    AssertionError
        If the actual notebooks do not match the expected notebooks.

    """
    actual = read_nbs_in_repo(
        nb_paths,
        new_root,
        nbm_folder=nbm_folder,
        tmp_folder=tmp_folder,
        nbs_folder=nbs_folder,
    )
    assert compare_nbs(actual, expected)


# %% [markdown]
# ### check_py_modules


# %%
# | export
def check_py_modules(
    nb_paths: List[str],  # type: ignore
    expected: List[str],
    new_root: str,  # type: ignore
    lib_folder: str = "nbmodular",
    interactive_notebook: bool = True,
):
    """
    Check if the Python modules in the given notebook paths match the expected modules.

    Parameters
    ----------
    nb_paths : List[str]
        List of paths to the notebooks.
    expected : List[str]
        List of expected Python modules.
    new_root : str
        The new root directory.
    lib_folder : str, optional
        The name of the library folder, by default "nbmodular".
    interactive_notebook: bool, optional
        Whether the notebook is run in VSC interactive mode, by default True.

    Raises
    ------
    AssertionError
        If the actual Python modules do not match the expected modules.
    """
    actual = read_pymodules_in_repo(
        nb_paths,
        new_root,
        lib_folder=lib_folder,
        interactive_notebook=interactive_notebook,
    )
    assert compare_texts(actual, expected)


# %% [markdown]
# ### check_test_repo_content


# %%
# | export
def check_test_repo_content(
    nb_paths: List[str],
    expected_nbs: Optional[List[str]] = None,
    expected_py_modules: Optional[List[str]] = None,
    current_root: Optional[str] = None,
    new_root: Optional[str] = None,
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
    lib_folder: Optional[str] = "nbmodular",
    clean: bool = False,
    keep_cwd: bool = False,
):
    """
    Check the content of a test repository.

    This function checks the content of a test repository based on the provided parameters.
    It verifies the structure and presence of notebooks and Python modules.

    Parameters
    ----------
    nb_paths : List[str]
        The list of notebook paths to check.
    expected_nbs : List[str], optional
        The list of expected notebook filenames, by default None.
    expected_py_modules : List[str], optional
        The list of expected Python module filenames, by default None.
    current_root : Optional[str], optional
        The current root directory of the test repository, by default None.
    new_root : Optional[str], optional
        The new root directory of the test repository, by default None.
    nbm_folder : str | None, optional
        The name of the folder containing the notebook modules, by default "nbm".
    tmp_folder : str | None, optional
        The name of the temporary folder, by default ".nbs".
    nbs_folder : str | None, optional
        The name of the folder containing the notebooks, by default "nbs".
    lib_folder : str | None, optional
        The name of the folder containing the Python modules, by default "nbmodular".
    clean : bool, optional
        Whether to clean the new root directory after checking, by default False.
    keep_cwd : bool, optional
        Whether to keep the current working directory after checking, by default False.

    Raises
    ------
    ValueError
        Raised when both clean and keep_cwd are set to True.
    """
    changed_dir = False
    if current_root is not None:
        assert Path(current_root).name == "nbmodular"
        new_wd = os.getcwd()

        assert Path(new_wd).resolve() == Path(f"{current_root}/{new_root}").resolve()
        os.chdir(current_root)
        changed_dir = True
    if new_root is not None:
        if not (Path(new_root) / "settings.ini").exists():
            if changed_dir:
                os.chdir(new_wd)
            raise FileNotFoundError(f"settings.ini not found in {new_root}")
        use_new_root = True
    else:
        new_root = "./"
        use_new_root = False

    if expected_nbs is not None:
        try:
            check_nbs(
                nb_paths, expected_nbs, new_root, nbm_folder, tmp_folder, nbs_folder
            )
        except AssertionError as e:
            if changed_dir:
                os.chdir(new_wd)
            raise e
    if expected_py_modules is not None:
        try:
            check_py_modules(nb_paths, expected_py_modules, new_root, lib_folder)
        except AssertionError as e:
            if changed_dir:
                os.chdir(new_wd)
            raise e
    if clean and use_new_root:
        shutil.rmtree(new_root)
    if keep_cwd and use_new_root:
        if clean:
            if changed_dir:
                os.chdir(new_wd)
            raise ValueError("keep_cwd can't be True if clean is True")
        os.chdir(new_root)


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
    nbs: List[str] | str | None = None,
    nb_paths: Optional[List[str] | List[Path] | str | Path] = None,
    nb_folder: str = "nbm",
    py_modules: List[str] | str | None = None,
    py_paths: Optional[List[str] | List[Path] | str | Path] = None,
    lib_folder: Optional[str] = "nbmodular",
    cell_types_lists: List[List[str]] | None = None,
    cell_types_paths: Optional[str | Path] = None,
    cell_types_folder: str | Path = ".nbmodular",
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
    nbs = texts2nbs(nbs) if nbs is not None else []

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

    if py_modules is not None:
        py_modules = [py_modules] if isinstance(py_modules, str) else py_modules
        if py_paths is None:
            py_paths = [f"f{idx}" for idx in range(len(py_modules))]
        else:
            if not isinstance(py_paths, list):
                py_paths = [py_paths]
            if len(py_paths) != len(py_modules):
                raise ValueError(
                    "py_paths must have same number of items as py_modules"
                )

        for py_module, py_path in zip(py_modules, py_paths):
            full_py_path = Path(new_root) / lib_folder / py_path
            full_py_path.parent.mkdir(parents=True, exist_ok=True)
            full_py_path.write_text(py_module)

    if cell_types_lists is not None:
        if cell_types_paths is None:
            cell_types_paths = [f"f{idx}" for idx in range(len(cell_types_lists))]
        else:
            if len(cell_types_paths) != len(cell_types_lists):
                raise ValueError(
                    "cell_types_paths must have same number of items as cell_types_lists"
                )

        for cell_types, cell_types_path in zip(cell_types_lists, cell_types_paths):
            full_cell_types_path = Path(new_root) / cell_types_folder / cell_types_path
            full_cell_types_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(cell_types, full_cell_types_path)

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
    nb_paths,
    expected_nbs=[nb1, nb2],
    current_root=current_root,
    new_root=new_root,
    nbs_folder=None,
    tmp_folder=None,
    lib_folder=None,
    clean=True,
)
