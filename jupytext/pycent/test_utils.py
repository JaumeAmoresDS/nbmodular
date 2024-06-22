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
from token import OP
from typing import List, Tuple, Optional
import re

# 3rd party
from execnb.nbio import new_nb, write_nb, mk_cell, read_nb
from plum import Val

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
# ### read_nbs_in_repo


# %%
# | export
def read_nbs_in_repo(
    nb_paths: List[str],  # type: ignore
    new_root: str = "new_test",
    nbm_folder: str = "nbm",
    tmp_folder: str = ".nbs",
    nbs_folder: str = "nbs",
    print_as_list: bool = False,
    print: bool = False,
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
    content = read_nbs(nb_paths)
    if print:
        print_files(content, print_as_list=print_as_list, paths=nb_paths)
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

    Returns:
    -------
    content : str
        Content of the Python modules.

    """
    py_paths = derive_py_paths(nb_paths, new_root, lib_folder=lib_folder)
    content = read_text_files(py_paths)
    if print:
        print_files(content, print_as_list=print_as_list, paths=py_paths)
    return content


# %% [markdown]
# ### read_content_in_repo


# %%
# | export
def read_content_in_repo(
    nb_paths: List[str],
    new_root: Union[str, Path],
    nbm_folder: str = "nbm",
    tmp_folder: str = ".nbs",
    nbs_folder: str = "nbs",
    lib_folder: str = "nbmodular",
    print_as_list: bool = False,
    print: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Read the content in a repository.

    Parameters:
    ----------
    nb_paths: List[str]
        List of notebook paths.
    new_root: Union[str, Path]
        New root directory.
    nbm_folder: str, optional
        Folder name for nbm files. Defaults to "nbm".
    tmp_folder: str, optional
        Temporary folder name. Defaults to ".nbs".
    nbs_folder: str, optional
        Folder name for nbs files. Defaults to "nbs".
    lib_folder: str, optional
        Folder name for nbmodular files. Defaults to "nbmodular".
    print_as_list: bool, optional
        Whether to print the content as a list. Defaults to False.
    print: bool, optional
        Whether to print the content. Defaults to True.

    Returns:
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists - the nbs content and the py_modules content.
    """

    nbs = read_nbs_in_repo(
        nb_paths, new_root, nbm_folder, tmp_folder, nbs_folder, print_as_list, print
    )
    py_modules = read_pymodules_in_repo(
        nb_paths, new_root, lib_folder, print_as_list, print
    )
    return nbs, py_modules


# %% [markdown]
# ### check_nbs


# %% [markdown]
# ### derive_nb_paths


# %%
# | export
def derive_nb_paths(
    nb_paths: List[str],
    new_root: str | Path,
    nbm_folder: str = "nbm",
    tmp_folder: str = ".nbs",
    nbs_folder: str = "nbs",
):
    all_nb_paths = []
    for nb_path in nb_paths:
        all_nb_paths.append(Path(new_root) / nbm_folder / nb_path)
        all_nb_paths.append(Path(new_root) / nbs_folder / nb_path)
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
# ### derive_all_paths


# %%
# | export
def derive_all_paths(
    nb_paths: List[str],
    new_root: str | Path,
    nbm_folder: str = "nbm",
    tmp_folder: str = ".nbs",
    nbs_folder: str = "nbs",
    lib_folder: str = "nbmodular",
):
    all_nb_paths = derive_nb_paths(
        nb_paths,
        new_root,
        nbm_folder=nbm_folder,
        tmp_folder=tmp_folder,
        nbs_folder=nbs_folder,
    )
    py_paths = derive_py_paths(nb_paths, new_root, lib_folder=lib_folder)
    return all_nb_paths, py_paths


# %% [markdown]
# #### Example usage

# %%
nb_paths, py_paths = derive_all_paths(
    nb_paths=["folder_A/nb_A.ipynb", "folder_B/nb_B.ipynb"], new_root="tmp_repo"
)
assert nb_paths == [
    Path("tmp_repo/nbm/folder_A/nb_A.ipynb"),
    Path("tmp_repo/nbs/folder_A/nb_A.ipynb"),
    Path("tmp_repo/.nbs/folder_A/nb_A.ipynb"),
    Path("tmp_repo/.nbs/folder_A/test_nb_A.ipynb"),
    Path("tmp_repo/nbm/folder_B/nb_B.ipynb"),
    Path("tmp_repo/nbs/folder_B/nb_B.ipynb"),
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
def read_nbs(paths: List[str], as_text: bool = True) -> List[str] | List[dict]:
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
    for path in paths:
        # Check that file exists. useful for being called inside a test utility
        # to see where it fails.
        assert os.path.exists(path)
        nbs_in_disk.append(read_nb(path))

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
# ### read_pymodules


# %%
# | export
def read_text_files(paths: List[str]) -> List[str]:
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
    for path in paths:
        # Check that file exists. useful for being called inside a test utility
        # to see where it fails.
        assert os.path.exists(path)
        with open(path, "rt") as file:
            text_files.append(file.read())
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
) -> None:
    if print_as_list:
        print(f"[")
    for idx, file in enumerate(files):
        if not print_as_list:
            print(f"{'-'*50}")
        if paths is not None:
            print(paths[idx])
        print('"""')
        print(file)
        print('"""', end="")
        if print_as_list and idx < (len(files) - 1):
            print(",")
        else:
            print()
    if print_as_list:
        print("]")


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
    nbm_folder: str = "nbm",
    tmp_folder: str = ".nbs",
    nbs_folder: str = "nbs",
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

    Raises
    ------
    AssertionError
        If the actual Python modules do not match the expected modules.
    """
    actual = read_pymodules_in_repo(nb_paths, new_root, lib_folder=lib_folder)
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
    nbm_folder: str = "nbm",
    tmp_folder: str = ".nbs",
    nbs_folder: str = "nbs",
    lib_folder: str = "nbmodular",
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
    nbm_folder : str, optional
        The name of the folder containing the notebook modules, by default "nbm".
    tmp_folder : str, optional
        The name of the temporary folder, by default ".nbs".
    nbs_folder : str, optional
        The name of the folder containing the notebooks, by default "nbs".
    lib_folder : str, optional
        The name of the folder containing the Python modules, by default "nbmodular".
    clean : bool, optional
        Whether to clean the new root directory after checking, by default False.
    keep_cwd : bool, optional
        Whether to keep the current working directory after checking, by default False.

    Raises
    ------
    ValueError
        If a validation error occurs during the checking process.
    """
    if current_root is not None:
        assert Path(current_root).name == "nbmodular"
        new_wd = os.getcwd()

        assert Path(new_wd).resolve() == Path(f"{current_root}/{new_root}").resolve()
        os.chdir(current_root)
    if new_root is not None:
        assert (Path(new_root) / "settings.ini").exists()
        use_new_root = True
    else:
        new_root = "./"
        use_new_root = False

    if expected_nbs is not None:
        check_nbs(nb_paths, expected_nbs, new_root, nbm_folder, tmp_folder, nbs_folder)
    if expected_py_modules is not None:
        check_py_modules(nb_paths, expected_py_modules, new_root, lib_folder)
    if clean and use_new_root:
        shutil.rmtree(new_root)
    if keep_cwd and use_new_root:
        if clean:
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
    nb_paths,
    expected_nbs=[nb1, nb2],
    current_root=current_root,
    new_root=new_root,
    clean=True,
)
