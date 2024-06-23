

# %% auto 0
__all__ = ['nb1', 'mixed_nb1', 'py1', 'convert_nested_nb_cells_to_dicts', 'parse_nb_sections', 'text2nb', 'texts2nbs', 'nb2text',
           'nbs2text', 'printnb', 'strip_nb', 'read_nbs_in_repo', 'read_pymodules_in_repo', 'read_content_in_repo',
           'derive_nb_paths', 'derive_py_paths', 'derive_all_paths', 'read_nbs', 'compare_nb', 'compare_nbs',
           'read_text_files', 'write_text_files', 'compare_texts', 'print_files', 'read_and_print', 'check_nbs',
           'check_py_modules', 'check_test_repo_content', 'create_and_cd_to_new_root_folder', 'create_test_content']

# %% ../../nbs/test_utils.ipynb 2
# standard
from code import interact
import logging
from math import log
import os
import shutil
from pathlib import Path
from token import OP
from typing import List, Tuple, Optional, Union
import re

# 3rd party
from execnb.nbio import new_nb, write_nb, mk_cell, read_nb
from plum import Val
from requests import post

# ours
from ..core.utils import cd_root

# %% ../../nbs/test_utils.ipynb 5
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

# %% ../../nbs/test_utils.ipynb 9
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

# %% ../../nbs/test_utils.ipynb 12
py1 = """
def hello ():
    print ('hello')

def one_plus_one ():
    a=1+1
    print (a)
"""

# %% ../../nbs/test_utils.ipynb 17
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

# %% ../../nbs/test_utils.ipynb 19
def parse_nb_sections(nb):
    # Define the regex pattern to match sections
    pattern = "\[(markdown|code)\](.*?)((?=\[markdown\])|(?=\[code\])|$)"

    # Find all matches using re.findall which returns a list of tuples
    matches = re.findall(pattern, nb, re.DOTALL)

    # Transform the matches to the required format
    result = [(match[0], match[1].strip()) for match in matches]

    return result

# %% ../../nbs/test_utils.ipynb 23
def text2nb(nb: str):
    cells = [
        mk_cell(text, cell_type=cell_type) for cell_type, text in parse_nb_sections(nb)
    ]
    return new_nb(cells)

# %% ../../nbs/test_utils.ipynb 29
def texts2nbs(nbs: List[str] | str) -> List[dict]:
    if not isinstance(nbs, list):
        nbs = [nbs]
    return [text2nb(nb) for nb in nbs]

# %% ../../nbs/test_utils.ipynb 31
def nb2text(nb: dict) -> str:
    return "\n\n".join(
        [f"[{cell['cell_type']}]\n{cell['source']}" for cell in nb["cells"]]
    )


def nbs2text(nbs: List[dict]) -> List[str]:
    return [nb2text(nb) for nb in (nbs if isinstance(nbs, list) else [nbs])]

# %% ../../nbs/test_utils.ipynb 37
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

# %% ../../nbs/test_utils.ipynb 43
def strip_nb(nb: str) -> str:
    return nb2text(text2nb(nb))

# %% ../../nbs/test_utils.ipynb 45
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

# %% ../../nbs/test_utils.ipynb 47
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

# %% ../../nbs/test_utils.ipynb 49
def read_content_in_repo(
    nb_paths: List[str],
    new_root: Union[str, Path],
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
    lib_folder: Optional[str] = "nbmodular",
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
    return nbs, py_modules

# %% ../../nbs/test_utils.ipynb 52
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
            all_nb_paths.append(Path(new_root) / nbs_folder / nb_path)
        if tmp_folder is not None:
            tmp_nb = Path(new_root) / tmp_folder / nb_path
            all_nb_paths.append(tmp_nb)
            tmp_test_nb = tmp_nb.parent / f"test_{tmp_nb.name}"
            all_nb_paths.append(tmp_test_nb)

    return all_nb_paths

# %% ../../nbs/test_utils.ipynb 54
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

# %% ../../nbs/test_utils.ipynb 56
def derive_all_paths(
    nb_paths: List[str],
    new_root: str | Path,
    nbm_folder: Optional[str] = "nbm",
    tmp_folder: Optional[str] = ".nbs",
    nbs_folder: Optional[str] = "nbs",
    lib_folder: Optional[str] = "nbmodular",
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
    return all_nb_paths, py_paths

# %% ../../nbs/test_utils.ipynb 60
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

# %% ../../nbs/test_utils.ipynb 64
def compare_nb(nb1: str, nb2: str) -> bool:
    return strip_nb(nb1) == strip_nb(nb2)


def compare_nbs(nbs1: List[str], nbs2: List[str]) -> bool:
    return all(map(compare_nb, nbs1, nbs2))

# %% ../../nbs/test_utils.ipynb 68
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

# %% ../../nbs/test_utils.ipynb 70
def write_text_files(texts: List[str], paths: List[str]) -> None:
    for text, path in zip(texts, paths):
        with open(path, "wt") as file:
            file.write(text)

# %% ../../nbs/test_utils.ipynb 72
def compare_texts(texts1: List[str], texts2: List[str]) -> bool:
    return all(map(lambda x, y: x.strip() == y.strip(), texts1, texts2))

# %% ../../nbs/test_utils.ipynb 77
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

# %% ../../nbs/test_utils.ipynb 78
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

# %% ../../nbs/test_utils.ipynb 81
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

# %% ../../nbs/test_utils.ipynb 83
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
    actual = read_pymodules_in_repo(nb_paths, new_root, lib_folder=lib_folder, interactive_notebook=interactive_notebook)
    assert compare_texts(actual, expected)

# %% ../../nbs/test_utils.ipynb 85
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

# %% ../../nbs/test_utils.ipynb 90
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

# %% ../../nbs/test_utils.ipynb 92
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
