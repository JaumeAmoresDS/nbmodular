

# %% auto 0
__all__ = ['nb1', 'mixed_nb1', 'convert_nested_nb_cells_to_dicts', 'parse_nb_sections', 'text2nb', 'texts2nbs', 'nb2text',
           'nbs2text', 'printnb', 'strip_nb', 'check_test_repo_content', 'create_and_cd_to_new_root_folder',
           'create_test_content']

# %% ../../nbs/test_utils.ipynb 2
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

# %% ../../nbs/test_utils.ipynb 14
def parse_nb_sections(nb):
    # Define the regex pattern to match sections
    pattern = "\[(markdown|code)\](.*?)((?=\[markdown\])|(?=\[code\])|$)"

    # Find all matches using re.findall which returns a list of tuples
    matches = re.findall(pattern, nb, re.DOTALL)

    # Transform the matches to the required format
    result = [(match[0], match[1].strip()) for match in matches]

    return result

# %% ../../nbs/test_utils.ipynb 18
def text2nb(nb: str):
    cells = [
        mk_cell(text, cell_type=cell_type) for cell_type, text in parse_nb_sections(nb)
    ]
    return new_nb(cells)

# %% ../../nbs/test_utils.ipynb 24
def texts2nbs(nbs: List[str] | str) -> List[dict]:
    if not isinstance(nbs, list):
        nbs = [nbs]
    return [text2nb(nb) for nb in nbs]

# %% ../../nbs/test_utils.ipynb 26
def nb2text(nb: dict) -> str:
    return "\n\n".join(
        [f"[{cell['cell_type']}]\n{cell['source']}" for cell in nb["cells"]]
    )


def nbs2text(nbs: List[dict]) -> List[str]:
    return [nb2text(nb) for nb in (nbs if isinstance(nbs, list) else [nbs])]

# %% ../../nbs/test_utils.ipynb 32
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

# %% ../../nbs/test_utils.ipynb 38
def strip_nb(nb: str) -> str:
    return nb2text(text2nb(nb))

# %% ../../nbs/test_utils.ipynb 40
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

# %% ../../nbs/test_utils.ipynb 45
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

# %% ../../nbs/test_utils.ipynb 47
def create_test_content(
    nbs: List[str] | str,
    nb_paths=None,
    nb_folder="nbm",
    new_root="new_test",
    config_path="settings.ini",
) -> Tuple[str, List[str]]:
    # we start from the root folder of our repo
    cd_root()
    current_root = os.getcwd()

    # Convert input texts into corresponding dicts with notebook structure
    nbs = texts2nbs(nbs)

    # Generate list of nb_paths if None
    if nb_paths is None:
        nb_paths = [f"f{idx}" for idx in range(len(nbs))]
    for nb, nb_path in zip(nbs, nb_paths):
        full_nb_path = Path(new_root) / nb_folder / nb_path
        full_nb_path.parent.mkdir(parents=True, exist_ok=True)
        write_nb(nb, full_nb_path)

    # Copy settings.ini in new root folder, so that this file
    # can be read later on by our export / import functions.
    # Also, cd to new root folder.
    _ = create_and_cd_to_new_root_folder(new_root, config_path)

    return current_root, nb_paths
