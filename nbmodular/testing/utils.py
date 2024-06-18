

# %% auto 0
__all__ = ['nb1', 'mixed_nb1', 'parse_nb_sections', 'create_notebook', 'nb_to_text', 'printnb', 'create_notebooks',
           'create_and_cd_to_new_root_folder', 'create_test_content']

# %% ../../nbs/test_utils.ipynb 2
# standard
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import re

# 3rd party
from execnb.nbio import new_nb, write_nb, mk_cell

# %% ../../nbs/test_utils.ipynb 5
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
def parse_nb_sections(nb):
    # Define the regex pattern to match sections
    pattern = "\[(markdown|code)\](.*?)((?=\[markdown\])|(?=\[code\])|$)"

    # Find all matches using re.findall which returns a list of tuples
    matches = re.findall(pattern, nb, re.DOTALL)

    # Transform the matches to the required format
    result = [(match[0], match[1].strip()) for match in matches]

    return result

# %% ../../nbs/test_utils.ipynb 16
def create_notebook(nb: str):
    cells = [
        mk_cell(text, cell_type=cell_type) for cell_type, text in parse_nb_sections(nb)
    ]
    return new_nb(cells)

# %% ../../nbs/test_utils.ipynb 20
def nb_to_text(nb: dict) -> str:
    return "\n\n".join(
        [f"[{cell['cell_type']}]\n{cell['source']}" for cell in nb["cells"]]
    )

# %% ../../nbs/test_utils.ipynb 23
def printnb(nb_text: str, no_newlines: bool = False) -> None:
    print(f'''"""{nb_text}"""''' if no_newlines else f'''"""\n{nb_text}\n"""''')

# %% ../../nbs/test_utils.ipynb 26
def create_notebooks(nbs: List[str] | str) -> List[dict]:
    if not isinstance(nbs, list):
        nbs = [nbs]
    return [create_notebook(nb) for nb in nbs]

# %% ../../nbs/test_utils.ipynb 28
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

# %% ../../nbs/test_utils.ipynb 30
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
