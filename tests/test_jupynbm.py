# %% imports
# standard library
from pathlib import Path
import shutil
import warnings
from unittest.mock import patch, MagicMock
from importlib import reload

# 3rd party
import pytest

# ours
import nbmodular.jupynbm as jnbm
import nbmodular.test_utils as tst
import nbmodular.export as xp
from test_data import nb

reload(tst)
reload(jnbm)


# %%
# test_rename_mfe_files_triggers_warning
src_mfe_files = [
    Path("/src/path/file1.ipynb"),
    Path("/src/path/subdir/file2.ipynb"),
]
src_path = "/src/path"
dst_path = "/dst/path"
alternative_suffix = ".py"

expected_dst_mfe_files = [
    Path("/dst/path/file1.ipynb"),
    Path("/dst/path/subdir/file2.ipynb"),
]
expected_dst_ae_files = [
    Path("/dst/path/file1.py"),
    Path("/dst/path/subdir/file2.py"),
]

dst_mfe_files, dst_ae_files = jnbm.srcpaths_in_dst(
    src_mfe_files, src_path, dst_path, alternative_suffix
)

assert dst_mfe_files == expected_dst_mfe_files
assert dst_ae_files == expected_dst_ae_files


# %%
# def test_rename_mfe_files_triggers_warning():
tmp_path = Path("tmp")
shutil.rmtree(tmp_path, ignore_errors=True)
src_path = tmp_path / "src"
dst_path = tmp_path / "dst"
src_path.mkdir(parents=True, exist_ok=True)
dst_path.mkdir(parents=True, exist_ok=True)

src_mfe_files = [
    src_path / "file1.ipynb",
    src_path / "subdir" / "file2.ipynb",
]
for file in src_mfe_files:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()

dst_mfe_files = [
    dst_path / "file1.ipynb",
    dst_path / "subdir" / "file2.ipynb",
]
dst_ae_files = [
    dst_path / "file1.py",
    dst_path / "subdir" / "file2.py",
]


# %%
# (cont.)
# def test_rename_mfe_files_triggers_warning():

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    jnbm.rename_mfe_files(src_mfe_files, dst_mfe_files, dst_ae_files)
    assert len(w) == 2
    assert issubclass(w[-1].category, UserWarning)
    assert "does not exist" in str(w[-1].message)

for dst_mfe_file in dst_mfe_files:
    assert dst_mfe_file.exists()

for src_mfe_file in src_mfe_files:
    assert not src_mfe_file.exists()


# %%
# (cont.)
# def test_rename_mfe_files_triggers_warning():
shutil.rmtree(tmp_path)


# %% test_rename_mfe_files2


@patch("pathlib.Path.rename")
@patch("pathlib.Path.exists", MagicMock(return_value=False))
@patch("pathlib.Path.mkdir")
def test_rename_mfe_files2(mock_mkdir, mock_rename):
    src_mfe_files = [
        Path("/src/path/file1.ipynb"),
        Path("/src/path/subdir/file2.ipynb"),
    ]
    dst_mfe_files = [
        Path("/dst/path/file1.ipynb"),
        Path("/dst/path/subdir/file2.ipynb"),
    ]
    dst_ae_files = [
        Path("/dst/path/file1.py"),
        Path("/dst/path/subdir/file2.py"),
    ]

    jnbm.rename_mfe_files(src_mfe_files, dst_mfe_files, dst_ae_files)

    assert mock_rename.call_count == 2
    assert mock_mkdir.call_count == 2
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)


# %%
# test_rename_mfe_files2()


# %%
# def test_rename_mfe_files ():
tmp_path = Path("tmp")
shutil.rmtree(tmp_path, ignore_errors=True)
src_path = tmp_path / "src"
dst_path = tmp_path / "dst"
src_path.mkdir(parents=True, exist_ok=True)
dst_path.mkdir(parents=True, exist_ok=True)

src_mfe_files = [
    src_path / "file1.ipynb",
    src_path / "subdir" / "file2.ipynb",
]
dst_ae_files = [
    dst_path / "file1.py",
    dst_path / "subdir" / "file2.py",
]
for src_mfe_file, dst_ae_file in zip(src_mfe_files, dst_ae_files):
    src_mfe_file.parent.mkdir(parents=True, exist_ok=True)
    dst_ae_file.parent.mkdir(parents=True, exist_ok=True)
    src_mfe_file.touch()
    dst_ae_file.touch()


dst_mfe_files = [
    dst_path / "file1.ipynb",
    dst_path / "subdir" / "file2.ipynb",
]

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    jnbm.rename_mfe_files(src_mfe_files, dst_mfe_files, dst_ae_files)
    assert len(w) == 0

shutil.rmtree(tmp_path)

# %%
# def test_migrate_files ():
tmp_path = Path("tmp")
shutil.rmtree(tmp_path, ignore_errors=True)
src_path = tmp_path / "src"
dst_path = tmp_path / "dst"
src_path.mkdir(parents=True, exist_ok=True)
dst_path.mkdir(parents=True, exist_ok=True)

src_mfe_files = [
    src_path / "file1.ipynb",
    src_path / "subdir" / "file2.ipynb",
]
for file in src_mfe_files:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()

jnbm.migrate_files(str(src_path), str(dst_path), ".ipynb", ".py")

dst_mfe_files = [
    dst_path / "file1.ipynb",
    dst_path / "subdir" / "file2.ipynb",
]
for dst_mfe_file in dst_mfe_files:
    assert dst_mfe_file.exists()

for src_mfe_file in src_mfe_files:
    assert not src_mfe_file.exists()

shutil.rmtree(tmp_path)


# %%
# def test_migrate_files_with_existing_files():
tmp_path = Path("tmp")
shutil.rmtree(tmp_path, ignore_errors=True)
src_path = tmp_path / "src"
dst_path = tmp_path / "dst"
src_path.mkdir(parents=True, exist_ok=True)
dst_path.mkdir(parents=True, exist_ok=True)

src_mfe_files = [
    src_path / "file1.ipynb",
    src_path / "subdir" / "file2.ipynb",
]
for file in src_mfe_files:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()

dst_mfe_files = [
    dst_path / "file1.ipynb",
    dst_path / "subdir" / "file2.ipynb",
]
for file in dst_mfe_files:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()

with pytest.raises(FileExistsError):
    jnbm.migrate_files(str(src_path), str(dst_path), ".ipynb", ".py")

shutil.rmtree(tmp_path)

# %%
reload(tst)
new_root = "test_jupynbm"
nb_folder = "nbm"
nb_paths = ["first_folder/first.ipynb", "second_folder/second.ipynb"]
current_root, nb_paths = tst.create_test_content(
    nbs=[tst.nb1, tst.nb2],
    nb_paths=nb_paths,
    nb_folder=nb_folder,
    new_root=new_root,
)


# %%
jupytext_path = "nbm_py"
nbm_path = nb_folder
jnbm.migrate_files(nbm_path, jupytext_path, ".ipynb", ".py")


# %%
reload(jnbm)
jnbm.update_jupytext_notebooks(jupytext_path, extension=".ipynb")


# %%
jnbm.migrate_files(jupytext_path, nbm_path, ".ipynb", ".py")


# %%
nbs, existing_nb_paths, py_modules, existing_py_paths, cell_types_lists = (
    tst.read_content_in_repo(
        nb_paths=nb_paths,
        tmp_folder=None,
        nbs_folder=None,
        lib_folder="nbm_py",
        cell_types_folder=None,
        use_config_paths=False,
    )
)


# %%
tst.check_py_modules(new_root=".", nb_paths=nb_paths, expected=[tst.jupy1, tst.jupy2])


# %%
reload(xp)
# xp.nbm_export_all_paths(nbm_path)
# xp.nbm_export_all_paths(nbm_path, from_notebook=True)
xp.nbm_export_all_paths(nbm_path, restrict_inputs=True)


# %%
# %%
nb_paths = ["first_folder/first.ipynb", "second_folder/second.ipynb"]
tst.check_test_repo_content(
    # nb_paths,
    nb_paths=nb_paths,
    expected_nbs=tst.nbs_after_jupynbm,
    expected_py_modules=tst.py_modules_after_jupynbm,
    current_root=current_root,
    new_root=new_root,
    clean=True,
    keep_cwd=False,
)


# %%
tst.check_py_modules(new_root=".", nb_paths=nb_paths, expected=[tst.jupy1, tst.jupy2])


# %% [markdown]
# #### Example usage

# %%
# path_with_nb_folder = str(Path(os.getcwd()) / nb_folder)
# xp.parse_argv_and_run_nbm_export_all_paths(["--path", path_with_nb_folder])

# # %%
# reload(jnbm)


# jnbm.sync_nbm_and_jupytext(jupytext_path, nbm_path, extension=".ipynb")

# # %%


# %%
multiple_updated_py_modules = [
    x.replace("@%%", "%%") for x in tst.multiple_updated_py_modules
]
new_root = "test_parse_argv_and_run_nbm_update_all_paths"
nb_folder = "nbm"
lib_folder = "nbmodular"
cell_types_folder = ".nbmodular"
# Create notebook in "new repo", and cd to it
current_root, nb_paths = tst.create_test_content(
    nbs=tst.nbs_after_jupynbm,
    nb_paths=tst.nb_paths_after_jupynbm,
    nb_folder="",
    py_modules=tst.py_modules_after_jupynbm,
    py_paths=tst.py_paths_after_jupynbm,
    lib_folder="",
    cell_types_lists=tst.cell_types_lists_after_jupynbm,
    cell_types_paths=tst.cell_types_paths_after_jupynbm,  # type: ignore
    cell_types_folder="",
    new_root=new_root,
)
