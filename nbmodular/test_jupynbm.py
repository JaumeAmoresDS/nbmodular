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
from nbmodular.jupynbm import srcpaths_in_dst, rename_mfe_files, migrate_files
import nbmodular.test_utils as tst

reload(tst)


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

dst_mfe_files, dst_ae_files = srcpaths_in_dst(
    src_mfe_files, src_path, dst_path, alternative_suffix
)

assert dst_mfe_files == expected_dst_mfe_files
assert dst_ae_files == expected_dst_ae_files


# %%
# def test_rename_mfe_files_triggers_warning():
tmp_path = Path("tmp")
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
    rename_mfe_files(src_mfe_files, dst_mfe_files, dst_ae_files)
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

    rename_mfe_files(src_mfe_files, dst_mfe_files, dst_ae_files)

    assert mock_rename.call_count == 2
    assert mock_mkdir.call_count == 2
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)


# %%
test_rename_mfe_files2()


# %%
# def test_rename_mfe_files ():
tmp_path = Path("tmp")
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
    rename_mfe_files(src_mfe_files, dst_mfe_files, dst_ae_files)
    assert len(w) == 0

shutil.rmtree(tmp_path)

# %%
# def test_migrate_files ():
tmp_path = Path("tmp")
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

migrate_files(str(src_path), str(dst_path), ".ipynb", ".py")

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
    migrate_files(str(src_path), str(dst_path), ".ipynb", ".py")

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

# %% [markdown]
# #### Example usage

# %%
path_with_nb_folder = str(Path(os.getcwd()) / nb_folder)
parse_argv_and_run_nbm_export_all_paths(["--path", path_with_nb_folder])

# %%
