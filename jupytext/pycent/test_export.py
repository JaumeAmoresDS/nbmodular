import pytest
from pathlib import Path
from nbmodular.sync.export import nbm_update, read_nb


def test_nbm_update(tmp_path):
    # Create a temporary notebook file
    nb_path = tmp_path / "test_nb.ipynb"
    nb_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": ["#|export\n", "def add(a, b):\n", "    return a + b\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    nb_path.write_text(nb_content)

    # Run nbm_update on the temporary notebook file
    nbm_update(nb_path)

    # Check that the notebook file has been updated correctly
    updated_nb = read_nb(nb_path)
    assert (
        updated_nb["cells"][0]["source"]
        == "#|export\ndef add(a, b):\n    return a + b\n"
    )
