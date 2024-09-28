# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:percent
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
# # function_io
#
# > I/O functions used for loading / saving results and local variables from function.

# %%
#| default_exp function_io

# %%
#| export
import pdb
import joblib
import os
import re
import argparse
import shlex
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
import sys
import ast
import logging
import warnings
import pandas as pd
import numpy as np
import json
from sklearn.utils import Bunch

from IPython import get_ipython
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
import ipynbname
from sklearn.utils import Bunch
from fastcore.all import argnames
import nbdev


# %% [markdown]
# ## df

# %%
#| export
def load_df (path, **kwargs):
    path=Path(path)
    path_without_extension=path.parent / path.name[:-len('.df')]
    if (path_without_extension.parent / f'{path_without_extension.name}.parquet').exists():
        df = pd.read_parquet (path_without_extension.parent / f'{path_without_extension.name}.parquet', **kwargs)
    elif (path_without_extension.parent / f'{path_without_extension.name}.pk').exists():
        df = pd.read_parquet (path_without_extension.parent / f'{path_without_extension.name}.pk', **kwargs)
    elif (path_without_extension.parent / f'{path_without_extension.name}.csv').exists():
        df = pd.read_parquet (path_without_extension.parent / f'{path_without_extension.name}.csv', **kwargs)
    else:
        raise RuntimeError (f'File {path} not found')
    return df


# %%
#| export
def save_df (df, path, **kwargs):
    path=Path(path)
    path.parent.mkdir (parents=True, exist_ok=True)
    name_without_extension = path.name[:-len('.df')]
    extension = ''
    try:
        df.to_parquet (path.parent / f'{name_without_extension}.parquet', **kwargs)
        extension='parquet'
    except:
        try:
            df.to_pickle (path.parent / f'{name_without_extension}.pk', **kwargs)
            extension='pickle'
        except:
            df.to_csv (path.parent / f'{name_without_extension}.csv', **kwargs)
            extension='csv'
    with open (path, 'wt') as f: 
        f.write (extension)


# %% [markdown]
# ## pickle

# %%
#| export
def load_pickle (path, **kwargs):
    return joblib.load (path, **kwargs)


# %%
#| export
def save_pickle (data, path, **kwargs):
    joblib.dump (data, path, **kwargs)


# %% [markdown]
# ## csv

# %%
#| export
def load_csv (path, **kwargs):
    return pd.read_csv (path, **kwargs)


# %%
#| export
def save_csv (df, path, **kwargs):
    df.to_csv (path, **kwargs)


# %% [markdown]
# ## parquet

# %%
#| export
def load_parquet (path, **kwargs):
    return pd.read_parquet (path, **kwargs)


# %%
#| export
def save_parquet (df, path, **kwargs):
    df.to_parquet (path, **kwargs)


# %% [markdown]
# ## structured

# %%
#| export
def _load_structured (structure, io_type_load_args={}):
    if structure['single']:
        io_type = structure['io_type']
        load_args={} if io_type not in io_type_load_args else io_type_load_args[io_type]
        data = load (
            structure['path'], 
            io_type=io_type,
            **load_args,
        )
    elif structure['list_like']:
        data = []
        for i, element_structure in enumerate (structure['structure']):
            element = _load_structured (element_structure, io_type_load_args)
            data.append (element)
        if structure['io_type'] != 'list':
            type_data = eval(structure['io_type'])
            try:
                data = type_data (data)
            except:
                pass
    elif structure['dict_like']:
        data = {}
        for k, element_structure in structure['structure'].items():
            element = _load_structured (element_structure, io_type_load_args)
            data[k] = element
        if structure['io_type'] != 'dict':
            if structure['io_type']=='Bunch':
                data = Bunch (**data)
            else:
                type_data = eval(structure['io_type'])
                try:
                    data = type_data (data)
                except:
                    pass
    return data

def load_structured (path, **kwargs):
    path = Path(path)
    with open (path / 'structure.json', 'rt') as f:
        structure = json.load (f)
    return _load_structured (structure)


# %%
#| export
def _is_inhomogeneous (data, max_length=100, types=[pd.DataFrame]):
    inhomogeneous=False
    try:
        _ = np.array (data, float)
    except:
        try:
            _ = np.array (data, str)
        except:
            inhomogeneous=True
    if not inhomogeneous and len(data)<max_length:
        for t in types:
            if all (map(lambda x: isinstance(x, t), data)):
                return True
    return inhomogeneous

def _save_structured (
    data,
    path,
    io_types={pd.DataFrame: {'io_type': 'df', 'save_args': {}}},
    list_like_types=[list, tuple, np.ndarray],
    dict_like_types=[dict]
):
    for type_element in io_types:
        if isinstance (data, type_element):
            io_type = io_types[type_element]['io_type']
            save_args = io_types[type_element]['save_args']
            path = path / f'data.{io_type}'
            save (data, path, io_type=io_type, **save_args)
            structure = {'io_type': io_type, 'single': True, 'path': str(path), 'structure': None, 'list_like': False, 'dict_like': False}
            return structure
        
    if any (map (lambda t: isinstance (data, t), list_like_types)):
        if _is_inhomogeneous (data):
            structure = {
                'io_type': type (data).__name__,
                'single': False,
                'path': str(path),
                'structure': [],
                'list_like': True, 
                'dict_like': False,
            }
            for i, element in enumerate (data):
                element_structure = _save_structured (
                    element, 
                    path / str(i), 
                    io_types=io_types,
                    list_like_types=list_like_types,
                    dict_like_types=dict_like_types,
                )
                structure['structure'].append (element_structure)
        else:
            path = path / 'data.pickle'
            save (data, path, io_type='pickle')
            structure = {'io_type': 'pickle', 'single': True, 'path': str(path), 'structure': None}
        
    elif any (map (lambda t: isinstance (data, t), dict_like_types)):
        structure = {
            'io_type': type (data).__name__,
            'single': False,
            'path': str(path),
            'structure': {},
            'list_like': False, 
            'dict_like': True,
        }
        for k, element in data.items():
            element_structure = _save_structured (
                element, 
                path / k, 
                io_types=io_types,
                list_like_types=list_like_types,
                dict_like_types=dict_like_types,
            )
            structure['structure'][k] = element_structure
    else:
        raise ValueError (f'Invalid data type: {type(data)}')
    
    return structure

def save_structured (
    data,
    path,
    io_types={pd.DataFrame: {'io_type': 'df', 'save_args': {}}},
    list_like_types=[list, tuple, np.ndarray],
    dict_like_types=[dict],
    make_relative=True,
    **kwargs,
):
    if make_relative:
        current_dir = str (Path('.').absolute())
        path = str(path).replace (current_dir, '')
        if len(path) > 0 and path[0]=='/':
            path = path[1:]
    path = Path (path)
    structure = _save_structured (
        data,
        path,
        io_types=io_types,
        list_like_types=list_like_types,
        dict_like_types=dict_like_types,
    )
    with open (path / 'structure.json', 'wt') as f:
        json.dump (structure, f, indent=4)


# %% [markdown]
# ## load

# %%
#| export
def load (
    path_variables,
    io_type='pickle',
    **kwargs,
):
    load_function = eval (f'load_{io_type}')
    return load_function (path_variables, **kwargs)


# %% [markdown]
# ## save

# %%
#| export
def save (
    data,
    path_variables,
    io_type='pickle',
    **kwargs,
):
    Path(path_variables).parent.mkdir (parents=True, exist_ok=True)
    save_function = eval (f'save_{io_type}')
    save_function (data, path_variables, **kwargs)


# %% [markdown]
# ## Example usage structured data

# %%
data = [{
        'numbers' :[1,2,3],
        'table': pd.DataFrame ([[1,2],[3,4]], columns=['a','b'], index=['this','that']),
        'vector': np.array ([10,20,30]),
    },
    pd.DataFrame ([[11,21,31],[31,41,51]], columns=['c','d','e'], index=['i0','i1']),
    [1000,2000,3000],
    pd.DataFrame ([[101,201],[301,401]], columns=['e','f']),
]
save_structured (data, 'test_structured')
assert sorted(os.listdir('test_structured'))==['0', '1', '2', '3', 'structure.json']
assert sorted(os.listdir('test_structured/0'))==sorted(['table', 'vector', 'numbers'])
assert sorted(os.listdir('test_structured/0/table'))==sorted(['data.parquet', 'data.df'])
assert os.listdir('test_structured/0/vector')==['data.pickle']
assert os.listdir('test_structured/0/numbers')==['data.pickle']
assert sorted(os.listdir('test_structured/1')) == sorted(['data.parquet', 'data.df'])
assert os.listdir('test_structured/2')==['data.pickle']
assert sorted(os.listdir('test_structured/3'))==sorted(['data.parquet', 'data.df'])
data2 = load_structured ('test_structured')
assert data2[0]['numbers']==data[0]['numbers']
pd.testing.assert_frame_equal (data2[0]['table'], data[0]['table'])
np.testing.assert_array_equal (data2[0]['vector'], data[0]['vector'])
pd.testing.assert_frame_equal (data2[1], data[1])
assert data2[2]==data[2]
pd.testing.assert_frame_equal (data2[3], data[3])
