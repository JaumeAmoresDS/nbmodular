# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/cell2func.ipynb.

# %% auto 0
__all__ = ['Cell2Func', 'load_ipython_extension', 'keep_variables']

# %% ../../nbs/cell2func.ipynb 2
from functools import reduce
from pathlib import Path
import sys
import ast
from IPython import get_ipython
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)
from sklearn.utils import Bunch
from fastcore.all import argnames

# %% ../../nbs/cell2func.ipynb 3
@magics_class
class Cell2Func(Magics):
    """
    Base magic class for converting cells to modular functions.
    """
    def __init__(self, shell):
        super().__init__(shell)
        self.function_info = Bunch()
        self.function_list = []
        
    @cell_magic
    def cell2file (self, folder, cell):
        
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / "module.py", "w") as file_handle:
            file_handle.write(cell)

        get_ipython().run_cell(cell)
    
    @cell_magic
    def function (self, func, cell):
        "Converts cell to function"
        
        # parameters (temporary)
        collect_variables_values = True
        make_function = True
        tab_size=4
        
        this_function = Bunch(idx=len(self.function_list))
        if func not in self.function_info:
            self.function_info[func] = this_function
            self.function_list.append (this_function)
            
        idx = this_function['idx']
        
        # get variables specific about this function
        if collect_variables_values:
            get_variables_before_code = f'\nkeep_variables ("{func}", "values_before", locals ())'
            get_ipython().run_cell(get_variables_before_code)
            
            get_variables_here_code = cell + f'\nkeep_variables ("{func}", "values_here", locals ())'
            get_ipython().run_cell(get_variables_here_code)
            values_before, values_here = this_function['values_before'], this_function['values_here']
            values_here = {k:values_here[k] for k in set(values_here).difference(values_before)}
            this_function['values_here'] = values_here
            print (values_here)
        
        root = ast.parse (cell)
        variables_here = {node.id for node in ast.walk(root) if isinstance(node, ast.Name) and not callable(eval(node.id))}
        print (variables_here)
        if idx > 0:
            variables_before = reduce (lambda x, y: x['variables_here'] | y['variables_here'], self.function_list[:idx])
        else:
            variables_before = []
        variables_here = sorted (variables_here.difference(variables_before))
        print (variables_here)
        this_function.update (variables_here=variables_here, variables_before=variables_before)
        
        if make_function:
            function_code = ''
            arguments = ', '.join (variables_before)
            return_values = ','.join (variables_here + variables_before)
            for line in cell.splitlines():
                function_code += f'{" " * tab_size}{line}\n'
            return_line = f'return {return_values}'
            function_code = f'def {func}({arguments}):\n' + function_code + f'{" " * tab_size}{return_line}\n'
            this_function['code'] = function_code
            get_ipython().run_cell(function_code)
            print (function_code)

# %% ../../nbs/cell2func.ipynb 4
def load_ipython_extension(ipython):
    """
    This module can be loaded via `%load_ext core.cell2func` or be configured to be autoloaded by IPython at startup time.
    """
    magics = Cell2Func(ipython)
    ipython.register_magics(magics)

# %% ../../nbs/cell2func.ipynb 5
def keep_variables (function, field, variable_values, self=None):
    """
    Store `variables` in dictionary entry `self.variables_field[function]`
    """
    frame_number = 1
    while not isinstance (self, Cell2Func):
        fr = sys._getframe(frame_number)
        args = argnames(fr, True)
        if len(args)>0:
            self = fr.f_locals[args[0]]
        frame_number += 1
    variable_values = {k: variable_values[k] for k in variable_values if not k.startswith ('_') and not callable(variable_values[k])}
    function_info = getattr(self, 'function_info')
    function_info[function][field]=variable_values
