# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/cell2func.ipynb.

# %% auto 0
__all__ = ['FunctionProcessor', 'CellProcessor', 'CellProcessorMagic', 'load_ipython_extension', 'keep_variables']

# %% ../../nbs/cell2func.ipynb 2
import pdb
import os
import re
import argparse
import shlex
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
import sys
import ast
from IPython import get_ipython
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
import ipynbname
from sklearn.utils import Bunch
from fastcore.all import argnames
import nbdev

# %% ../../nbs/cell2func.ipynb 4
class FunctionProcessor (Bunch):
    """
    Function processor.
    """
    def to_file (self, file_path, mode='w'):
        with open (file_path, mode=mode) as file:
            file.write (self.code)
    
    def write (self, file):
        file.write (self.code)
        
    def print (self):
        print (self.code)
    
    def update_code (
        self, 
        arguments=None, 
        return_values=None,
        display=False
    ) -> None:
        if arguments is not None:
            self.arguments = arguments
        arguments = ', '.join (self.arguments)
        if return_values is not None:
            self.return_values = return_values
        return_values = ','.join (self.return_values)
        function_code = ''
        for line in self.original_code.splitlines():
            function_code += f'{" " * self.tab_size}{line}\n'
        if return_values != '':
            return_line = f'return {return_values}'
            return_line = f'{" " * self.tab_size}{return_line}\n'
        else:
            return_line = ''
        function_code = f'def {self.name}({arguments}):\n' + function_code + return_line
        self.code = function_code
        get_ipython().run_cell(function_code)
        if display:
            print (function_code)
    
    def get_ast(self, original=True, code=None):
        if code is None:
            code = self.original_code if original else self.code
        print(ast.dump(ast.parse(code), indent=2))
    
    def __str__ (self):
        name = None if not hasattr(self, 'name') else self.name
        return f'FunctionProcessor with name {name}, and fields: {self.keys()}\n    Arguments: {self.arguments}\n    Output: {self.return_values}\n    Variables: {self.values_here.keys()}'
    
    def __repr__ (self):
        return str(self)

# %% ../../nbs/cell2func.ipynb 6
class CellProcessor():
    """
    Processes the cell's code according to the magic command.
    """
    def __init__(self, tab_size=4, **kwargs):
        self.function_info = Bunch()
        self.current_function = Bunch()
        self.function_list = []
        self.tab_size=tab_size
        try:
            self.file_name = ipynbname.name().replace ('.ipynb', '.py')
            nb_path = ipynbname.path ()
            found_notebook = True
        except FileNotFoundError:
            self.file_name = 'temporary.py'
            nb_path = Path ('.').absolute()
            found_notebook = False
        self.nbs_folder = self.get_nbs_path ()
        self.lib_folder = self.get_lib_path ()
        
        if found_notebook:
            index = nb_path.parts.index(self.nbs_folder.name)
            self.file_path = (self.nbs_folder.parent / self.lib_folder.name).joinpath (*nb_path.parts[index+1:])
        else:
            self.file_path = nb_path / self.file_name
            
        self.call_history = []
        
        self.parser = argparse.ArgumentParser(description='Process some integers.')
        self.parser.add_argument('-i', '--input', type=str, nargs='+', help='input')
        self.parser.add_argument('-o', '--output', type=str, nargs='+', help='output')
        
    def reset (self):
        values_to_remove = [x for function in self.function_list for x in function.values_here.keys()]
        remove_variables_code = '\n'.join([f'''
            try:
                exec("del {x}")
            except:
                print (f'could not remove {x}')
                ''' for x in values_to_remove])
        get_ipython().run_cell(remove_variables_code)
        self.function_list = []
        self.function_info = Bunch()
    
    def process_function_call (self, line, cell, add_call=True):
        call = (line, cell)
        if add_call:
            self.add_call (call)
        function_name, signature = self.parse_signature (line)
        self.function (function_name, cell, call=call, **signature)

    def add_call (self, call):
        self.call_history.append (call)
        
    def cell2file (self, folder, cell):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / "module.py", "w") as file_handle:
            file_handle.write(cell)

        get_ipython().run_cell(cell)
                    
    def create_function (
        self,
        func, 
        cell,
        call=None,
        input=None,
        unknown_input=True,
        output=None,
        unknown_output=True,
        collect_variables_values=True,
        make_function=True,
        update_previous_functions=True,
        show=False,
        register_pipeline=True,
        pipeline_name=None
    ) -> FunctionProcessor:
        
        this_function = FunctionProcessor (
            idx=len(self.function_list), 
            original_code=cell, 
            name=func, 
            values_before=[],
            tab_size=self.tab_size,
            call=call
        )
        self.current_function = this_function
            
        idx = this_function.idx
        
        # get variables specific about this function
        if collect_variables_values:
            get_previous_variables_code = f'from nbmodular.core.cell2func import keep_variables\nkeep_variables ("{func}", "values_before", locals ())'
            get_ipython().run_cell(get_previous_variables_code)
            
            get_new_variables_code = cell + f'\nfrom nbmodular.core.cell2func import keep_variables\nkeep_variables ("{func}", "values_here", locals ())'
            get_ipython().run_cell(get_new_variables_code)
            values_before, values_here = this_function['values_before'], this_function['values_here']
            values_here = {k:values_here[k] for k in set(values_here).difference(values_before)}
            this_function['values_here'] = values_here
            # print (values_here)
        
        root = ast.parse (cell)
        variables_in_function = this_function['values_before'] | this_function['values_here']
        # we shouldn't need to check if node.id is callable, there is surely an attribute that indicates that in the AST!
        new_variables = {node.id for node in ast.walk(root) if isinstance(node, ast.Name) and node.id in variables_in_function and not callable(variables_in_function[node.id])}
        this_function.variables_here = new_variables
        # print (new_variables)
        if idx > 0:
            previous_variables = []
            for x in self.function_list[:idx]: 
                previous_variables += x['new_variables']
            #previous_variables = reduce (lambda x, y: x['new_variables'] + y['new_variables'], self.function_list[:idx])
        else:
            previous_variables = []
        new_variables = sorted (new_variables.difference(previous_variables))
        # print (new_variables)
        this_function.update (new_variables=new_variables, previous_variables=previous_variables, posterior_variables=[])
        
        if make_function:
            this_function.update_code ( 
                arguments=[x for x in previous_variables if x in this_function.variables_here] if unknown_input else input, 
                #arguments=previous_variables if unknown_input else input, 
                return_values=[] if unknown_output else output,
                display=show
            )
            
        # add variables from current function to posterior_variables of all the previous functions
        for function in self.function_list[:idx]:
            function.posterior_variables += [v for v in this_function.previous_variables+this_function.new_variables if v not in function.posterior_variables]
            if update_previous_functions and unknown_output:
                function.update_code (
                    return_values=[x for x in function.previous_variables+function.new_variables if x in function.posterior_variables and x in function.variables_here], 
                    #return_values=[x for x in function.previous_variables+function.new_variables if x in function.posterior_variables], 
                    display=False
                )
                
        if register_pipeline:
            self.register_pipeline (pipeline_name=pipeline_name)
        
        return this_function
    
    def function (
        self,
        func,
        cell,
        merge=False,
        **kwargs
    ) -> None:
        
        this_function = self.create_function (func, cell, **kwargs)
        if func in self.function_info and merge:
            new_function = self.merge_function (self.function_info[func], this_function)
            self.function_list.remove (this_function)
            this_function = new_function
        else:
            self.function_info[func] = this_function
            self.function_list.append (this_function)
                
    def parse_signature (self, line):
        argv = shlex.split(line, posix=(os.name == 'posix'))
        
        function_name=argv[0]
        signature = dict(
            input=None,
            unknown_input=True,
            output=None,
            unknown_output=True
        )
        found_io = False
        for idx, arg in enumerate(argv[1:], 1):
            if arg and arg.startswith('-') and arg != '-' and arg != '->':
                found_io = True
                break
        if found_io:
            pars = self.parser.parse_args(argv[idx:])
            unknown_input = 'input' not in pars
            if not unknown_input:
                signature.update (input=() if pars.input==['None'] else pars.input, unknown_input=pars.input is None)
            unknown_output = 'output' not in pars
            if not unknown_output:
                signature.update (output=() if pars.output==['None'] else pars.output, unknown_output=pars.output is None)
            
        # print (function_name, signature)
        return function_name, signature
    
    def write (self):
        with open (str(self.file_path), 'w') as file:
            for function in self.function_list:
                function.write (file)
                
    def print (self, function_name):
        if function_name == 'all':
            for function in self.function_list:
                function.print ()
        else:
            self.function_info[function_name].print ()
            
    def get_lib_path (self):
        return nbdev.config.get_config()['lib_path']
                   
    def get_nbs_path (self):
        return nbdev.config.get_config()['nbs_path']
    
    def pipeline_code (self, pipeline_name=None):
        pipeline_name = f'{self.file_name}_pipeline' if pipeline_name is None else pipeline_name
        code = f"def {pipeline_name} ():\n"
        for func in self.function_list:
            argument_list_str = ", ".join(func.arguments)
            return_list_str = f'{", ".join(func.return_values)} = ' if len(func.return_values)>0 else ''
            code += f'{" " * self.tab_size}' + f'{return_list_str}{func.name} ({argument_list_str})\n'
        return code, pipeline_name
    
    def register_pipeline (self, pipeline_name=None):
        code, name = self.pipeline_code (pipeline_name=pipeline_name)
        self.pipeline = FunctionProcessor (code=code,
                                           arguments=[],
                                           return_values=[],
                                           name=name)
        get_ipython().run_cell(code)
    
    def print_pipeline (self):
        code, name = self.pipeline_code()  
        print (code)

# %% ../../nbs/cell2func.ipynb 8
@magics_class
class CellProcessorMagic (Magics):
    """
    Base magic class for converting cells to modular functions.
    """
    def __init__(self, shell, **kwargs):
        super().__init__(shell)
        self.processor = CellProcessor (magic=self, **kwargs)
        
    @cell_magic
    def cell2file (self, folder, cell):
        self.processor.cell2file (folder, cell)
    
    @cell_magic
    def function (self, line, cell):
        "Converts cell to function"
        self.processor.process_function_call (line, cell)
    
    @line_magic
    def write (self, line):
        return self.processor.write ()
    
    @line_magic
    def print (self, line):
        return self.processor.print (line)
    
    @line_magic
    def function_info (self, function_name):
        return self.processor.function_info [function_name]
        
    @line_magic
    def cell_processor (self, line):
        return self.processor
        
    @line_magic
    def pipeline_code (self, line):
        return self.processor.pipeline_code ()
    
    @line_magic
    def print_pipeline (self, line):
        return self.processor.print_pipeline ()
          
    @line_magic
    def match (self, line):
        p0 = '[a-zA-Z]\S*\s*\\([^-()]*\\)\s*->\s*\\([^-()]*\\)'
        p = '\\([^-()]*\\)'
        m = re.search (p0, line)
        if m is not None:
            inp, out = re.findall (p, line)
            #print (inp)
            #print (out)

# %% ../../nbs/cell2func.ipynb 10
def load_ipython_extension(ipython):
    """
    This module can be loaded via `%load_ext core.cell2func` or be configured to be autoloaded by IPython at startup time.
    """
    magics = CellProcessorMagic(ipython)
    ipython.register_magics(magics)

# %% ../../nbs/cell2func.ipynb 11
load_ipython_extension (get_ipython())

# %% ../../nbs/cell2func.ipynb 13
def keep_variables (function, field, variable_values, self=None):
    """
    Store `variables` in dictionary entry `self.variables_field[function]`
    """
    frame_number = 1
    while not isinstance (self, CellProcessor):
        fr = sys._getframe(frame_number)
        args = argnames(fr, True)
        if len(args)>0:
            self = fr.f_locals[args[0]]
        frame_number += 1
    variable_values = {k: variable_values[k] for k in variable_values if not k.startswith ('_') and not callable(variable_values[k])}
    current_function = getattr(self, 'current_function')
    current_function[field]=variable_values
