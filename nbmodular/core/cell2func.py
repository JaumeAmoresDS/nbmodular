# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/cell2func.ipynb.

# %% auto 0
__all__ = ['FunctionVisitor', 'ReturnVisitor', 'FunctionProcessor', 'CellProcessor', 'CellProcessorMagic',
           'load_ipython_extension', 'keep_variables']

# %% ../../nbs/cell2func.ipynb 2
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
from IPython import get_ipython
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
import ipynbname
from sklearn.utils import Bunch
from fastcore.all import argnames
import nbdev

# %% ../../nbs/cell2func.ipynb 4
class FunctionVisitor (ast.NodeVisitor):
    def visit_FunctionDef (self, node):
        self.arguments = [x.arg for x in node.args.args]

class ReturnVisitor (ast.NodeVisitor):
    def visit_Return (self, node):
        return_values = [x.id for x in node.value.elts]
        if hasattr (self, 'return_values'):
            self.return_values += [x for x in return_values if x not in self.return_values]
        else:
            self.return_values = return_values

# %% ../../nbs/cell2func.ipynb 11
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
        if self.permanent:
            get_ipython().run_cell(self.code)
            return
        if self.data:
            arguments = []
            arguments = 'test=False'
        else:
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
        function_calls = '' if 'function_calls' not in self else self.function_calls
        function_code = f'def {self.name}({arguments}):\n' + function_calls + function_code + return_line
        self.code = function_code
        get_ipython().run_cell(function_code)
        if display:
            print (function_code)
    
    def get_ast(self, original=True, code=None):
        if code is None:
            code = self.original_code if original else self.code
        print(ast.dump(ast.parse(code), indent=2))
        
    def parse_variables (self, code=None):
        if code is None: code=self.original_code
        # variable parsing
        root = ast.parse (code)
        # newly created names: candidates for return list and not for argument list
        self.created_variables = list({node.id for node in ast.walk(root) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)})
        # names defined before: candidates for arguments list, if they are not callable
        self.loaded_names = list({node.id for node in ast.walk(root) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)})
        self.previous_variables = [x for x in self.loaded_names if x not in self.created_variables]
        
        # names that appear as arguments in functions -> some defined created the current function, some in the current one
        v=[node for node in ast.walk(root) if isinstance(node, ast.Call)]
        self.argument_variables = [y.id  for x in v for y in x.args if isinstance(y, ast.Name)]
        # argument variables might still be modified in the function, so they need to be marked as I/O, i.e., candidates for return list and for argument list
        
        # loaded names that are not arguments and not created in the current function are most probably read-only, i.e., not candidates for return list
        self.read_only_variables = [x for x in self.previous_variables if x not in self.argument_variables]
        self.posterior_variables = []
        self.all_variables = self.created_variables.copy()
        self.all_variables += [k for k in self.previous_variables if k not in self.all_variables]
        self.all_variables += [k for k in self.argument_variables if k not in self.all_variables]
        
    def parse_arguments_and_results (self, root):
        function_visitor = FunctionVisitor ()
        function_visitor.visit (root)
        return_visitor = FunctionVisitor ()
        return_visitor.visit (root)
        self.arguments = function_visitor.arguments
        self.return_values = function_visitor.return_values
        loaded_names_not_in_arguments = set(self.loaded_names).difference (self.arguments)
        if len(loaded_names_not_in_arguments) > 0:
            print (f'The following loaded names were not found in the arguments list: {loaded_names_not_in_arguments}')
        
    def run_code_and_collect_locals (self, code=None):
        if code is None: code=self.original_code
        
        get_old_variables_code = f'\nfrom nbmodular.core.cell2func import keep_variables\nkeep_variables ("{self.name}", "previous_values", locals ())'
        get_ipython().run_cell(get_old_variables_code)
        
        get_new_variables_code = code + f'\nfrom nbmodular.core.cell2func import keep_variables\nkeep_variables ("{self.name}", "current_values", locals ())'
        get_ipython().run_cell(get_new_variables_code)
        
        self.match_variables_and_locals ()
        
    def match_variables_and_locals (self):
        # previous variables / values
        self.previous_variables = [k for k in self.previous_variables if k in self.previous_values]
        self.previous_variables += [k for k in self.argument_variables if k in self.previous_values and k not in self.previous_variables]
        self.previous_variables += [k for k in self.created_variables if k in self.previous_values and k in self.loaded_names+self.argument_variables and k not in self.previous_variables]
        self.previous_values = {k:self.previous_values[k] for k in self.previous_values if k in self.previous_variables}
        
        # created variables / current values
        self.current_values = {k:self.current_values[k] for k in self.current_values if k in self.created_variables}
        self.all_values = {**self.previous_values, **self.current_values}
        self.argument_variables = [k for k in self.argument_variables if k in self.all_values]
        self.read_only_variables = [k for k in self.read_only_variables if k in self.all_values]
        
        self.all_variables = self.created_variables.copy()
        self.all_variables += [k for k in self.previous_variables if k not in self.all_variables]
        self.all_variables += [k for k in self.argument_variables if k not in self.all_variables]
        
        self.current_values = {k:self.current_values[k] for k in self.current_values if k in self.all_variables}
        self.previous_values = {k:self.current_values[k] for k in self.current_values if k in self.all_variables}
        
    def merge_functions (self, new_function, show=False):
        self.original_code += new_function.original_code
        self.parse_variables ()
        self.current_values = {**self.current_values, **new_function.current_values}
        self.previous_values = {**self.previous_values, **new_function.previous_values}
        self.match_variables_and_locals ()

        self.arguments = [] if self.unknown_input else self.arguments
        self.return_values = [] if self.unknown_output else self.return_values
        self.update_code (
            arguments=self.arguments, 
            return_values=self.return_values,
            display=show
        )
    
    def add_function_call (self, function):
        if 'added_functions' not in self:
            self.added_functions = []
        if 'function_calls' not in self:
            self.function_calls = '' 
        if function.name not in self.added_functions:
            self.added_functions.append (function.name)
            self.function_calls += f'{" "*self.tab_size}' + ','.join (function.return_values) + f' = {function.name}()\n'
            
    def add_to_signature (self, input=None, output=None, **kwargs):
        if input is not None:
            self.arguments += input
        if output is not None:
            self.return_values += output
        if input is not None or output is not None:
            self.update_code ()
    
    def __str__ (self):
        name = None if not hasattr(self, 'name') else self.name
        current_values = self.current_values.keys() if hasattr(self, 'current_values') else None
        return f'FunctionProcessor with name {name}, and fields: {self.keys()}\n    Arguments: {self.arguments}\n    Output: {self.return_values}\n    Locals: {current_values}'
    
    def __repr__ (self):
        return str(self)

# %% ../../nbs/cell2func.ipynb 15
class CellProcessor():
    """
    Processes the cell's code according to the magic command.
    """
    def __init__(self, tab_size=4, **kwargs):
        self.function_info = Bunch()
        self.current_function = Bunch()
        self.function_list = []
        
        self.test_function_info = Bunch()
        self.test_function_list = []
        
        self.test_data_function_info = Bunch()
        self.test_data_function_list = []
        
        self.all_variables = set()
        self.test_data_all_variables = set()
        self.test_all_variables = set()

        
        self.imports = ''
        #self.test_imports = 'from sklearn.utils import Bunch\nfrom pathlib import Path\nimport joblib\nimport pandas as pd\nimport numpy as np\n'
        self.test_imports = ''
        
        self.tab_size=tab_size
        #pdb.set_trace()
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
            self.file_path = self.file_path.parent / self.file_path.name.replace ('.ipynb', '.py')
            self.test_file_path = (self.nbs_folder.parent / 'tests').joinpath (*nb_path.parts[index+1:])/ f'test_{self.file_path.name}'
            self.test_file_path.parent.mkdir (parents=True, exist_ok=True)
        else:
            file_name = self.file_name.replace ('.ipynb', '.py')
            self.file_path = nb_path / file_name
            self.test_file_path = nb_path /  f'test_{file_name}'
            
        self.call_history = []
        
        self.parser = argparse.ArgumentParser(description='Process some integers.')
        self.parser.add_argument('-i', '--input', type=str, nargs='+', help='input')
        self.parser.add_argument('-o', '--output', type=str, nargs='+', help='output')
        self.parser.add_argument('-m', '--merge',  action='store_true', help='merge with previous function')
        self.parser.add_argument('-s', '--show',  action='store_true', help='show function code')
        self.parser.add_argument('-l', '--load',  action='store_true', help='load variables')
        self.parser.add_argument('--save',  action='store_true', help='save variables')
        self.parser.add_argument('-t', '--test',  action='store_true', help='test function / imports')
        self.parser.add_argument('-d', '--data',  action='store_true', help='data function')
        self.parser.add_argument('-n', '--norun',  action='store_true', help='data function')
        self.parser.add_argument('-p', '--permanent',  action='store_true', help='data function')
        
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
        function_name, kwargs = self.parse_signature (line)
        self.function (function_name, cell, call=call, **kwargs)

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
        cell, 
        func, 
        call,
        unknown_input=None,
        unknown_output=None,
        test=False,
        data=False,
        permanent=False
    ):
        root = ast.parse (cell)
        if False:
            function_visitor = FunctionVisitor ()
            function_visitor.visit (root)
        name=[x.name for x in ast.walk(root) if isinstance (x, ast.FunctionDef)]
            
        #if hasattr(function_visitor, 'name'):
        if len(name)>0:
            #func = function_visitor.name
            #arguments = function_visitor.arguments
            #return_visitor = ReturnVisitor ()
            #return_values = return_visitor.return_values
            if len(name) > 0:
                name=name[0]
            arguments = [[x.arg for x in node.args.args] for node in ast.walk(root) if isinstance (node, ast.FunctionDef)]
            if len(arguments)>0:
                arguments = arguments[0]
            return_values = [([x.id for x in node.value.elts] if hasattr(node.value, 'elts') else [node.value.id]) for node in ast.walk(root) if isinstance (node, ast.Return)] 
            if len(return_values)>0:
                return_values = return_values[0]
            
            unknown_input = False
            unknown_output = False
            input = arguments
            output = return_values
            defined = True
        else:
            defined = False
            arguments=[]
            return_values=[]
        
        if defined and not permanent:
                return_lines = 0
                original_code = ''
                for line in cell.splitlines():
                    if 'return' in line:
                        return_lines += 1
                    elif 'def' not in line:
                        original_code += line.strip() + '\n'
                cell = original_code
        
        this_function = FunctionProcessor (
            original_code=cell, 
            name=func, 
            call=call,
            tab_size=self.tab_size,
            arguments=arguments,
            return_values=return_values,
            unknown_input=unknown_input,
            unknown_output=unknown_output,
            test=test,
            data=data,
            defined=defined,
            permanent=permanent
        )
        if defined and permanent:
            this_function.code = cell
        this_function.parse_variables ()
        
        if defined:
            loaded_names_not_in_arguments = set(this_function.loaded_names).difference (arguments)
            if len(loaded_names_not_in_arguments) > 0:
                print (f'The following loaded names were not found in the arguments list: {loaded_names_not_in_arguments}')
        
        return this_function
    
    def create_function_register_and_run_code (
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
        load=False,
        save=False,
        test=False,
        norun=False,
        data=False,
        permanent=False
    ) -> FunctionProcessor:
        
        if test:
            func = 'test_' + func
        
        self.current_function = self.create_function (
            cell, 
            func, 
            call, 
            unknown_input=unknown_input,
            unknown_output=unknown_output,
            test=test,
            data=data,
            permanent=permanent
        )
        
        # register
        idx = self.current_function.idx = len(self.function_list)
        
        # get variables specific about this function
        path_variables = Path (self.file_name) / f'{func}.pk'
        if load and path_variables.exists():
            get_ipython().run_cell(f'''
            import joblib
            current_values = joblib.load ("{path_variables}")
            v = locals()
            v.update (current_values)''')
            return

        if not norun and collect_variables_values:           
            self.current_function.run_code_and_collect_locals()
        
        if save:
            path_variables.parent.mkdir (parents=True, exist_ok=True)
            joblib.dump (values_here, path_variables)
            
        if make_function:
            self.current_function.update_code ( 
                arguments=self.current_function.previous_variables if unknown_input and not self.current_function.test and not self.current_function.defined else [] if self.current_function.test else self.current_function.arguments if self.current_function.defined else input, 
                return_values=[] if unknown_output and not self.current_function.defined else self.current_function.return_values if self.current_function.defined else output,
                display=show
            )
            
        # add variables from current function to posterior_variables of all the previous functions
        #pdb.set_trace()
        # test data functions have output dependencies on test functions
        # test functions have no output dependencies
        function_list = (self.function_list if not self.current_function.test and not self.current_function.data 
                         else self.test_data_function_list if self.current_function.test and not self.current_function.data
                         else [])
        # if test function, its input comes from test data functions: 
        # - 1 add output dependencies to test data functions
        # - 2 add input dependencies from each test data function
        #if self.current_function.test:
        #    pdb.set_trace()
        for function in function_list[:idx]:
            function.posterior_variables += [v for v in self.current_function.previous_variables if v not in function.posterior_variables]
            if update_previous_functions and unknown_output and not function.defined:
                function.update_code (
                    return_values=[x for x in function.created_variables + function.argument_variables if x in function.posterior_variables], 
                    display=False
                )
            if self.current_function.test and function.test and function.data:
                self.current_function.add_function_call (function)
        
        if self.current_function.test and not self.current_function.data:
            self.current_function.update_code()
            
        if self.current_function.defined:
            self.previous_not_in_arguments = list (set (self.current_function.previous_variables).difference (self.current_function.arguments))
            self.posterior_not_in_results = list (set (self.current_function.posterior_variables).difference (self.current_function.return_values))
            if len (self.previous_not_in_arguments) > 0:
                print (f'Detected the following previous variables that are not in the argument list: {self.previous_not_in_arguments}')
                print (f'Detected the following posterior variables that are not in the return list: {self.posterior_not_in_results}')
        
        if self.current_function.test and self.current_function.data:
            common = set(self.current_function.all_variables).intersection (self.test_data_all_variables)
            if len(common)>0:
                raise ValueError (f'detected common variables with other test data functions {common}:')
        
        if not self.current_function.test and not self.current_function.data:
            self.all_variables |= set(self.current_function.all_variables)
        elif self.current_function.test and self.current_function.data:
            self.test_data_all_variables |= set(self.current_function.all_variables)
        elif self.current_function.test and not self.current_function.data:
            self.test_all_variables |= set(self.current_function.all_variables)
        
        return self.current_function
    
    def function (
        self,
        func,
        cell,
        merge=False,
        show=False,
        register_pipeline=True,
        pipeline_name=None,
        write=True,
        **kwargs
    ) -> None:
        
        for f in self.function_list:
            if f.name == func:
                self.function_list.remove (f)
                break
        
        this_function = self.create_function_register_and_run_code (func, cell, show=show, **kwargs)
        if func in self.function_info and merge:
            this_function = self.merge_functions (self.function_info[func], this_function, show=show)
        
        function_name = this_function.name
        if this_function.test:
            if this_function.data:
                self.test_data_function_info[function_name] = this_function
                self.test_data_function_list.append (this_function)
            else:
                self.test_function_info[function_name] = this_function
                self.test_function_list.append (this_function)
        else:
            self.function_info[function_name] = this_function
            self.function_list.append (this_function)
        
        if register_pipeline:
            self.register_pipeline (pipeline_name=pipeline_name)
        else:
            self.pipeline = None
        if write:
            self.write ()
            self.write (test=True)

            
    def merge_functions (self, f, g, show=False):
        f.merge_functions (g, show=show)
        return f
    
    def parse_args (self, line):
        argv = shlex.split(line, posix=(os.name == 'posix'))
        pars = self.parser.parse_args(argv)
        kwargs = vars(pars)
        return kwargs
                
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
            kwargs = vars(pars)
        else:
            kwargs = {}
        kwargs.update (signature)
            
        # print (function_name, signature)
        return function_name, kwargs
    
    def write_imports (
        self,
        cell,
        test=False,
        **kwargs
    ):
        get_ipython().run_cell (cell)
        if not test:
            self.imports += cell
        else:
            self.test_imports += cell
        self.write (test=test)
    
    def write (self, test=False):
        #pdb.set_trace()
        function_list = self.function_list if not test else self.test_function_list
        file_path = self.file_path if not test else self.test_file_path
        imports = self.imports if not test else self.test_imports
        with open (str(file_path), 'w') as file:
            #pdb.set_trace()
            file.write (imports)
            for function in function_list:
                function.write (file)
            if not test and self.pipeline is not None:
                self.pipeline.write (file)
                
    def print (self, function_name, test=False, data=False, **kwargs):
        if function_name == 'all':
            function_list = self.test_data_function_list if test and data else self.test_function_list if test else self.function_list
            for function in function_list:
                function.print ()
        else:
            if test and data:
                self.test_data_function_info[function_name].print ()
            elif test:
                self.test_function_info[function_name].print ()
            else:
                self.function_info[function_name].print ()
            
    def get_lib_path (self):
        return nbdev.config.get_config()['lib_path']
                   
    def get_nbs_path (self):
        return nbdev.config.get_config()['nbs_path']
    
    def pipeline_code (self, pipeline_name=None):
        pipeline_name = f'{self.file_name}_pipeline' if pipeline_name is None else pipeline_name
        
        code = (
f'''
def {pipeline_name} (test=False, load=True, save=True, result_file_name="{pipeline_name}"):

    # load result
    result_file_name += '.pk'
    path_variables = Path ("{self.file_name}") / result_file_name
    if load and path_variables.exists():
        result = joblib.load (path_variables)
        return result

''')
        return_values = set()
        for func in self.function_list:
            argument_list_str = ", ".join(func.arguments) if not func.data else "test=test"
            return_list_str = f'{", ".join(func.return_values)} = ' if len(func.return_values)>0 else ''
            return_values |= set(func.return_values)
            code += f'{" " * self.tab_size}' + f'{return_list_str}{func.name} ({argument_list_str})\n'
        
        return_values = list (return_values)
        result_str = "Bunch (" + "".join([f"{k}={k}," for k in return_values[:-1]])
        if len(return_values)>0:
            k = return_values[-1] 
            result_str = result_str + f"{k}={k}" + ")"
        else:
            result_str = result_str + ")"
        
        code += (
f'''
    # save result
    result = {result_str}
    if save:    
        path_variables.parent.mkdir (parents=True, exist_ok=True)
        joblib.dump (result, path_variables)
    return result
''')
            
        return code, pipeline_name
    
    def test_pipeline_code (self, pipeline_name=None):
        pipeline_name = f'{self.file_name}_pipeline' if pipeline_name is None else pipeline_name
        result_file_name_with_braces = 'f"test_{result_file_name}"'
        code = (
f'''
from sklearn.utils import Bunch
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def test_{pipeline_name} (test=True, prev_result=None, result_file_name="{pipeline_name}"):
    result = {pipeline_name} (test=test, load=True, save=True, result_file_name=result_file_name)
    if prev_result is None:
        prev_result = {pipeline_name} (test=test, load=True, save=True, result_file_name={result_file_name_with_braces})
    for k in prev_result:
        assert k in result
        if type(prev_result[k]) is pd.DataFrame:    
            pd.testing.assert_frame_equal (result[k], prev_result[k])
        elif type(prev_result[k]) is np.array:
            np.testing.assert_array_equal (result[k], prev_result[k])
        else:
            assert result[k]==prev_result[k]
''')
        return code, f'test_{pipeline_name}'
    
    def register_pipeline (self, pipeline_name=None):
        code, name = self.pipeline_code (pipeline_name=pipeline_name)
        self.pipeline = FunctionProcessor (code=code,
                                           arguments=[],
                                           return_values=[],
                                           name=name)
        get_ipython().run_cell(self.imports)
        get_ipython().run_cell(code)
        
        code, name = self.test_pipeline_code (pipeline_name=pipeline_name)
        self.test_pipeline = FunctionProcessor (code=code,
                                           arguments=[],
                                           return_values=[],
                                           name=name)
        get_ipython().run_cell(self.test_imports)
        get_ipython().run_cell(code)
    
    def print_pipeline (self, test=False, **kwargs):
        if test:
            code, name = self.test_pipeline_code()  
        else:
            code, name = self.pipeline_code()  
        print (code)

# %% ../../nbs/cell2func.ipynb 17
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
        
    @cell_magic
    def imports (self, line, cell):
        "Converts cell to function"
        kwargs = self.processor.parse_args (line)
        self.processor.write_imports (cell, **kwargs)
    
    @line_magic
    def write (self, line):
        return self.processor.write ()
    
    @line_magic
    def print (self, line):
        #pdb.set_trace()
        function_name, kwargs = self.processor.parse_signature (line)
        return self.processor.print (function_name, **kwargs)
    
    @line_magic
    def function_info (self, line):
        function_name, kwargs = self.processor.parse_signature (line)
        #pdb.set_trace()
        if kwargs.get('test', False):
            return self.processor.test_function_info [function_name]
        else:
            return self.processor.function_info [function_name]
        
    @line_magic
    def add_to_signature (self, line):
        function_name, kwargs = self.processor.parse_signature (line)
        self.function_info[function_name].add_to_signature (**kwargs)
    
    @line_magic
    def cell_processor (self, line):
        return self.processor
        
    @line_magic
    def pipeline_code (self, line):
        return self.processor.pipeline_code ()
    
    @line_magic
    def print_pipeline (self, line):
        kwargs = self.processor.parse_args (line)
        return self.processor.print_pipeline (**kwargs)
          
    @line_magic
    def match (self, line):
        p0 = '[a-zA-Z]\S*\s*\\([^-()]*\\)\s*->\s*\\([^-()]*\\)'
        p = '\\([^-()]*\\)'
        m = re.search (p0, line)
        if m is not None:
            inp, out = re.findall (p, line)
            #print (inp)
            #print (out)

# %% ../../nbs/cell2func.ipynb 20
def load_ipython_extension(ipython):
    """
    This module can be loaded via `%load_ext core.cell2func` or be configured to be autoloaded by IPython at startup time.
    """
    magics = CellProcessorMagic(ipython)
    ipython.register_magics(magics)

# %% ../../nbs/cell2func.ipynb 22
import pdb
def keep_variables (function, field, variable_values, self=None):
    """
    Store `variables` in dictionary entry `self.variables_field[function]`
    """
    frame_number = 0
    #pdb.set_trace()
    while not isinstance (self, FunctionProcessor):
        try:
            fr = sys._getframe(frame_number)
        except:
            break
        args = argnames(fr, True)
        if len(args)>0:
            self = fr.f_locals[args[0]]
        frame_number += 1
    if isinstance (self, FunctionProcessor):
        variable_values = {k: variable_values[k] for k in variable_values if not k.startswith ('_') and not callable(variable_values[k])}
        #pdb.set_trace()
        self[field]=variable_values
    
