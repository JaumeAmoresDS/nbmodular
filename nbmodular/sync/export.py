# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/export.ipynb.

# %% auto 0
__all__ = ['obtain_function_name_and_test_flag', 'NBExporter', 'process_notebook', 'NBImporter']

# %% ../../nbs/export.ipynb 2
import shlex
import os
import ast
import joblib
from pathlib import Path
import logging

from nbdev.processors import Processor, NBProcessor
from execnb.shell import CaptureShell
from execnb.nbio import new_nb, mk_cell, read_nb, write_nb, NbCell

from ..core.utils import set_log_level

# %% ../../nbs/export.ipynb 5
def obtain_function_name_and_test_flag (line, cell):
    root = ast.parse (cell)
    name=[x.name for x in ast.walk(root) if isinstance (x, ast.FunctionDef)]
    argv = shlex.split(line, posix=(os.name == 'posix'))
    is_test = '--test' in argv
    
    if len(name)>0:
        function_name=name[0]
    else:        
        function_name=argv[0] if len(argv)>0 else ''
        if function_name.startswith('-'):
            function_name = ''

    if function_name=='':
        raise RuntimeError ("Couldn't find function name.")
    
    return function_name, is_test

# %% ../../nbs/export.ipynb 7
class NBExporter(Processor):
    def __init__ (
        self, 
        path,
        nb=None,
        code_cells_file_name=None,
        code_cells_path='.nbmodular',
        execute=True,
        logger=None,
        log_level='INFO',
    ):
        nb = read_nb(path) if nb is None else nb
        super().__init__ (nb)
        self.logger = logging.getLogger('nb_exporter') if logger is None else logger
        set_log_level (self.logger, log_level)
        path=Path(path)
        file_name_without_extension = path.name[:-len('.ipynb')]
        self.code_cells_path=Path(code_cells_path)
        code_cells_file_name = file_name_without_extension if code_cells_file_name is None else code_cells_file_name
        path_to_code_cells_file = self.code_cells_path / f'{code_cells_file_name}.pk'
        test_path_to_code_cells_file = self.code_cells_path / f'test_{code_cells_file_name}.pk'
        if not path_to_code_cells_file.exists() and not test_path_to_code_cells_file.exists():
            if path.exists() and execute:
                self.logger.info (f'Executing notebook {path}')
                caputure_shell = CaptureShell ()
                caputure_shell.execute(path, 'tmp.ipynb')
            elif not execute:
                raise RuntimeError (f'Exported pickle files not found {path_to_code_cells_file} and execute is False')
            else:
                raise RuntimeError (f'Neither the exported pickle files {path_to_code_cells_file} nor the notebook {path} were found.')
        
        self.code_cells = joblib.load (path_to_code_cells_file) if path_to_code_cells_file.exists() else {}
        self.test_code_cells = joblib.load (test_path_to_code_cells_file) if path_to_code_cells_file.exists() else {}

        self.function_names = {}
        self.test_function_names = {}
        self.cells = []
        self.test_cells = []
        self.dest_nb_path = path.parent / f'dest_{file_name_without_extension}.ipynb'
        self.test_dest_nb_path = path.parent / f'dest_test_{file_name_without_extension}.ipynb'
    
    def cell(self, cell):
        source_lines = cell.source.splitlines() if cell.cell_type=='code' else []
        is_test = False
        if len(source_lines) > 0 and source_lines[0].strip().startswith('%%'):
            line = source_lines[0]
            source = '\n'.join (source_lines[1:])
            to_export = False
            if line.startswith('%%function') or line.startswith('%%method'):
                function_name, is_test = obtain_function_name_and_test_flag (line, source)
                function_names = self.test_function_names if is_test else self.function_names
                if function_name in function_names:
                    function_names[function_name] += 1
                else:
                    function_names[function_name] = 0
                idx = function_names[function_name]
                print (f'{function_name}, {idx}, is test: {is_test}')
                code_cells = self.test_code_cells if is_test else self.code_cells
                if function_name not in code_cells:
                    raise RuntimeError (f'Function {function_name} not found in code_cells dictionary with keys {code_cells.keys()}')
                code_cells = code_cells[function_name]
                if len (code_cells) <= idx:
                    raise RuntimeError (f'Function {function_name} has {len(code_cells)} cells, which is lower than index {idx}.')
                code_cell = code_cells[idx]
                print ('code:')
                print (code_cell.code, 'valid: ', code_cell.valid)
                if code_cell.valid:
                    source = code_cell.code
                    to_export = True
            elif line.startswith ('%%include') or line.startswith ('%%class'):
                to_export = True
            line = line.replace ('%%', '#@@')
            source = line + '\n' + source
            if to_export:
                source = '#|export\n' + source
            #new_cell = {**cell}
            #new_cell = NbCell (cell.idx_, cell)
            #new_cell['source'] = source
            cell['source'] = source
        #else:
        #    new_cell = cell
        #self.cells.append (new_cell)
        if is_test:
            self.test_cells.append (cell)
        else:
            self.cells.append (cell)
    def end(self): 
        #nb = new_nb (self.cells)
        #write_nb (nb, self.dest_nb_path)
        self.nb.cells = self.cells
        write_nb (self.nb, self.dest_nb_path)
        self.nb.cells = self.test_cells
        write_nb (self.nb, self.test_dest_nb_path)


# %% ../../nbs/export.ipynb 9
def process_notebook (
    path,
    action='export',
    **kwargs,
):
    path=Path(path)
    nb = read_nb(path)
    if action=='export':
        processor = NBExporter (
            path,
            nb=nb,
            **kwargs,
        )
    else:
        test_path = path.parent / f'test_{path.name}'
        test_nb = read_nb(test_path)
        processor = NBImporter (
            path,
            test_path,
            nb=nb,
            test_nb=test_nb,
            **kwargs,
        )
    NBProcessor (path, processor, rm_directives=False, nb=nb).process()

class NBImporter (Processor):
    def __init__ (
        self, 
        path,
        test_path,
        nb=None,
        test_nb=None,
        code_cells_file_name=None,
        code_cells_path='.nbmodular',
        execute=True,
        logger=None,
        log_level='INFO',
    ):
        self.nb = read_nb(path) if nb is None else nb
        self.test_nb = read_nb(test_path) if test_nb is None else test_nb
        super().__init__ (self.nb)
        self.logger = logging.getLogger('nb_importer') if logger is None else logger
        set_log_level (self.logger, log_level)
        path=Path(path)
        test_path=Path(test_path)

        self.cells = []
        self.test_cells = []
        file_name_without_extension = path.name[:-len('.ipynb')]
        self.dest_nb_path = path.parent / f'orig_{file_name_without_extension}.ipynb'
        self.test_dest_nb_path = path.parent / f'orig_test_{file_name_without_extension}.ipynb'
    
    def cell(self, cell):
        source_lines = cell.source.splitlines() if cell.cell_type=='code' else []
        is_test = False
        if len(source_lines) > 0 and source_lines[0].strip().startswith('%%'):
            line = source_lines[0]
            source = '\n'.join (source_lines[1:])
            to_export = False
            if line.startswith('%%function') or line.startswith('%%method'):
                function_name, is_test = obtain_function_name_and_test_flag (line, source)
                function_names = self.test_function_names if is_test else self.function_names
                if function_name in function_names:
                    function_names[function_name] += 1
                else:
                    function_names[function_name] = 0
                idx = function_names[function_name]
                print (f'{function_name}, {idx}, is test: {is_test}')
                code_cells = self.test_code_cells if is_test else self.code_cells
                if function_name not in code_cells:
                    raise RuntimeError (f'Function {function_name} not found in code_cells dictionary with keys {code_cells.keys()}')
                code_cells = code_cells[function_name]
                if len (code_cells) <= idx:
                    raise RuntimeError (f'Function {function_name} has {len(code_cells)} cells, which is lower than index {idx}.')
                code_cell = code_cells[idx]
                print ('code:')
                print (code_cell.code, 'valid: ', code_cell.valid)
                if code_cell.valid:
                    source = code_cell.code
                    to_export = True
            elif line.startswith ('%%include') or line.startswith ('%%class'):
                to_export = True
            line = line.replace ('%%', '#@@')
            source = line + '\n' + source
            if to_export:
                source = '#|export\n' + source
            #new_cell = {**cell}
            #new_cell = NbCell (cell.idx_, cell)
            #new_cell['source'] = source
            cell['source'] = source
        #else:
        #    new_cell = cell
        #self.cells.append (new_cell)
        if is_test:
            self.test_cells.append (cell)
        else:
            self.cells.append (cell)
    def end(self): 
        #nb = new_nb (self.cells)
        #write_nb (nb, self.dest_nb_path)
        self.nb.cells = self.cells
        write_nb (self.nb, self.dest_nb_path)
        self.nb.cells = self.test_cells
        write_nb (self.nb, self.test_dest_nb_path)
