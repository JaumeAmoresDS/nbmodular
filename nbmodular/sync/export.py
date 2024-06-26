

# %% auto 0
__all__ = ['obtain_function_name_and_test_flag', 'transform_test_source_for_docs', 'set_paths_nb_processor', 'NbMagicProcessor',
           'NbMagicExporter', 'nbm_export', 'process_cell_for_nbm_update', 'nbm_update']

# %% ../../nbs/export.ipynb 2
# Standard
import shlex
import os
import ast
from pathlib import Path
import logging
import joblib
import warnings
from typing import List

# 3rd party
from sklearn.utils import Bunch
from nbdev.processors import Processor, NBProcessor
from nbdev.config import get_config
from nbdev.export import nb_export
from nbdev.sync import _update_mod
from execnb.shell import CaptureShell
from execnb.nbio import new_nb, mk_cell, read_nb, write_nb, NbCell

# nbmodular
from ..core.utils import set_log_level
from ..core.cell2func import CellProcessor

# %% ../../nbs/export.ipynb 5
def obtain_function_name_and_test_flag (line, cell):
    root = ast.parse (cell)
    name=[x.name for x in ast.walk(root) if isinstance (x, ast.FunctionDef)]
    argv = shlex.split(line, posix=(os.name == "posix"))
    is_test = "--test" in argv
    
    if len(name)>0:
        function_name=name[0]
    else:        
        function_name=argv[0] if len(argv)>0 else ""
        if function_name.startswith("-"):
            function_name = ""

    if function_name=="":
        raise RuntimeError ("Couldn't find function name.")
    
    return function_name, is_test

# %% ../../nbs/export.ipynb 7
def transform_test_source_for_docs (source_lines, idx, tab_size):
    start = 2 if idx==0 else 1
    transformed_lines=[]
    for line in source_lines[start:]:
        transformed_lines.append (line[tab_size:] if line.startswith (" "*tab_size) else line)
    return "\n".join (transformed_lines)

# %% ../../nbs/export.ipynb 9
def set_paths_nb_processor (
    nb_processor,
    path,
):
    nb_processor.path=Path(path)
    nb_processor.file_name_without_extension = nb_processor.path.name[:-len(".ipynb")]

    config = get_config ()
    nb_processor.root_path=config.config_path
    nb_processor.nbs_path = config["nbs_path"]
    nb_processor.nbm_path = config["nbm_path"]
    nb_processor.lib_path = config["lib_path"]

    # In diagram: nbs/nb.ipynb
    nb_processor.dest_nb_path = Path(
        str(nb_processor.path).replace(
            str(nb_processor.nbm_path), 
            str(nb_processor.nbs_path)
        )
    )
    # In diagram: nbs/test_nb.ipynb
    nb_processor.test_dest_nb_path = nb_processor.dest_nb_path.parent / f"test_{nb_processor.file_name_without_extension}.ipynb"
    # In diagram: .nbs/nb.ipynb
    nb_processor.tmp_nb_path = Path(str(nb_processor.path).replace(nb_processor.nbm_path, ".nbs"))
    nb_processor.tmp_nb_path.parent.mkdir (parents=True, exist_ok=True)

    # step 2 (beginning) in diagram
    # In diagram: .nbs/_nb.ipynb
    nb_processor.duplicate_tmp_path = nb_processor.tmp_nb_path.parent / f"_{nb_processor.tmp_nb_path.name}"

    # step 3 in diagram
    # In diagram: .nbs/nb.ipynb
    nb_processor.tmp_dest_nb_path = nb_processor.tmp_nb_path.parent / nb_processor.dest_nb_path.name
    # In diagram: .nbs/test_nb.ipynb
    nb_processor.tmp_test_dest_nb_path = nb_processor.tmp_nb_path.parent / nb_processor.test_dest_nb_path.name

    # step 5 (beginning) in diagram
    nb_processor.duplicate_dest_nb_path = nb_processor.dest_nb_path.parent / f"_{nb_processor.dest_nb_path.name}"

    # python module paths
    try:
        index = nb_processor.path.parts.index(nb_processor.nbm_path)
    except:
        raise RuntimeError (f"{nb_processor.nbm_path.name} not found in {nb_processor.path}")
    parent_parts = nb_processor.path.parent.parts[index+1:]
    # module paths
    nb_processor.dest_python_path = (
        nb_processor.lib_path.name + "/" +
        "/".join(parent_parts) + "/" +
        nb_processor.file_name_without_extension + ".py"
    )
    nb_processor.test_dest_python_path = (
        nb_processor.lib_path.name + "/tests/" + 
        "/".join(parent_parts) + "/" +
        "test_" + nb_processor.file_name_without_extension + ".py"
    )
    # to be used in default_exp cell (see NBExporter)
    nb_processor.dest_module_path = ".".join(parent_parts) + "." + nb_processor.file_name_without_extension
    nb_processor.test_dest_module_path = "tests." + ".".join(parent_parts) + "." + f"test_{nb_processor.file_name_without_extension}"

# %% ../../nbs/export.ipynb 11
class NbMagicProcessor (Processor):
    def __init__ (
        self, 
        path,
        nb=None,
        logger=None,
        log_level="INFO",
    ):
        nb = read_nb(path) if nb is None else nb
        super().__init__ (nb)
        self.logger = logging.getLogger("nb_exporter") if logger is None else logger
        set_log_level (self.logger, log_level)
        self.logger.info (f"Analyzing code from notebook {path}")
        self.cell_processor = CellProcessor ()
        self.cell_processor.set_run_tests (False)

    def cell(self, cell):
        source_lines = cell.source.splitlines() if cell.cell_type=="code" else []
        if len(source_lines) > 0 and source_lines[0].strip().startswith("%%"):
            line = source_lines[0]
            words = line.split ()
            command = words[0][2:]
            if command in self.cell_processor.magic_commands_list:
                self.cell_processor.process_function_call (
                    line=" ".join (words[1:]), 
                    cell="\n".join(source_lines[1:]) if len(source_lines)>1 else "",
                    add_call=True, 
                    is_class=command=="class"
                )

# %% ../../nbs/export.ipynb 17
class NbMagicExporter(Processor):
    def __init__ (
        self, 
        path,
        nb=None,
        code_cells_file_name=None,
        code_cells_path=".nbmodular",
        execute=True,
        logger=None,
        log_level="INFO",
        tab_size=4,
    ):
        nb = read_nb(path) if nb is None else nb
        super().__init__ (nb)
        self.logger = logging.getLogger("nb_exporter") if logger is None else logger
        set_log_level (self.logger, log_level)
        set_paths_nb_processor (self, path)
        self.code_cells_path=Path(code_cells_path)
        code_cells_file_name = self.file_name_without_extension if code_cells_file_name is None else code_cells_file_name
                
        self.logger.info (f"Analyzing code from notebook {self.path}")
        self.nb_magic_processor = NbMagicProcessor (
            path, 
            nb=nb, 
            logger=logger, 
            log_level=log_level
        )
        NBProcessor (path, self.nb_magic_processor, rm_directives=False, nb=nb).process()

        self.function_names = {}
        self.test_function_names = {}
        self.cells = []
        self.test_cells = []
        self.doc_cells = []
        
        self.default_exp_cell = mk_cell (f"#|default_exp {self.dest_module_path}")
        self.default_test_exp_cell = mk_cell (f"#|default_exp {self.test_dest_module_path}")

        # list of types of cells, to be used by importer
        # can be "code", "test", "original"
        self.cell_types = []

        # other
        self.tab_size = tab_size
    
    def cell(self, cell):
        source_lines = cell.source.splitlines() if cell.cell_type=="code" else []
        is_test = False
        cell_type = "original"
        if len(source_lines) > 0 and source_lines[0].strip().startswith("%%"):
            line = source_lines[0]
            source = "\n".join (source_lines[1:])
            to_export = False
            is_test = False
            if line.startswith("%%function") or line.startswith("%%method"):
                function_name, is_test = obtain_function_name_and_test_flag (line, source)
                function_names = self.test_function_names if is_test else self.function_names
                if function_name in function_names:
                    function_names[function_name] += 1
                else:
                    function_names[function_name] = 0
                idx = function_names[function_name]
                self.logger.debug (f"{function_name}, {idx}, is test: {is_test}")
                code_cells = self.nb_magic_processor.cell_processor.test_code_cells if is_test else self.nb_magic_processor.cell_processor.code_cells
                if function_name not in code_cells:
                    raise RuntimeError (f"Function {function_name} not found in code_cells dictionary with keys {code_cells.keys()}")
                code_cells = code_cells[function_name]
                if len (code_cells) <= idx:
                    raise RuntimeError (f"Function {function_name} has {len(code_cells)} cells, which is lower than index {idx}.")
                code_cell = code_cells[idx]
                self.logger.debug ("code:")
                self.logger.debug (f"{code_cell.code}valid: {code_cell.valid}")
                if code_cell.valid:
                    source = code_cell.code
                    to_export = True
            elif line.startswith ("%%include") or line.startswith ("%%class"):
                to_export = True
            if to_export:
                line = line.replace ("%%", "#@@")
                code_source = line + "\n" + source
                code_source = "#|export\n" + code_source
                doc_source = "#|export\n" + source # doc_source does not include first line with #@@
                new_cell = NbCell (cell.idx_, cell)
                new_cell["source"] = code_source
                if is_test:
                    self.test_cells.append (new_cell)
                    cell_type = "test"
                else:
                    self.cells.append (new_cell)
                    cell_type = "code"
            else:
                doc_source=source # doc_source does not include first line with %% (? to think about)
            if is_test:
                doc_source = transform_test_source_for_docs (source_lines, idx, self.tab_size)
            cell["source"]=doc_source
            self.doc_cells.append(cell)

        self.cell_types.append (cell_type)

    def end(self):
        # store cell_types for later use by NBImporter
        joblib.dump (self.cell_types, self.code_cells_path / "cell_types.pk")

        write_nb (self.nb, self.tmp_nb_path)
        self.nb.cells = self.cells
        if len(self.cells) > 0:
            self.nb.cells = [self.default_exp_cell] + self.cells
            write_nb (self.nb, self.dest_nb_path)
            nb_export (self.dest_nb_path)
        if len(self.test_cells) > 0:
            self.nb.cells = [self.default_test_exp_cell] + self.test_cells
            write_nb (self.nb, self.test_dest_nb_path)
            nb_export (self.test_dest_nb_path)
        
        # step 2 (beginning) in diagram
        self.tmp_nb_path.rename (self.duplicate_tmp_path)
        
        # step 3 in diagram
        self.dest_nb_path.rename (self.tmp_dest_nb_path)
        self.test_dest_nb_path.rename (self.tmp_test_dest_nb_path)

        # step 2 (end) in diagram
        self.duplicate_tmp_path.rename (self.dest_nb_path)

# %% ../../nbs/export.ipynb 19
def nbm_export (
    path,
    **kwargs,
):
    path=Path(path)
    nb = read_nb(path)
    processor = NbMagicExporter (
        path,
        nb=nb,
        **kwargs,
    )
    NBProcessor (path, processor, rm_directives=False, nb=nb).process()

# %% ../../nbs/export.ipynb 32
def process_cell_for_nbm_update (cell: NbCell):
    source_lines = cell.source.splitlines() if cell.cell_type=="code" else []
    found_directive = False
    found_magic = False
    for line_number, line in enumerate(source_lines):
        line = line.strip()
        if len(line) > 0:
            if found_directive:
                if line.startswith("#@@") or line.startswith("# @@"):
                    line = line[3:] if line.startswith("#@@") else line[4:]
                    words = line.split()
                    if len(words) > 0 and words[0] not in ["function", "method", "include", "class"]:
                        warnings.warn (f"Found #@@ with a word after that, and this word is not in ['function', 'method', 'include', 'class']")
                    line = f"{'%%'}{line}"
                    found_magic = True
                    break
            elif line.startswith("#|"):
                found_directive = True
            else:
                if found_directive:
                    raise ValueError ("Line with #@@, corresponding to magic line in notebook, not found after having found line with directive #|")
                else:
                    raise ValueError ("Directive line not found at beginning of cell")
    if not found_magic:
        raise ValueError ("Magic line not found at beginning of cell")
    cell.source = "\n".join ([line] + source_lines [line_number+1:])

# %% ../../nbs/export.ipynb 34
def nbm_update (
    path,
    code_cells_path=".nbmodular",
    logger=None,
    log_level="INFO",
):
    nb_processor = Bunch ()
    nb_processor.code_cells_path=Path(code_cells_path)

    nb_processor.logger = logging.getLogger("nb_importer") if logger is None else logger
    set_log_level (nb_processor.logger, log_level)
    set_paths_nb_processor (nb_processor, path)

    # prior to step 5 in diagram:
    # nbs/nb.ipynb => nbs/_nb.ipynb
    nb_processor.dest_nb_path.rename (nb_processor.duplicate_dest_nb_path)

    # step 5 in diagram:
    # .nbs/nb.ipynb => nbs/nb.ipynb
    nb_processor.tmp_dest_nb_path.rename (nb_processor.dest_nb_path)
    # .nbs/test_nb.ipynb => nbs/test_nb.ipynb
    nb_processor.tmp_test_dest_nb_path.rename (nb_processor.test_dest_nb_path)

    # step 5 in diagram: nbdev_update
    _update_mod (nb_processor.dest_python_path, lib_dir=nb_processor.lib_path.parent)
    _update_mod (nb_processor.test_dest_python_path, lib_dir=nb_processor.lib_path.parent)

    # obtain cell types and read them from notebooks
    nb_processor.cell_types = joblib.load (nb_processor.code_cells_path / "cell_types.pk")
    original_nb = read_nb(path)
    dest_nb = read_nb(nb_processor.dest_nb_path)
    test_dest_nb = read_nb(nb_processor.test_dest_nb_path)
    nb_processor.cells = []
    code_idx, test_idx = 1, 1
    for original_idx, cell_type in enumerate(nb_processor.cell_types):
        cell = None
        if cell_type == "original":
            cell = original_nb.cells[original_idx]
        elif cell_type == "code":
            cell = dest_nb.cells[code_idx]
            code_idx += 1
        elif cell_type == "test":
            cell = test_dest_nb.cells[test_idx]
            test_idx += 1
        if cell is not None:
            if cell_type in ["code", "test"]:
                process_cell_for_nbm_update (cell)
            nb_processor.cells.append (cell)

    original_nb.cells = nb_processor.cells
    write_nb (original_nb, path)
