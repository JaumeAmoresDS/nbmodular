def first():
    pass

def second():
    pass

# -----------------------------------------------------
# pipeline
# -----------------------------------------------------
def export_pipeline (test=False, load=True, save=True, result_file_name="export_pipeline"):
    """Pipeline calling each one of the functions defined in this module."""
    
    # load result
    result_file_name += '.pk'
    path_variables = Path ("export") / result_file_name
    if load and path_variables.exists():
        result = joblib.load (path_variables)
        return result

    first ()
    second ()

    # save result
    result = Bunch ()
    if save:    
        path_variables.parent.mkdir (parents=True, exist_ok=True)
        joblib.dump (result, path_variables)
    return result

