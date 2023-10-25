def first_function():
    a = 3 
    print ('a', a)
    return a
def second_function():
    b = 4
    c = a+b
    print (a, b, c)
    return c,b
def myf():
    print ('hello')
    a = 3
    b = 4
    c = 5
    d = a+b+c
    return a,b
def myf():
    print ('hello')
    a = 3
    b = 4
    c = 5
    d = a+b+c
    return c,a,d,b
def myf():
    print ('hello')
    a = 3
    b = 4
    c = 5
    d = a+b+c
    return c,a,d,b
def my_defined_function (x, a=3):

    print (x)
    print (a)

def nbdev_sync_pipeline (test=False, load=True, save=True, result_file_name="nbdev_sync_pipeline"):

    # load result
    result_file_name += '.pk'
    path_variables = Path ("nbdev_sync") / result_file_name
    if load and path_variables.exists():
        result = joblib.load (path_variables)
        return result

    a = first_function ()
    c, b = second_function ()
    a, b = myf ()
    c, a, d, b = myf ()
    c, a, d, b = myf ()
    my_defined_function (x, a)

    # save result
    result = Bunch (c=c,a=a,d=d,b=b)
    if save:    
        path_variables.parent.mkdir (parents=True, exist_ok=True)
        joblib.dump (result, path_variables)
    return result
