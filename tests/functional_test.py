'''
Venn diagram plotting routines.
Test module (meant to be used via py.test).

The images, corresponding to the tests here are shown in the ipython notebook Venn2 - Special Case Tests.

Copyright 2014, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
import os.path

def exec_ipynb(filename):
    '''Executes all cells in a given ipython notebook consequentially'''
    import json, sys
    s = json.load(open(filename))
    for ws in s['worksheets']:
        for cell in ws['cells']:
            if cell['cell_type'] == 'code':
                code = ''.join(cell['input'])
                if sys.version_info.major == 2:
                    exec("exec code in locals()")
                else:
                    exec(code, locals())

def test_venn2():
    exec_ipynb(os.path.join(os.path.dirname(__file__), "venn2_functional.ipynb"))

def test_venn3():
    exec_ipynb(os.path.join(os.path.dirname(__file__), "venn3_functional.ipynb"))

    
    
    