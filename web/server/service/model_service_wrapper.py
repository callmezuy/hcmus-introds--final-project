import os
from importlib.machinery import SourceFileLoader

_here = os.path.dirname(__file__)
_module_path = os.path.join(_here, 'model.service.py')

_model = SourceFileLoader('model_service', _module_path).load_module()

# Re-export selected functions
build_model_input = getattr(_model, 'build_model_input')
get_top100_symbols = getattr(_model, 'get_top100_symbols')

# Optional exports used by API layer
run_model_on_top100 = getattr(_model, 'run_model_on_top100', None)
predict_for_symbol = getattr(_model, 'predict_for_symbol', None)
