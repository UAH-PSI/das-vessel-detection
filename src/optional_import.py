def optional_import(module_name, alias=None, message=None):
    try:
        module = __import__(module_name)
    except ImportError:
        # Stop execution immediately with your custom message
        raise ImportError(message or f"{module_name!r} unavailable in this edition.")
    if alias:
        globals()[alias] = module
    return module
