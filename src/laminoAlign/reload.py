import importlib
import pkgutil


def lamReload():
    """
    - Call lamReload() to reload all packages
    - Useful for when you are actively developing/changing code
    """

    # For development: reload all packages
    packageName = 'laminoAlign'
    package = importlib.import_module(packageName)
    # Iterate over the modules in the package
    for _, moduleName, _ in pkgutil.iter_modules(package.__path__):
        fullModuleName = f"{packageName}.{moduleName}"
        # Reload the module
        importlib.reload(importlib.import_module(fullModuleName))
