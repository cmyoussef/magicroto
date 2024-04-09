import ast
import importlib
import os
import re
import sys

try:
    import nuke
except ModuleNotFoundError:
    pass


def get_current_package():
    """
    Returns the full package name of the current module and its directory path.

    :return: Tuple containing the full package name and the current directory path.
    """
    current_path = os.path.dirname(__file__)

    # Get the list of parent folders and reverse it
    parent_folders = os.path.normpath(current_path).split(os.sep)[::-1]

    # Find the root package by checking the loaded modules
    package_name = []
    for folder in parent_folders:
        package_name.insert(0, folder)
        full_package_name = '.'.join(package_name)
        if full_package_name in sys.modules:
            print(f'{full_package_name}, Is found')
            return full_package_name, current_path

    print(f'{full_package_name}, not found')
    return full_package_name, current_path


def all_classes_from_module(module):
    classes = []
    for m in sys.modules:
        if module in m:
            classes.append(m)
    return classes


def get_classes():
    """
    Fetches and returns classes grouped by their MENU_GRP attribute.

    :return: Dictionary containing classes categorized by their MENU_GRP attribute.
    """
    classes = {}
    full_package_name, current_path = get_current_package()
    current_module = os.path.splitext(current_path)[0]
    # Loop through files in the current directory
    for filename in os.listdir(current_path):
        if filename == '__init__.py' or filename == current_module:
            continue
        module_name, ext = os.path.splitext(filename)
        if ext == '.py':
            full_module_name = f'{full_package_name}.{module_name}'
            module = importlib.import_module(full_module_name)
            menu_grp = 'generate'
            dev_mode = False
            # Fetch MENU_GRP if available
            if hasattr(module, 'MENU_GRP'):
                menu_grp = getattr(module, 'MENU_GRP')

            # Fetch MENU_GRP if available
            if hasattr(module, 'DEV_CLS'):
                dev_mode = getattr(module, 'DEV_CLS')

            # Fetch classes and categorize them
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name.lower() == module_name:
                    classes.setdefault(menu_grp, []).append({'cls': attr, 'module': full_package_name,
                                                             'dev_mode': dev_mode})
    return classes


def reload_classes(classes_list):
    classes_list = ast.literal_eval(classes_list)
    unique_packages = set()

    # Identify base packages and collect all relevant modules and submodules
    modules_to_reload = set()
    for cls_name in classes_list:
        if '.' not in cls_name:
            continue
        package_name = cls_name.split('.')[0]
        unique_packages.add(package_name)

    for module_name in sys.modules.keys():
        if any(module_name.startswith(package + '.') or module_name == package for package in unique_packages):
            modules_to_reload.add(module_name)

    # Delete identified modules from sys.modules to force a reload
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            print(f"Deleting {module_name}")
            del sys.modules[module_name]

    print('-' * 10)

    # Reload packages and their submodules
    for module_name in sorted(modules_to_reload):
        print(f"Reloading {module_name}")
        try:
            # Attempt to reload the module
            reloaded_module = importlib.import_module(module_name)
            importlib.reload(reloaded_module)
        except Exception as e:
            print(f"Error reloading {module_name}: {e}")


def populate_toolbar(toolbar, dev_mode=False):
    """
    Populates the toolbar with menu items for each class.

    :param toolbar: nuke.toolbar, Toolbar object to populate.
    :param dev_mode: Bool, override build for the dev classes
    """
    dev_mode = dev_mode or os.environ.get("NUKE_DEV", False)
    full_package_name, current_path = get_current_package()
    icon_path = os.path.join(os.path.dirname(current_path), 'icons', "toolbar.png")
    m = toolbar.addMenu("MagicRoto", icon=icon_path)
    classes_dict = get_classes()
    classes_list = []
    # Populate the menu with commands for each class
    print("Loading MagicRoto toolbar:")
    for group, gizmos in classes_dict.items():
        print(f'|_Populating {group}')
        for g in gizmos:
            if not dev_mode and g.get('dev_mode', False):
                continue
            cls = g.get('cls')
            cmd_name = cls.__name__
            cmd_module = cls.__module__
            classes_list.append(f'{cmd_module}.{cmd_name}')
            base_class = cmd_module.rsplit('.')[0]
            m.addCommand(f'NB_{cmd_name}', f"import {base_class};{cmd_module}.{cmd_name}(name='NB_{cmd_name}')")
            print(f"|_|_creating button NB_{cmd_name} with class {cmd_module}.{cmd_name}(name='NB_{cmd_name}')")
        m.addSeparator()

    for c in all_classes_from_module(full_package_name.rsplit('.')[0]):
        if c not in classes_list:
            classes_list.append(c)

    # if dev_mode:
    # Add reload command to the menu
    m.addCommand('Reload modules', f'{full_package_name}.reload_classes("{classes_list}")')


def fix_mangled_text():
    pattern = re.compile(r'ð[^\x00-\x7F]+')

    def replacer(match):
        try:
            return match.group(0).encode('latin1').decode('utf-8')
        except UnicodeEncodeError:
            return match.group(0)

    for node in nuke.allNodes():
        all_knobs = node.knobs()

        if 'gizmo_class_type' not in all_knobs:
            continue

        for knob_name, knob_obj in all_knobs.items():
            mangled_text = knob_obj.label()

            if any(ord(char) >= 128 for char in mangled_text):
                fixed_text = pattern.sub(replacer, mangled_text)
                knob_obj.setLabel(fixed_text)
