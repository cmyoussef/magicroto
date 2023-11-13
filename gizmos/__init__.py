import ast
import importlib
import os
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
            return full_package_name, current_path

    print(f'{full_package_name}, not found')
    return full_package_name, current_path


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


def update_command(classes_list):
    """
    Updates the unique commands in Nuke to use reloaded classes.

    :param classes_list: List of fully qualified class names.
    """
    toolbar = nuke.menu("Nodes")
    sd_menu = toolbar.findItem("SD2")

    if sd_menu is None:
        print("SD2 menu not found.")
        return

    print("Updating Nuke SD2 menu:")
    for cls_name in classes_list:
        module_name, cmd_name = cls_name.rsplit('.', 1)

        full_cmd_name = f'SD_{cmd_name}'
        existing_command = sd_menu.findItem(full_cmd_name)

        if existing_command:
            new_command = f"{module_name}.{cmd_name}(name='SD_{cmd_name}')"
            existing_command.setScript(new_command)
            print(f"|_|_updating command {full_cmd_name} to {new_command}")


def reload_classes(classes_list):
    classes_list = ast.literal_eval(classes_list)
    unique_modules = set()
    base_modules = set()

    for cls_name in classes_list:
        module_name, cls = cls_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls_obj = getattr(module, cls)

        for base_cls in cls_obj.__bases__:
            base_module_name = base_cls.__module__
            base_modules.add(base_module_name)

        unique_modules.add(module_name)

    all_modules = base_modules | unique_modules
    for module_name in list(all_modules):
        module_parts = module_name.split('.')
        parent = ''
        for part in module_parts[:-1]:
            parent = f"{parent}.{part}" if parent else part
            all_modules.add(parent)

    sorted_modules = sorted(all_modules, key=lambda x: x.count('.'))
    reversed_modules = sorted_modules[::-1]

    for module_name in reversed_modules:
        if module_name in sys.modules:
            print(f"Deleting {module_name}")
            del sys.modules[module_name]

    print('-' * 10)

    for module_name in sorted_modules:
        print(f"Reloading {module_name}")
        importlib.import_module(module_name)
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    print('-' * 10)
    update_command(classes_list)


def populate_toolbar(toolbar, dev_mode=False):
    """
    Populates the toolbar with menu items for each class.

    :param toolbar: nuke.toolbar, Toolbar object to populate.
    :param dev_mode: Bool, override build for the dev classes
    """
    full_package_name, current_path = get_current_package()
    icon_path = os.path.join(os.path.dirname(current_path), 'icons', "sd.png")
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
            suffix = 'MR'
            m.addCommand(f'{suffix}_{cmd_name}', f"import {base_class};{cmd_module}.{cmd_name}(name='{suffix}_{cmd_name}')")
            print(f"|_|_creating button {suffix}_{cmd_name} with class {cmd_module}.{cmd_name}(name='{suffix}_{cmd_name}')")
        m.addSeparator()

    if dev_mode:
        # Add reload command to the menu
        m.addCommand('Reload modules', f'{full_package_name}.reload_classes("{classes_list}")')
