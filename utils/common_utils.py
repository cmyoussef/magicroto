import ast
import json
import os
import re
import time
from fnmatch import fnmatch


# region check file
class FileLockException(Exception):
    pass


def acquire_lock(file_path, timeout=1):
    lock_file = file_path + ".lock"
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.1)
    return False


def release_lock(file_path):
    lock_file = file_path + ".lock"
    try:
        os.remove(lock_file)
    except FileNotFoundError:
        pass


def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0


def check_file_complete(file_path):
    if os.path.exists(file_path):
        if is_file_empty(file_path):
            return False
        lock_acquired = acquire_lock(file_path)
        if lock_acquired:
            release_lock(file_path)
            return True
        else:
            return False
    return False


# endregion


def list_bin_files(directory):
    result = {}
    if not os.path.isdir(directory):
        # print(f'{directory} Dose not exists')
        return result
    # Traverse directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check for .bin files
            if file.endswith('.bin') or file.endswith('.safetensors'):
                folder_name = os.path.basename(root)  # Get folder name
                weight_name = None if file == "pytorch_lora_weights.bin" else os.path.basename(file)
                # Add details to the dictionary
                result[folder_name] = {"dir": root, "weight_name": weight_name}

    return result


def find_model_files(directory):
    """
    Searches for .safetensors and .ckpt files in a given directory and its subdirectories.

    :param directory: The root directory to start searching from.
    :return: Dictionary containing the filenames and their full paths of .safetensors and .ckpt files.
    """
    model_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.safetensors') or file.endswith('.ckpt'):
                model_files[file] = os.path.join(root, file)
    return model_files


def list_models_in_directory(cache_dir, target='model_index.json'):
    models_dir = {}
    if not os.path.isdir(cache_dir):
        print(f'{cache_dir} Dose not exists')
        return models_dir
    # Walking through the directory
    for dir_name in os.listdir(cache_dir):

        full_dir_path = os.path.join(cache_dir, dir_name)
        if not os.path.isdir(full_dir_path):
            continue

        contents = os.listdir(full_dir_path)
        if target in contents:
            models_dir[dir_name] = full_dir_path
        elif 'refs' in contents and 'snapshots' in contents:
            refs_dir_path = os.path.join(full_dir_path, 'refs')
            if 'main' in os.listdir(refs_dir_path):
                with open(os.path.join(refs_dir_path, 'main'), 'r') as file:
                    folder_name = file.read().strip()
                    snapshots_dir_path = os.path.join(full_dir_path, 'snapshots')
                    if folder_name in os.listdir(snapshots_dir_path):
                        full_dir_model_path = os.path.join(snapshots_dir_path, folder_name)
                        model_index_path = os.path.join(snapshots_dir_path, folder_name, target)
                        if os.path.exists(model_index_path):
                            models_dir[dir_name] = full_dir_model_path
    return models_dir


# region Serialize data manually
def dict_to_str(d):
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            v_str = f"dict:{dict_to_str(v)}"
        elif isinstance(v, list):
            if all(isinstance(sub_item, list) for sub_item in v):
                v_str = f"list:{';'.join([','.join(map(str, sub_list)) for sub_list in v])}"
            else:
                v_str = f"list:{','.join(map(str, v))}"
        else:
            v_str = f"scalar:{str(v)}"
        items.append(f"{k}={v_str}")
    return "|".join(items)


def str_to_dict(s):
    d = {}

    if not s:
        return d

    for item in s.split("|"):
        k, v_str = item.split("=", 1)
        v_type, v_value = v_str.split(":", 1)
        if v_type == "dict":
            v = str_to_dict(v_value)
        elif v_type == "list":
            if v_value:  # Check if v_value is not empty
                if ";" in v_value or "(" in v_value:
                    # Use regular expression to find all pairs of numbers
                    v = [list(map(int, match.groups())) for match in re.finditer(r'\((\d+),\s*(\d+)\)', v_value)]
                else:
                    # Split by comma and convert to integers
                    v = list(map(int, v_value.split(",")))
            else:
                v = []  # Assign an empty list if v_value is empty
        elif v_type == "scalar":
            # Convert scalar values to number if possible, else keep as string
            v = convert_to_number(v_value)
        d[k] = v
    return d


def convert_to_number(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def get_dict_type(inp):
    if isinstance(inp, dict):
        return inp
    # TODO not to use str to dict in finally and write an error
    try:
        inp = json.loads(inp)
    except:
        inp = ast.literal_eval(inp)
    return inp


def extract_number(string):
    # Find the first number in the string
    match = re.search(r'\d+\.\d+|\d+', string)

    # If a number was found, convert it to a float, round it, and convert it to an integer
    if match:
        number = float(match.group())
        # rounded_number = round(number)
        return number

    # If no number was found, return None
    else:
        return -1


# endregion


def match_dict_keys(dictionary, wildcard_list, neg_wildcard=None, convert_model_id=True):
    if not wildcard_list and not neg_wildcard:
        return dictionary

    matched_dict = {}
    if wildcard_list:
        if isinstance(wildcard_list, str):
            wildcard_list = [wildcard_list]

        if convert_model_id:
            wildcard_list = [get_model_name_from_id(m) for m in wildcard_list]

        for key in dictionary.keys():
            for wildcard in wildcard_list:
                if fnmatch(key.lower(), wildcard):
                    matched_dict[key] = dictionary[key]
                    break  # Skip remaining wildcards for this key
    else:
        matched_dict = dictionary

    if neg_wildcard:
        if isinstance(neg_wildcard, str):
            neg_wildcard = [neg_wildcard]
        matching_neg = match_dict_keys(matched_dict.copy(), neg_wildcard)
        for k in matching_neg:
            matched_dict.pop(k)
    return matched_dict


def get_model_name_from_id(model_id):
    if '/' in model_id:
        return f'models--{"--".join(model_id.split("/"))}'
    return model_id


def parse_metadata_string(img):
    metadata_string = img.info.get('parameters', '')

    parts = metadata_string.split('\n')
    data = {'prompt': parts[0]}
    parameters = parts[-1]
    if len(parts) == 3:
        data['negative_prompt'] = parts[1].strip('Negative prompt: ')

    # Find matches for "<word>:"
    keys = re.findall(r'([\w\s]+):\s', parameters)
    for i, k in enumerate(keys):
        keys[i] = k[1:] if k.startswith(' ') else k

    # Split string by "<word>:" to get values
    values = re.split(r'[\w\s]+:\s', parameters)

    # Remove leading/trailing whitespaces from each value
    values = [value.strip() for value in values if value]

    # Pair keys and values in a dictionary
    parameters_dict = {k: v for k, v in zip(keys, values)}
    data.update(parameters_dict)

    return data
