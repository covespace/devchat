import datetime
import getpass
import json
import os
import re
import socket
import subprocess
from typing import List, Tuple
from dateutil import tz
import pytz
from devchat.message import MessageType


class MessageTypeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, MessageType):
            return o.value
        return super().default(o)

    def encode(self, o):
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {self.default(k) if isinstance(k, MessageType) else k: convert_keys(v)
                        for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            return obj

        return super().encode(convert_keys(o))


def find_git_root():
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        return root.decode("utf-8").strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Not inside a Git repository") from error


def git_ignore(git_root_dir, ignore_entry):
    gitignore_path = os.path.join(git_root_dir, '.gitignore')

    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as gitignore_file:
            gitignore_content = gitignore_file.read()

        if ignore_entry not in gitignore_content:
            with open(gitignore_path, 'a', encoding='utf-8') as gitignore_file:
                gitignore_file.write(f'\n# DevChat\n{ignore_entry}\n')
    else:
        with open(gitignore_path, 'w', encoding='utf-8') as gitignore_file:
            gitignore_file.write(f'# DevChat\n{ignore_entry}\n')


def unix_to_local_datetime(unix_time) -> datetime.datetime:
    # Get the local time zone
    local_tz = tz.tzlocal()

    # Convert the Unix time to a naive datetime object
    naive_dt = datetime.datetime.utcfromtimestamp(unix_time)

    # Localize the naive datetime object to the local time zone
    local_dt = naive_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)

    return local_dt


def get_git_user_info() -> Tuple[str, str]:
    try:
        cmd = ['git', 'config', 'user.name']
        git_user_name = subprocess.check_output(cmd).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_user_name = getpass.getuser()

    try:
        cmd = ['git', 'config', 'user.email']
        git_user_email = subprocess.check_output(cmd).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_user_email = git_user_name + '@' + socket.gethostname()

    return git_user_name, git_user_email


def parse_files(file_paths_str) -> List[str]:
    if not file_paths_str:
        return []

    file_paths = file_paths_str.split(',')

    for file_path in file_paths:
        expanded_file_path = os.path.expanduser(file_path)
        if not os.path.isfile(expanded_file_path):
            raise ValueError(f"File {file_path} does not exist.")

    contents = []
    for file_path in file_paths:
        expanded_file_path = os.path.expanduser(file_path)
        with open(expanded_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content:
                raise ValueError(f"File {file_path} is empty.")
            contents.append(content)
    return contents


def is_valid_hash(hash_str):
    """Check if a string is a valid hash value."""
    # Hash values are usually alphanumeric with a fixed length
    # depending on the algorithm used to generate them
    pattern = re.compile(r'^[a-fA-F0-9]{40}$')  # Example pattern for SHA-1 hash
    return bool(pattern.match(hash_str))


def parse_hashes(hashes) -> List[str]:
    if not hashes:
        return []
    values = []
    for value in hashes.split(','):
        if not is_valid_hash(value):
            raise ValueError(f"Invalid hash value {value}.")
        values.append(value)
    return values


def update_dict(dict_to_update, key, value) -> dict:
    """
    Update a dictionary with a key-value pair and return the dictionary.
    """
    dict_to_update[key] = value
    return dict_to_update


def store_to_git(dict_object: dict) -> str:
    # Serialize the dictionary as a JSON string
    json_data = json.dumps(dict_object, cls=MessageTypeEncoder)

    # Store the JSON string as a Git blob object
    with subprocess.Popen(
        ["git", "hash-object", "-w", "--stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as git_hash_object_process:
        git_hash_object_output, git_hash_object_error = git_hash_object_process.communicate(
            input=json_data.encode("utf-8")
        )

    if git_hash_object_error:
        raise RuntimeError(f"Error storing Git object: {git_hash_object_error.decode('utf-8')}")

    # Get the hash of the stored Git object
    git_object_hash = git_hash_object_output.decode("utf-8").strip()

    # Create a Git note referencing the stored Git object
    with subprocess.Popen(
        ["git", "notes", "--ref", "devchat", "add", "-m", "store", git_object_hash],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as git_notes_process:
        _, git_notes_error = git_notes_process.communicate()

    if git_notes_error:
        raise RuntimeError(f"Error creating Git note: {git_notes_error.decode('utf-8')}")

    return git_object_hash


def retrieve_from_git(sha1) -> dict:
    result = subprocess.run(
        ['git', 'cat-file', '-p', sha1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    serialized_object = result.stdout.decode('utf-8')
    return json.loads(serialized_object)
