"""
This module contains the main function for the DevChat CLI.
"""


import os
import time
from typing import Optional
import json
import sys
from contextlib import contextmanager
import rich_click as click
from devchat.store import Store
from devchat.openai import OpenAIPrompt
from devchat.openai import OpenAIChatConfig, OpenAIChat
from devchat.assistant import Assistant
from devchat.utils import find_git_root, git_ignore, get_git_user_info, parse_files


click.rich_click.USE_MARKDOWN = True


@click.group()
def main():
    """DevChat CLI: A command-line interface for the DevChat chatbot."""


@contextmanager
def handle_errors():
    """Handle errors in the CLI."""
    try:
        yield
    except Exception as error:
        click.echo(f"Error: {error}", err=True)
        sys.exit(os.EX_SOFTWARE)


def load_config_data(chat_dir: str) -> dict:
    default_config_data = {
        'llm': 'OpenAI',
        'OpenAI': {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.2
        }
    }

    try:
        with open(os.path.join(chat_dir, 'config.json'), 'r', encoding='utf-8') as file:
            config_data = json.load(file)
    except FileNotFoundError:
        config_data = default_config_data

    return config_data


@main.command()
@click.argument('content', required=False)
@click.option('-p', '--parent', help='Input the previous prompt hash to continue the conversation.')
@click.option('-r', '--reference', help='Input one or more specific previous prompt hashes to '
              'include in the current prompt.')
@click.option('-i', '--instruct', help='Add one or more files to the prompt as instructions.')
@click.option('-c', '--context', help='Add one or more files to the prompt as a context.')
def prompt(content: Optional[str], parent: Optional[str], reference: Optional[str],
           instruct: Optional[str], context: Optional[str]):
    """
    Main function to run the chat application.

    This function initializes the chat system based on the specified large language model (LLM),
    and performs interactions with the user by sending prompts and retrieving responses.

    Examples
    --------

    To send a single-line message to the LLM, provide the content as an argument:

    ```bash
    devchat prompt "What is the capital of France?"
    ```

    To send a multi-line message to the LLM, use the here-doc syntax:

    ```bash
    devchat prompt << 'EOF'
    What is the capital of France?
    Can you tell me more about its history?
    EOF
    ```

    Note the quotes around EOF in the first line, to prevent the shell from expanding variables.

    Configuration
    -------------

    The DevChat CLI reads its configuration from a file `.chatconfig.json` in the current directory.
    If the file is not found, it uses the following default configuration:
    ```json
    {
        "llm": "OpenAI",
        "OpenAI": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.2
        }
    }
    ```

    To customize the configuration, create a `.chatconfig.json` file in the current directory and
    modify the settings as needed. We recoommend the following settings:
    ```json
    {
        "llm": "OpenAI",
        "OpenAI": {
            "model": "gpt-4",
            "temperature": 0.2,
            "stream": true
        }
    }
    ```

    Note: To use OpenAI's APIs, you must have an API key to run the CLI.
    Run the following command line with your API key:

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

    """
    git_root = find_git_root()
    chat_dir = os.path.join(git_root, ".chat")
    if not os.path.exists(chat_dir):
        os.makedirs(chat_dir)

    config_data = load_config_data(chat_dir)

    with handle_errors():
        if content is None:
            content = click.get_text_stream('stdin').read()

        if content == '':
            return

        instruct_contents = parse_files(instruct)
        context_contents = parse_files(context)

        store_path = os.path.join(chat_dir, 'store.graphml')
        store = Store(store_path)
        git_ignore(git_root, '.chat/store.graphml')

        llm = config_data.get('llm')
        if llm == 'OpenAI':
            openai_config = OpenAIChatConfig(**config_data['OpenAI'])
            chat = OpenAIChat(openai_config)

            openai_asisstant = Assistant(chat, store)
            openai_asisstant.make_prompt(content, instruct_contents, context_contents,
                                         parent, reference)

            for response in openai_asisstant.iterate_responses():
                click.echo(response, nl=False)
        else:
            click.echo(f"Error: Invalid LLM in configuration '{llm}'", err=True)
            sys.exit(os.EX_DATAERR)


@main.command()
@click.option('--skip', default=0, help='Skip number prompts before showing the prompt history.')
@click.option('--max-count', default=100, help='Limit the number of commits to output.')
def log(skip, max_count):
    """
    Show the prompt history.
    """
    # Implement the logic to display the prompt history based on the `skip` and `max_count` options.
    logs = []
    for index in range(skip, skip + max_count):
        name, email = get_git_user_info()
        openai_prompt = OpenAIPrompt(model="gpt-3.5-turbo", user=name, email=email)
        openai_prompt.set_request(f"Prompt {index}")
        response = {
            "model": "gpt-3.5-turbo-0301",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Response {index}",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        }
        openai_prompt.set_response(json.dumps(response))
        logs = openai_prompt.shortlog('38645242908d4f379f13e6ca886523526d9b28ab') + logs
    click.echo(json.dumps(logs))
