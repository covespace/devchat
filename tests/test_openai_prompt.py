import json
import os
import pytest
from devchat.message import MessageType
from devchat.openai import OpenAIMessage
from devchat.openai import OpenAIPrompt
from devchat.utils import get_git_user_info


def test_prompt_init_and_set_response():
    name, email = get_git_user_info()
    prompt = OpenAIPrompt(model="gpt-3.5-turbo", user=name, email=email)
    assert prompt.model == "gpt-3.5-turbo"

    prompt.set_request("Where was the 2020 World Series played?")
    response_str = '''
    {
      "id": "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve",
      "object": "chat.completion",
      "created": 1677649420,
      "model": "gpt-3.5-turbo-0301",
      "usage": {"prompt_tokens": 56, "completion_tokens": 31, "total_tokens": 87},
      "choices": [
        {
          "message": {
            "role": "assistant",
            "content": "The 2020 World Series was played in Arlington, Texas."
          },
          "finish_reason": "stop",
          "index": 0
        }
      ]
    }
    '''
    prompt.set_response(response_str)

    assert prompt.timestamp == 1677649420
    assert prompt.request_tokens == 56
    assert prompt.response_tokens == 31
    assert len(prompt.responses) == 1
    assert prompt.responses[0].role == "assistant"
    assert prompt.responses[0].content == "The 2020 World Series was played in Arlington, Texas."


def test_prompt_model_mismatch():
    name, email = get_git_user_info()
    prompt = OpenAIPrompt(model="gpt-3.5-turbo", user=name, email=email)

    response_str = '''
    {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "\\n\\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10.",
            "role": "assistant"
          }
        }
      ],
      "created": 1677825456,
      "id": "chatcmpl-6ptKqrhgRoVchm58Bby0UvJzq2ZuQ",
      "model": "gpt-4",
      "object": "chat.completion",
      "usage": {
        "completion_tokens": 301,
        "prompt_tokens": 36,
        "total_tokens": 337
      }
    }
    '''
    with pytest.raises(ValueError):
        prompt.set_response(response_str)


@pytest.fixture(scope="module", name='responses')
def fixture_responses():
    current_dir = os.path.dirname(__file__)
    folder_path = os.path.join(current_dir, "stream_responses")
    stream_responses = []

    file_names = os.listdir(folder_path)
    sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0][8:]))

    for file_name in sorted_file_names:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            response = json.load(file)
            stream_responses.append(response)

    return stream_responses


def test_append_response(responses):
    name, email = get_git_user_info()
    prompt = OpenAIPrompt(model="gpt-3.5-turbo-0301", user=name, email=email)

    for response in responses:
        prompt.append_response(json.dumps(response))

    expected_messages = [
        OpenAIMessage(type=MessageType.CONTEXT, role='assistant', content='Tomorrow.'),
        OpenAIMessage(type=MessageType.CONTEXT, role='assistant', content='Tomorrow!')
    ]

    assert len(prompt.responses) == len(expected_messages)
    for index, message in prompt.responses.items():
        assert message.role == expected_messages[index].role
        assert message.content == expected_messages[index].content


def test_messages_empty():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    assert prompt.messages == []


def test_messages_instruct():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    instruct_message = OpenAIMessage(type=MessageType.INSTRUCT, role='system',
                                     content='Instructions')
    prompt.append_message(MessageType.INSTRUCT, 'Instructions')
    assert prompt.messages == [instruct_message.to_dict()]


def test_messages_context():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    context_message = OpenAIMessage(type=MessageType.CONTEXT, role='system', content='Context')
    prompt.append_message(MessageType.CONTEXT, 'Context')
    expected_message = context_message.to_dict()
    expected_message["content"] = "<context>" + context_message.content
    assert prompt.messages == [expected_message]


def test_messages_record():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    with pytest.raises(ValueError):
        prompt.append_message(MessageType.RECORD, 'Record')


def test_messages_request():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    request_message = OpenAIMessage(type=MessageType.RECORD, role='user', content='Request')
    prompt.set_request('Request')
    expected_message = request_message.to_dict()
    expected_message["content"] = "<request>" + expected_message["content"]
    assert prompt.messages == [expected_message]


def test_messages_combined():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    instruct_message = OpenAIMessage(type=MessageType.INSTRUCT, role='system',
                                     content='Instructions')
    context_message = OpenAIMessage(type=MessageType.CONTEXT, role='system', content='Context')
    request_message = OpenAIMessage(type=MessageType.RECORD, role='user', content='Request')

    prompt.append_message(MessageType.INSTRUCT, 'Instructions')
    prompt.append_message(MessageType.CONTEXT, 'Context')
    prompt.set_request('Request')

    expected_context_message = context_message.to_dict()
    expected_context_message["content"] = "<context>" + context_message.content

    expected_request_message = request_message.to_dict()
    expected_request_message["content"] = "<request>" + request_message.content

    assert prompt.messages == [
        instruct_message.to_dict(),
        expected_request_message,
        expected_context_message
    ]


def test_messages_invalid_append():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    with pytest.raises(KeyError):
        prompt.append_message('invalid', 'Instructions')


def test_messages_invalid_request():
    prompt = OpenAIPrompt(model="davinci-codex", user="John Doe", email="john.doe@example.com")
    with pytest.raises(ValueError):
        prompt.set_request("")
