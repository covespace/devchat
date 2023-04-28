import pytest
from devchat.message import MessageType
from devchat.openai import OpenAIMessage


def test_valid_message_creation():
    message = OpenAIMessage(type=MessageType.CONTEXT, role="user", content="Hello, World!")
    assert message.role == "user"
    assert message.content == "Hello, World!"
    assert message.name is None


def test_valid_message():
    message = OpenAIMessage(MessageType.EXAMPLE, "user", "Hello, World!", "John_Doe")
    assert message.to_dict() == {"role": "user", "content": "Hello, World!", "name": "John_Doe"}


def test_invalid_role():
    with pytest.raises(ValueError):
        OpenAIMessage(MessageType.INSTRUCTION, "invalid_role", "Hello, World!")


def test_invalid_type():
    with pytest.raises(ValueError):
        OpenAIMessage("invalid_type", "user", "Hello, World!")


def test_none_content():
    message = OpenAIMessage(type=MessageType.INSTRUCTION, role="system", content=None)
    assert message.content is None


def test_invalid_name():
    with pytest.raises(ValueError):
        OpenAIMessage(MessageType.CONTEXT, "user", "Hello, World!", "Invalid@Name")


def test_empty_name():
    with pytest.raises(ValueError):
        OpenAIMessage(MessageType.CONTEXT, "user", "Hello, World!", "")


def test_blank_name():
    with pytest.raises(ValueError):
        OpenAIMessage(MessageType.CONTEXT, "user", "Hello, World!", "  ")


def test_none_name():
    message = OpenAIMessage(MessageType.EXAMPLE, "user", "Hello, World!", None)
    assert message.to_dict() == {"role": "user", "content": "Hello, World!"}


def test_from_dict():
    message_data = {
        "content": "Welcome to the chat.",
        "role": "system"
    }
    message = OpenAIMessage.from_dict(MessageType.INSTRUCTION, message_data)
    assert message.type == MessageType.INSTRUCTION
    assert message.role == "system"
    assert message.content == "Welcome to the chat."
    assert message.name is None


def test_from_dict_with_name():
    message_data = {
        "content": "Hello, Assistant!",
        "role": "user",
        "name": "JohnDoe"
    }
    message = OpenAIMessage.from_dict(MessageType.CONTEXT, message_data)
    assert message.type == MessageType.CONTEXT
    assert message.role == "user"
    assert message.content == "Hello, Assistant!"
    assert message.name == "JohnDoe"