from dataclasses import dataclass
from typing import Optional
from devchat.message import MessageType, Message


@dataclass
class OpenAIMessage(Message):
    """A class to represent a message in a conversation with OpenAI.

    Attributes:
        role (str): The role of the author of the message. One of 'system', 'user', or 'assistant'.
        name (str, optional): The name of the author of the message. May contain a-z, A-Z, 0-9, and
                              underscores, with a maximum length of 64 characters.
    """
    name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.role not in ["system", "user", "assistant"]:
            raise ValueError("Invalid role. Must be one of 'system', 'user', or 'assistant'.")
        if self.name is not None:
            self.name = self.name.strip()
            if not self.name or len(self.name) > 64 or not self.name.replace("_", "").isalnum():
                raise ValueError("Invalid name. Must contain a-z, A-Z, 0-9, and underscores, "
                                 "with a maximum length of 64 characters.")

    @classmethod
    def from_dict(cls, message_type: MessageType, message_data: dict) -> 'OpenAIMessage':
        """Construct a Message instance from a dictionary.

        Args:
            type (MessageType): The type of the message.
            message_data (Dict): A dictionary containing the message data with keys 'role',
                                 'content' (empty when it is a delta), and an optional 'name'.

        Returns:
            Message: A new Message instance with the attributes set from the dictionary.
        """
        return cls(type=message_type, **message_data)

    def append_from_dict(self, message_data: dict) -> str:
        """Append to the message from a dictionary returned from a chat API."""
        delta = message_data.get('content', '')
        self.content += delta
        return delta

    def to_dict(self) -> dict:
        """Convert the Message object to a dictionary for calling OpenAI APIs.

        Returns:
            dict: A dictionary representation of the Message object.
        """
        message_dict = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            message_dict["name"] = self.name
        return message_dict
