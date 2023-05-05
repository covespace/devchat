from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MessageType(Enum):
    INSTRUCT = "instruct"
    RECORD = "record"
    CONTEXT = "context"


@dataclass
class Message(ABC):
    type: MessageType
    role: str
    content: Optional[str] = ""

    def __post_init__(self):
        if not isinstance(self.type, MessageType):
            raise ValueError(f"Invalid message type: {self.type}")

    @classmethod
    @abstractmethod
    def from_dict(cls, message_type: MessageType, message_data: dict) -> "Message":
        """
        Construct a Message instance from a dictionary returned from a chat API.
        """

    @abstractmethod
    def append_from_dict(self, message_data: dict) -> str:
        """
        Append to the message from a dictionary returned from a chat API.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the Message object to a dictionary for calling a chat API.
        """
