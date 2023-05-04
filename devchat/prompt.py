from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from typing import Dict, List
from devchat.message import MessageType, Message
from devchat.utils import unix_to_local_datetime


@dataclass
class Prompt(ABC):
    """
    A class to represent a prompt and its corresponding responses from the chat API.

    Attributes:
        model (str): The name of the language model.
        user_name (str): The name of the user.
        user_email (str): The email address of the user.
        _request (Message): The request message.
        _messages (Dict[MessageType, Message]): The messages indexed by the message type.
        _responses (Dict[int, Message]): The responses indexed by an integer.
        _timestamp (int): The timestamp when the response was created.
        _request_tokens (int): The number of tokens used in the request.
        _response_tokens (int): The number of tokens used in the response.
        _hashes (Dict[int, str]): The hashes of the prompt indexed by the response index.
        parents (List[str]): The hashes of the parent prompts.
        references (List[str]): The hashes of the referenced prompts.
    """
    model: str
    user_name: str
    user_email: str
    _timestamp: int = field(init=False, default=None)
    _request: Message = field(init=False, default=None)
    _messages: Dict[MessageType, Message] = field(init=False, default_factory=lambda: {
        MessageType.INSTRUCT: [],
        MessageType.CONTEXT: [],
        MessageType.RECORD: []})
    _responses: Dict[int, Message] = field(init=False, default_factory=dict)
    _request_tokens: int = field(init=False, default=None)
    _response_tokens: int = field(init=False, default=None)
    _hash: str = field(init=False, default=None)
    parents: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    @abstractmethod
    def messages(self) -> List[dict]:
        """
        List of messages in the prompt to be sent to the chat API.
        """

    @property
    def responses(self) -> Dict[int, Message]:
        return self._responses

    @property
    def request_tokens(self) -> int:
        return self._request_tokens

    @property
    def response_tokens(self) -> int:
        return self._response_tokens

    @property
    def hash(self) -> str:
        return self._hash

    @abstractmethod
    def append_message(self, message_type: MessageType, content: str):
        """
        Append a message to the prompt.

        Args:
            message_type (MessageType): The type of the message. It cannot be RECORD.
            content (str): The content of the message.
        """

    @abstractmethod
    def set_request(self, content: str):
        """
        Set the request message for the prompt.

        Args:
            content (str): The request content to set.
        """

    @abstractmethod
    def set_response(self, response_str: str):
        """
        Parse the API response string and set the Prompt object's attributes.

        Args:
            response_str (str): The JSON-formatted response string from the chat API.
        """

    @abstractmethod
    def append_response(self, delta_str: str) -> str:
        """
        Append the content of a streaming response to the existing messages.

        Args:
            delta_str (str): The JSON-formatted delta string from the chat API.

        Returns:
            str: The delta content with index 0. None when the response is over.
        """

    def set_hash(self):
        """Set the hash of the prompt."""
        if not self._request or not self._responses:
            raise ValueError("Prompt is incomplete for hash.")
        hash_str = self._request.content
        for response in self._responses.values():
            hash_str += response.content
        self._hash = hashlib.sha1(hash_str.encode()).hexdigest()
        return self._hash

    def formatted_header(self) -> str:
        """Formatted string header of the prompt."""
        formatted_str = f"User: {self.user_name} <{self.user_email}>\n"

        local_time = unix_to_local_datetime(self._timestamp)
        formatted_str += f"Date: {local_time.strftime('%a %b %d %H:%M:%S %Y %z')}\n\n"

        return formatted_str

    def formatted_response(self, index: int) -> str:
        """Formatted response of the prompt."""
        formatted_str = self.formatted_header()

        response = self._responses.get(index, None)
        if response is None or response.content is None:
            raise ValueError(f"Response {index} is incomplete.")

        formatted_str += response.content.strip() + "\n\n"
        formatted_str += f"prompt {self.hash}"

        return formatted_str

    def shortlog(self) -> List[dict]:
        """Generate a shortlog of the prompt."""
        if not self._request or not self._responses:
            raise ValueError("Prompt is incomplete for shortlog.")
        logs = []
        for response in self._responses.values():
            shortlog_data = {
                "user": f'{self.user_name} <{self.user_email}>',
                "date": self._timestamp,
                "last_message": self._request.content,
                "response": response.content,
                "hash": self.hash
            }
            logs.append(shortlog_data)
        return logs
