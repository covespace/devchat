import json
from typing import List
from dataclasses import dataclass, field
from devchat.prompt import Prompt
from devchat.message import MessageType
from devchat.utils import update_dict
from .openai_message import OpenAIMessage


@dataclass
class OpenAIPrompt(Prompt):
    """
    A class to represent a prompt and its corresponding responses from OpenAI APIs.
    """
    _id: str = field(init=False, default=None)

    @property
    def id(self) -> str:
        return self._id

    @property
    def messages(self) -> List[dict]:
        combined_messages = []
        if self._messages[MessageType.INSTRUCT]:
            combined_messages += [msg.to_dict() for msg in self._messages[MessageType.INSTRUCT]]
        if self._request:
            combined_messages += [update_dict(self._request.to_dict(), 'content',
                                              '<request>' + self._request.content)]
        if self._messages[MessageType.CONTEXT]:
            combined_messages += [update_dict(msg.to_dict(), 'content', '<context>' + msg.content)
                                  for msg in self._messages[MessageType.CONTEXT]]
        if self._messages[MessageType.RECORD]:
            combined_messages += [msg.to_dict() for msg in self._messages[MessageType.RECORD]]
        return combined_messages

    def append_message(self, message_type: MessageType, content: str):
        """
        Append a message to the prompt.

        Args:
            message_type (MessageType): The type of the message. It cannot be RECORD.
            content (str): The content of the message.
        """
        if message_type == MessageType.RECORD:
            raise ValueError("Use set_request() to set a message of type RECORD.")
        self._messages[message_type].append(OpenAIMessage(type=message_type, role='system',
                                                          content=content))

    def set_request(self, content: str):
        if not content.strip():
            raise ValueError("The request cannot be empty.")
        self._request = OpenAIMessage(type=MessageType.RECORD, role='user', content=content)

    def set_response(self, response_str: str):
        """
        Parse the API response string and set the Prompt object's attributes.

        Args:
            response_str (str): The JSON-formatted response string from the chat API.
        """
        response_data = json.loads(response_str)
        self._validate_model(response_data)
        self._timestamp_from_dict(response_data)
        self._id_from_dict(response_data)

        self._request_tokens = response_data['usage']['prompt_tokens']
        self._response_tokens = response_data['usage']['completion_tokens']

        self._responses = {
            choice['index']: OpenAIMessage.from_dict(MessageType.RECORD, choice['message'])
            for choice in response_data['choices']
        }

    def append_response(self, delta_str: str) -> str:
        """
        Append the content of a streaming response to the existing messages.

        Args:
            delta_str (str): The JSON-formatted delta string from the chat API.

        Returns:
            str: The delta content with index 0. None when the response is over.
        """
        response_data = json.loads(delta_str)
        self._validate_model(response_data)
        self._timestamp_from_dict(response_data)
        self._id_from_dict(response_data)

        delta_content = ''
        for choice in response_data['choices']:
            delta = choice['delta']
            index = choice['index']

            if index not in self.responses:
                self.responses[index] = OpenAIMessage.from_dict(MessageType.RECORD, delta)
                if index == 0:
                    delta_content = self.formatted_header()
                    delta_content += self.responses[0].content
            else:
                if index == 0:
                    delta_content = self.responses[0].append_from_dict(delta)
                else:
                    self.responses[index].append_from_dict(delta)

        return delta_content

    def _validate_model(self, response_data: dict):
        if not response_data['model'].startswith(self.model):
            raise ValueError(f"Model mismatch: expected '{self.model}', "
                             f"got '{response_data['model']}'")

    def _timestamp_from_dict(self, response_data: dict):
        if self._timestamp is None:
            self._timestamp = response_data['created']
        elif self._timestamp != response_data['created']:
            raise ValueError(f"Time mismatch: expected {self._timestamp}, "
                             f"got {response_data['created']}")

    def _id_from_dict(self, response_data: dict):
        if self._id is None:
            self._id = response_data['id']
        elif self._id != response_data['id']:
            raise ValueError(f"ID mismatch: expected {self._id}, "
                             f"got {response_data['id']}")
