import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional

from devchat.message import Message


@dataclass
class OpenAIMessage(Message):
    role: str = None
    name: Optional[str] = None
    function_call: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self._validate_role():
            raise ValueError("Invalid role. Must be one of 'system', 'user', or 'assistant'.")

        if not self._validate_name():
            raise ValueError("Invalid name. Must contain a-z, A-Z, 0-9, and underscores, "
                             "with a maximum length of 64 characters.")

    def to_dict(self) -> dict:
        state = asdict(self)
        if state['name'] is None:
            del state['name']
        if state['role'] != "assistant":
            del state['function_call']
            
        return state
    
    def function_call_to_json(self):
        '''
        convert function_call to json
        function_call is like this:
        {
            "name": function_name,
            "arguments": '{"key": """value"""}'
        }
        '''
        if not self.function_call:
            return ''
        function_call_copy = self.function_call.copy()
        if 'arguments' in function_call_copy:
            # arguments field may be not a json string
            # we can try parse it by eval
            try:
                function_call_copy['arguments'] = eval(function_call_copy['arguments'])
            except Exception:
                # if it is not a json string, we can do nothing
                try:
                    function_call_copy['arguments'] = json.loads(function_call_copy['arguments'])
                except Exception:
                    pass
        return '\n```command\n' + json.dumps(function_call_copy) + '\n```\n'
        

    def stream_from_dict(self, message_data: dict) -> str:
        """Append to the message from a dictionary returned from a streaming chat API."""
        delta = message_data.get('content', '')
        if self.content:
            self.content += delta
        else:
            self.content = delta
                        
        return delta

    def _validate_role(self) -> bool:
        """Validate the role attribute.

        Returns:
            bool: True if the role is valid, False otherwise.
        """
        return self.role in ["system", "user", "assistant"]

    def _validate_name(self) -> bool:
        """Validate the name attribute.

        Returns:
            bool: True if the name is valid or None, False otherwise.
        """
        if self.name is None:
            return True
        if not self.name.strip():
            return False
        return len(self.name) <= 64 and self.name.replace("_", "").isalnum()
