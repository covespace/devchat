from typing import Optional, Union, List, Dict, Iterator
from pydantic import BaseModel, Field, Extra
import openai
from devchat.chat import Chat
from devchat.utils import get_git_user_info
from .openai_prompt import OpenAIPrompt


class OpenAIChatConfig(BaseModel):
    """
    Configuration object for the OpenAIChat class.
    """
    model: str
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    n: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = Field(None)
    stop: Optional[Union[str, List[str]]] = Field(None)
    max_tokens: Optional[int] = Field(None, ge=1)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[int, float]] = Field(None)
    user: Optional[str] = Field(None)

    class Config:
        """
        Configuration class to forbid extra fields in the model.
        """
        extra = Extra.forbid


class OpenAIChat(Chat):
    """
    OpenAIChat class that handles communication with the OpenAI Chat API.
    """
    def __init__(self, config: OpenAIChatConfig):
        """
        Initialize the OpenAIChat class with a configuration object.

        Args:
            config (OpenAIChatConfig): Configuration object with parameters for the OpenAI Chat API.
        """
        self.config = config

    def init_prompt(self, request: str) -> OpenAIPrompt:
        """
        Initialize a prompt for the chat system.

        Args:
            request (str): The basic request of the prompt.
                           The returned prompt can be combined with more instructions and context.
        """
        user, email = get_git_user_info()
        prompt = OpenAIPrompt(model=self.config.model, user=user, email=email)
        prompt.set_request(request)
        return prompt

    def complete_response(self, prompt: OpenAIPrompt) -> str:
        """
        Retrieve a complete response JSON string from the chat system.

        Args:
            prompt (Prompt): A prompt of messages representing the conversation.
        Returns:
            str: A JSON string representing the complete response.
        """
        # Filter the config parameters with non-None values
        config_params = {
            key: value
            for key, value in self.config.dict().items() if value is not None
        }
        config_params['stream'] = False

        response = openai.ChatCompletion.create(
            messages=prompt.messages,
            **config_params
        )
        return response

    def stream_response(self, prompt: OpenAIPrompt) -> Iterator[str]:
        """
        Retrieve a streaming response as an iterator of JSON strings from the chat system.

        Args:
            prompt (Prompt): A prompt of messages representing the conversation.
        Returns:
            Iterator[str]: An iterator over JSON strings representing the streaming response events.
        """
        # Filter the config parameters with non-None values
        config_params = {
            key: value
            for key, value in self.config.dict().items() if value is not None
        }
        config_params['stream'] = True

        response = openai.ChatCompletion.create(
            messages=prompt.messages,
            **config_params
        )
        return response
