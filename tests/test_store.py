from devchat.openai import OpenAIPrompt
from devchat.store import Store
from devchat.utils import get_git_user_info


def test_get_prompt(tmp_path):
    store = Store(tmp_path / "store.graphml")
    name, email = get_git_user_info()
    prompt = OpenAIPrompt(model="gpt-3.5-turbo", user=name, email=email)
    prompt.set_request("Where was the 2020 World Series played?")
    response_str = '''
    {
      "id": "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve",
      "object": "chat.completion",
      "created": 1577649420,
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
    prompt_hash = store.store_prompt(prompt)

    assert store.get_prompt(prompt_hash)['timestamp'] == prompt.timestamp


def test_select_recent(tmp_path):
    store = Store(tmp_path / "store.graphml")
    name, email = get_git_user_info()

    # Create and store 5 prompts
    hashes = []
    for index in range(5):
        prompt = OpenAIPrompt(model="gpt-3.5-turbo", user=name, email=email)
        prompt.set_request(f"Question {index}")
        response_str = f'''
        {{
          "id": "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve",
          "object": "chat.completion",
          "created": 167764942{index},
          "model": "gpt-3.5-turbo-0301",
          "usage": {{"prompt_tokens": 56, "completion_tokens": 31, "total_tokens": 87}},
          "choices": [
            {{
              "message": {{
                "role": "assistant",
                "content": "Answer {index}"
              }},
              "finish_reason": "stop",
              "index": 0
            }}
          ]
        }}
        '''
        prompt.set_response(response_str)
        prompt_hash = store.store_prompt(prompt)
        hashes.append(prompt_hash)

    # Test selecting recent prompts
    recent_prompts = store.select_recent(0, 3)
    assert len(recent_prompts) == 3
    for index, prompt_hash in enumerate(recent_prompts):
        assert prompt_hash == hashes[4 - index]
