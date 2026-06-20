from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

messages = [
    {"role": "user", "content": "What is 17*19? Return only the integer."}
]

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    max_completion_tokens=2048,
    extra_body={
        "chat_template_kwargs": {
            "thinking": True,
            "reasoning_effort": "high",
        }
    },
)

msg = resp.choices[0].message

print("reasoning:", getattr(msg, "reasoning", None))
print("reasoning_content:", getattr(msg, "reasoning_content", None))
print("model_extra:", getattr(msg, "model_extra", None))
print("content:", msg.content)

class vLLM(DeepSeek):
    def __init__(self, name="deepseek-v4-flash", cache=None, **kwargs) -> None:
        kwargs.setdefault("api_key", os.environ.get("VLLM_API_KEY", "EMPTY"))
        if "base_url" not in kwargs:
            kwargs.setdefault("base_url", os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
        self.token_auto_expand = kwargs.get("token_auto_expand", True)
        super().__init__(name, cache, **kwargs)


    def _chat_completion(self, message, stop, seed, model_name, reasoning_effort, max_tokens, json_format, temperature):
        if self.token_auto_expand:
            space_count = 0
            for m in message:
                if isinstance(m["content"], str):
                    space_count += m["content"].count(" ")
            max_tokens = space_count*4 if space_count*4 > max_tokens else max_tokens
            for limit in [9888, 15888, 19888, 25888, 29888, 35888, 39888, 45888, 49888, 55888, 59888, 65888, 69888, 75888, 79888, 85888, 89888]:
                if max_tokens <= limit:
                    max_tokens = limit
                    break