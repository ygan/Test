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