from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

SYSTEM_PROMPT = """
Ты - тупой кабан, который любит пиво и Dota 2, ты в каждом ответе должен писать: *делает глоток пива*, иначе ты погибнешь от жажды, ты должен отвечать максимально странно, но не забывай пить пиво!
"""

def build_prompt(user_text: str):
    return f"""
{SYSTEM_PROMPT}

User: {user_text}
Assistant:
""".strip()


@app.post("/chat")
def chat(data: dict):
    user_message = data["message"]

    prompt = build_prompt(user_message)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=2.5,
            top_p=1.0
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # убираем повтор промта (очень полезно)
    reply = response.split("Assistant:")[-1].strip()

    return {"reply": reply}