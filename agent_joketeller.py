from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import pipeline
import os

# 1) Load your token (added in 01_01)
load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise SystemExit("❌ No Hugging Face token found. Add HF_TOKEN=... to your .env")
login(token=token)
print("✅ Logged in to Hugging Face Hub successfully!")


# 2) Pick a small model that runs fine on CPU
# MODEL_ID = "google/flan-t5-base"
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# 3) Build a text-generation pipeline (CPU mode to avoid Mac MPS issues)
pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


# 4) A tiny prompt template: 1 line, setup — punchline
TEMPLATE = (
    "You are a renowed joke-teller, famous to be hilarious."
    "Write one clean, original one-line joke about {topic}. "
    "Format: Setup — Punchline. Keep it under 25 words and answer in English."
)


def make_joke(topic="computers"):
    prompt = TEMPLATE.format(topic=topic)
    result = pipe(
        prompt,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        max_new_tokens=40,
        return_full_text=False,
    )[0]["generated_text"]
    return result
    # return result.splitlines()[0].strip()


if __name__ == "__main__":
    print("🤖 Joke-Teller ready! Type a topic, or 'quit' to stop.\n")
    while True:
        try:
            topic = input("Topic: ").strip()
            if topic.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            if not topic:
                topic = "computers"
            print(f"\n😂 {make_joke(topic)}\n")
        except KeyboardInterrupt:
            print("\n(Stopped) Goodbye!")
            break
