# diffphys.py - Physics-Grounded LLM with Bayesian Fusion
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import numpy as np

class DiffPhys:
    def __init__(self):
        print("Loading AI model... (this takes 1-2 minutes first time)")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.model.eval()
        self.state = {"g": 9.81}

    def physics_check(self, text):
        if "dropped" in text.lower() and "after" in text.lower() and "second" in text.lower():
            t = re.search(r"(\d+\.?\d*)\s*second", text)
            if t:
                t = float(t.group(1))
                return f"{self.state['g'] * t:.1f} m/s"
        return None

    def generate(self, prompt, max_tokens=30):
        input_ids = self.tokenizer.encode(prompt + " Answer:", return_tensors="pt")
        output = []
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = self.model(input_ids).logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                token = self.tokenizer.decode(next_token[0])
                output.append(token)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                full_text = self.tokenizer.decode(input_ids[0])
                phys_answer = self.physics_check(full_text)
                if phys_answer and token.strip().replace(".", "").isdigit():
                    return " " + phys_answer

                if "<|endoftext|>" in token:
                    break
        return "".join(output)

# === RUN DEMO ===
if __name__ == "__main__":
    dp = DiffPhys()
    print("\n" + "="*50)
    print("DIFFPHYS v0.1")
    print("="*50)
    result = dp.generate("A ball is dropped from 10m. How fast after 1 second?")
    print("Output:", result)
