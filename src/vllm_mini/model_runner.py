import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelRunner:
    def __init__(self, model_name_or_path: str, device: str | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"].to(self.device)
    
    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids)
            return outputs.logits
    