import torch
from contextlib import nullcontext


def process_text_embeddings(
    texts,
    processor,
    text_model,
    device: torch.device,
    model_name: str = "BLIP",
):
    text_input = processor(text=texts, return_tensors="pt", padding=True).to(device)
    context_manager = getattr(text_model, "no_sync", None)
    context = context_manager() if callable(context_manager) else nullcontext()

    with torch.no_grad(), context:
        if model_name.upper() == "BLIP":
            text_out = text_model.text_encoder(**text_input)
        else:
            text_out = text_model(**text_input)
    return text_out.last_hidden_state

