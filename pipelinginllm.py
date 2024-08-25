from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load the correct tokenizer for LLaMA
#tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf"#)

# Load the model with offloading support
with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        use_auth_token="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",
        load_in_4bit=True
    )

# Define the device map to offload parts of the model to CPU
device_map = infer_auto_device_map(
    model,
    max_memory={"cpu": "30GB", 0: "10GB"}
)

# Dispatch the model according to the device map
model = dispatch_model(model, device_map=device_map)

# Prepare input
input_text = "How is offloading helpful?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate output without gradient calculation
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id
    )

# Decode and print the result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))