
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load the correct tokenizer for LLaMA
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Load the model with offloading support
with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        use_auth_token=tokenizer ,
        load_in_4bit=True
    )

# Define the device map to offload parts of the model to CPU
device_map = infer_auto_device_map(
    model,
    max_memory={"cpu": "30GB", 0: "10GB"}
)

# Dispatch the model according to the device map
model = dispatch_model(model, device_map=device_map)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset, #Give path to the Medical dataset
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

logging.set_verbosity(logging.CRITICAL)

prompt = "how to cure feaver?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])



# for chatbot we use Chainlit so the code for that will be
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    await cl.Message(content=answer).send()
