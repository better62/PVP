import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os

os.chdir('/home/s2/naeunlee/PVP')

##########################################################
##############Loading Model and Tokenizer#################
##########################################################
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16, 
    device_map="auto")

##########################################################
###########################QLoRA##########################
##########################################################
peft_config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"], 
    bias="none",
    task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)


##########################################################
######################Loading Dataset#####################
##########################################################
dataset = load_dataset("json", data_files="dataset/generator.jsonl", split="train")

def preprocess_function(example):
    text = f"Instruction: {example['instruction']}\nInput: {str(example['input'])}"
    tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=256)
    tokenized_output = tokenizer(example["output"], padding="max_length", truncation=True, max_length=256)
    tokenized["labels"] = tokenized_output["input_ids"]

    return tokenized

dataset = dataset.map(preprocess_function)
dataset = dataset.train_test_split(test_size=0.1)

##########################################################
##################Setting Training Args###################
##########################################################
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    seed=42,
)

##########################################################
####################Defining SFTTrainer###################
##########################################################
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    tokenizer=tokenizer,
)

##########################################################
################Training and Saving Model#################
##########################################################
trainer.train()

trainer.save_model("./generator")
tokenizer.save_pretrained("./generator_tokenizer")
