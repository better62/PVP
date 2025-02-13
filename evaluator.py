import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# 모델 및 토크나이저 로드
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16, 
    device_map="auto")

# QLoRA 설정
peft_config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"], 
    bias="none",
    task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)

# 데이터셋 로드 (예제 데이터셋 사용, 필요시 변경)
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)

dataset = dataset.train_test_split(test_size=0.1)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    seed=42,
)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    tokenizer=tokenizer,
)

# 학습 실행
trainer.train()

# 모델 저장
trainer.save_model("./llama3-8b-sft")
tokenizer.save_pretrained("./llama3-8b-sft")
