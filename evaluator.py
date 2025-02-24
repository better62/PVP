import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score
import os

os.chdir('/home/s2/naeunlee/PVP')
torch.cuda.empty_cache()

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
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.1)


##########################################################
################Defining Persuasive Loss##################
##########################################################
class PersuasiveLoss(torch.nn.Module):
    def __init__(self):
        super(PersuasiveLoss, self).__init__()

    def forward(self, predictions, labels):
        probabilities = F.softmax(predictions, dim=-1)
        predicted_ids = torch.argmax(probabilities, dim=-1)

        mask = labels != -100
        masked_pred = predicted_ids[mask]
        masked_labels = labels[mask]

        spearman_loss = 1 - self.spearman_loss(masked_pred, masked_labels)  
        pearson_loss = 1 - self.pearson_loss(masked_pred, masked_labels)    
        ndcg_loss = 1 - self.ndcg_loss(masked_pred, masked_labels)  
        rmse_loss = self.rmse_loss(masked_pred, masked_labels)

        ndcg_loss = ndcg_loss.clone().detach().requires_grad_(True)
        rmse_loss = rmse_loss.clone().detach().requires_grad_(True)

        total_loss = (spearman_loss + pearson_loss + ndcg_loss + rmse_loss) / 4.0
        return total_loss

    def spearman_loss(self, preds, labels):
        return torch.tensor(spearmanr(preds.cpu().numpy(), labels.cpu().numpy())[0], dtype=torch.float32, device=preds.device, requires_grad=True)

    def pearson_loss(self, preds, labels):
        return torch.tensor(pearsonr(preds.cpu().numpy(), labels.cpu().numpy())[0], dtype=torch.float32, device=preds.device, requires_grad=True)

    def ndcg_loss(self, preds, labels):
        return torch.tensor(ndcg_score([labels.cpu().numpy()], [preds.cpu().numpy()]), dtype=torch.float32, device=preds.device, requires_grad=True)

    def rmse_loss(self, preds, labels):
        print(labels)
        print(preds)
        return torch.tensor(mean_squared_error(labels.cpu().numpy(), preds.cpu().numpy(), squared=False), dtype=torch.float32, device=preds.device, requires_grad=True)


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
##############Defining SFTTrainer(Custom)#################
##########################################################
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = PersuasiveLoss() 

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 모델 예측 (labels와 inputs가 일치하는 경우)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        predictions = outputs.logits  # 모델 예측값
        
        # 손실 계산
        loss = self.loss_fn(predictions, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomSFTTrainer(
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

trainer.save_model("./llama3-8b-sft")
tokenizer.save_pretrained("./llama3-8b-sft")
