import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os

os.chdir('/home/s2/naeunlee/PVP')

##########################################################
############## Loading Model and Tokenizer ###############
##########################################################
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

##########################################################
########################### QLoRA ########################
##########################################################
peft_config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"], 
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

##########################################################
###################### Loading Dataset ###################
##########################################################
dataset = load_dataset("json", data_files="dataset/evaluator.jsonl", split="train")

def preprocess_function(example):
    text = f"Instruction: {example['instruction']}\nInput: {str(example['input'])}\nImage Description: {example['image description']}"
    tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=256)
    tokenized["labels"] = int(example["output"].strip("[]"))  # Extract score
    return tokenized

dataset = dataset.map(preprocess_function)
dataset = dataset.train_test_split(test_size=0.1)

##########################################################
################ Defining Persuasive Loss ###############
##########################################################
class PersuasiveLoss(torch.nn.Module):
    def __init__(self):
        super(PersuasiveLoss, self).__init__()

    def forward(self, predictions, labels):
        probabilities = F.softmax(predictions, dim=-1)

        spearman_loss = self.spearman_loss(probabilities.float(), labels.float())  
        pearson_loss = self.pearson_loss(probabilities.float(), labels.float())    
        ndcg_loss = self.ndcg_loss(probabilities.float(), labels.float())  
        rmse_loss = self.rmse_loss(probabilities.float(), labels.float())

        total_loss = (spearman_loss + pearson_loss + ndcg_loss + rmse_loss) / 4.0
        return total_loss


    def spearman_loss(self, preds, labels):
        preds, labels = preds.flatten(), labels.flatten()
        preds_rank = preds.argsort().argsort().float()
        labels_rank = labels.argsort().argsort().float()
        return 1 - torch.nn.functional.cosine_similarity(preds_rank, labels_rank, dim=0)


    def pearson_loss(self, preds, labels):
        preds, labels = preds.flatten(), labels.flatten()
        preds_mean, labels_mean = preds.mean(), labels.mean()
        preds_std, labels_std = preds.std(), labels.std()
        return 1 - torch.nn.functional.cosine_similarity((preds - preds_mean) / preds_std, 
                                                        (labels - labels_mean) / labels_std, dim=0)


    def ndcg_loss(self, preds, labels):
        """Differentiable NDCG loss using softmax-based ranking."""
        preds = preds.flatten()
        labels = labels.flatten()

        # Compute relevance scores (normalized)
        relevance = F.softmax(labels, dim=0)  # Approximate ideal ranking

        # Compute discounted cumulative gain
        discount = 1.0 / torch.log2(torch.arange(2, len(preds) + 2, device=preds.device).float())
        dcg = torch.sum(relevance * discount)

        # Ideal DCG (sorted labels)
        ideal_relevance = F.softmax(torch.sort(labels, descending=True).values, dim=0)
        ideal_dcg = torch.sum(ideal_relevance * discount)

        return 1 - (dcg / ideal_dcg)

    def rmse_loss(self, preds, labels):
        return torch.sqrt(torch.nn.functional.mse_loss(preds, labels))


##########################################################
################## Setting Training Args #################
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
############## Defining SFTTrainer (Custom) ##############
##########################################################
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = PersuasiveLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").float()
        outputs = model(**inputs)
        predictions = outputs.logits.mean(dim=-1)  # Aggregate logits for score prediction
        
        loss = self.loss_fn(predictions, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomSFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=256
)

##########################################################
############### Training and Saving Model ################
##########################################################
trainer.train()

trainer.save_model("./evaluator")
tokenizer.save_pretrained("./evaluator_tokenizer")
