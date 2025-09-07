import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import Dict, Any

class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def create_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Set padding token to EOS token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer

class Evaluator:
    def evaluate(self, text: str) -> float:
        pass

class TextEvaluator(Evaluator):
    def __init__(self, model_name: str):
        factory = ModelFactory(model_name)
        self.model, self.tokenizer = factory.create_model()

    def evaluate(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        if outputs.loss is None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = outputs.loss

        return loss.item()

    def fine_tune(self, dataset: Dict[str, Any], output_dir: str = "./fine_tuned_model"):
        train_dataset = Dataset.from_dict(dataset)

        def tokenize_function(examples):
            # Tokenize inputs and Labels
            model_inputs = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            # Create Labels by shifting the input_ids
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs

        tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=500,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets,
        )

        trainer.train()
        self.save_model(output_dir)

    def save_model(self, file_path: str):
        self.model.save_pretrained(file_path)
        self.tokenizer.save_pretrained(file_path)

# Sample dataset for testing
sample_dataset = {
    "text": ["Example sentence for evaluation.", "Another example sentence."]
}

if __name__ == "__main__":
    evaluator = TextEvaluator("gpt2")
    loss = evaluator.evaluate("This is a test sentence.")
    print(f"Evaluation Loss: {loss}")

    # Fine-tuning the model with the sample dataset
    evaluator.fine_tune(sample_dataset)

    # Save the model
    evaluator.save_model("model/fine_tuned_model")

