"""Fine-tuning script — exercises training task_type detection."""

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir="./out",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
    )

    trainer = Trainer(model=model, args=args)
    trainer.train()
    trainer.save_model("./out/final")


if __name__ == "__main__":
    main()
