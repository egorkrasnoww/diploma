from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Загружаем CSV с парами "source" и "target"
dataset = load_dataset('csv', data_files='fine_tune_rzd_dataset.csv')

# Загружаем модель и токенизатор
tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base-multitask", use_fast=False)
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base-multitask")

# Преобразование данных
def preprocess(example):
    input_enc = tokenizer(example['source'], truncation=True, padding='max_length', max_length=512)
    target_enc = tokenizer(example['target'], truncation=True, padding='max_length', max_length=128)
    input_enc['labels'] = target_enc['input_ids']
    return input_enc

# Применяем токенизацию
tokenized_dataset = dataset.map(preprocess, batched=True)

# Настройка обучения
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_rut5_model",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    report_to="none",
    evaluation_strategy="no",
    fp16=False  # Если есть GPU с поддержкой — можно True
)

# Тренер
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)

# Запускаем дообучение
trainer.train()

# Сохраняем модель
model.save_pretrained("finetuned_rut5_model")
tokenizer.save_pretrained("finetuned_rut5_model")
