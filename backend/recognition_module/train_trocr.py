import os
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq
from transformers import default_data_collator
import torch
from torch.utils.data import Dataset
from accelerate import Accelerator, DataLoaderConfiguration
from torch.nn.utils.rnn import pad_sequence


class OCRDataset(Dataset):
    def __init__(self, csv_file, root_dir, processor):
        self.annotations = pd.read_csv(csv_file, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
        self.root_dir = root_dir
        self.processor = processor

        # Проверка наличия изображений в папке и удаление отсутствующих
        self.annotations['exists'] = self.annotations['filename'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.annotations = self.annotations[self.annotations['exists']].drop(columns=['exists'])
        
        print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        text = self.annotations.iloc[idx, 1]
        
        # Preprocess the image and text
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()

        return {"pixel_values": pixel_values, "labels": labels}

def custom_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}

def train_trocr(train_csv, train_root, val_csv, val_root, model_name, output_dir, epochs=3, batch_size=8, learning_rate=5e-5, gpu=False):
    # Initialize the processor and model
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set decoder_start_token_id and bos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.bos_token_id = processor.tokenizer.cls_token_id

    # Prepare the datasets
    train_dataset = OCRDataset(csv_file=train_csv, root_dir=train_root, processor=processor)
    val_dataset = OCRDataset(csv_file=val_csv, root_dir=val_root, processor=processor)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",  # Evaluate every `eval_steps`
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=gpu,
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,  # Log every 100 steps
        save_steps=100,  # Save model every 500 steps
        eval_steps=100,  # Evaluate every 500 steps
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_total_limit=1,  # Only keep the best model
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
        greater_is_better=False  # Lower eval_loss is better
    )

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collator,
        tokenizer=processor.tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the trained model and processor
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Create and save the GenerationConfig object
    generation_config = GenerationConfig(max_length=64)
    generation_config.save_pretrained(output_dir)
    