import os
import time
import logging

import numpy as np

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt

from rouge import Rouge

import torch
import sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW, get_linear_schedule_with_warmup


class NewsDataset(Dataset):
    def __init__(self, tokenizer, data, input_length, output_length):
        self.dataset = data
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text

    def convert_to_features(self, example_batch):
        #         input_ = self.clean_text(example_batch['text']) + " </s>"
        #         target_ = self.clean_text(example_batch['headline']) + " </s>"

        input_ = self.clean_text(example_batch['text'])
        target_ = self.clean_text(example_batch['summary'])

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = hparams.model_name_or_path
        self.tokenizer = hparams.tokenizer_name_or_path
        self.rouge_metric = Rouge()

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def _generative_step(self, batch):

        t0 = time.time()

        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])

        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        #         rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(self.lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target)
        self.rouge_score = self.rouge_metric.get_scores(preds, target)

        #         rouge_results = self.rouge_metric.compute()
        #         rouge_dict = self.parse_score(rouge_results)
        #         base_metrics.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])

        return base_metrics

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

        # rouge_results = self.rouge_metric.compute()
        # rouge_dict = self.parse_score(rouge_results)

        tensorboard_logs.update(rouge1=self.rouge_score[0]['rouge-1'], rougeL=self.rouge_score[0]['rouge-l'])

        ## Clear out the lists for next epoch
        self.target_gen = []
        self.prediction_gen = []
        return {"avg_val_loss": avg_loss,
                "rouge1": self.rouge_score[0]['rouge-1'],
                "rougeL": self.rouge_score[0]['rouge-l'],
                "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       using_native_amp=False):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data=self.hparams.train_data, args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        validation_dataset = get_dataset(tokenizer=self.tokenizer, data=self.hparams.val_data, args=self.hparams)

        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

    def test_dataloader(self):
        test_dataset = get_dataset(tokenizer=self.tokenizer, data=self.hparams.test_data, args=self.hparams)

        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, trainer, pl_module):
        self.logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    self.logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        self.logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        self.logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

if __name__ == '__main__':
    pass




def get_dataset(tokenizer, data, args):
    return NewsDataset(tokenizer=tokenizer, data=data, input_length=args.max_input_length,
                       output_length=args.max_output_length)
