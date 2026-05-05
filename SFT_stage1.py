import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import shutil
import traceback
from jinja2 import Template

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, data=None):
        self.config = config
        self.tokenizer = tokenizer
        if data is not None:
            self.data = data
        else:
            with open(config.data_path) as f:
                self.data = json.load(f)
            newdata = []
            for da in self.data:
                newdata.append(da)
            print('过滤掉',len(self.data),len(newdata))
            self.data = newdata

        self.max_seq_len = self.config.max_seq_len
        self.debug = 0

        # 如果从Base LLMs训练，选择 llama3-instruct作为模版
        chat_template_llama3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        if not tokenizer.chat_template:
            tokenizer.chat_template = chat_template_llama3

        self.template = Template(tokenizer.chat_template)

    def __getitem__(self, index):
        return self.data[index]

    def get_response(self,da):
        temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        return temp.format(da['Complex_CoT'],da['Response'])


    def get_prompt(self,da):

        q = da['Question']
        a = self.get_response(da)
        assert q is not None and a is not None, f'q:{q} a:{a}'

        input =  self.template.render(messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],bos_token=self.tokenizer.bos_token,add_generation_prompt=False)
        input_ids = self.tokenizer.encode(input,add_special_tokens= False)

        query = self.template.render(messages=[{"role": "user", "content": q}],bos_token=self.tokenizer.bos_token,add_generation_prompt=True)
        query_ids = self.tokenizer.encode(query,add_special_tokens= False)

        labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:]}

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_seq_len)
        input_ids = [ item[:max_len] + [self.tokenizer.eos_token_id]*(max_len-len(item)) for item in input_ids]
        labels = [ item[:max_len] + [-100]*(max_len-len(item)) for item in labels]
        if self.debug < 3:
            print('input_ids',self.tokenizer.decode(input_ids[-1]))
            print('labels',self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug += 1

        return {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
            }

    def __len__(self):
        return len(self.data)

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / self.n_step

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


def train(args):

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir, mode="offline")
        tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))
    else:
        tb_writer = None

    accelerator.print(f'args:\n{args}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.resume_from_checkpoint:
        from peft import PeftModel
        accelerator.print(f'Loading LoRA adapter from {args.resume_from_checkpoint}')
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = bnb.optim.PagedAdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate)

    with open(args.data_path) as f:
        all_data = json.load(f)
    import random
    random.seed(args.seed)
    random.shuffle(all_data)
    val_size = max(1, int(len(all_data) * args.val_ratio))
    val_data, train_data = all_data[:val_size], all_data[val_size:]
    accelerator.print(f'train size: {len(train_data)}, val size: {val_size}')

    train_dataset = Train_dataset(args, tokenizer, data=train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)
    val_dataset = Train_dataset(args, tokenizer, data=val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False, drop_last=False, collate_fn=val_dataset.collate_fn)

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)
    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_path:{args.data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    start_epoch = 0
    start_step = 0
    global_step = 0
    if args.resume_from_checkpoint:
        state_file = os.path.join(os.path.dirname(args.resume_from_checkpoint), 'training_state.pt')
        if os.path.exists(state_file):
            state = torch.load(state_file, map_location='cpu')
            start_epoch = state['epoch']
            start_step = state['step'] + 1
            global_step = state['global_step']
            accelerator.print(f'Resumed from epoch={start_epoch} step={start_step} global_step={global_step}')

    metric = SFTMetric(device=torch.cuda.current_device())

    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            if args.max_ckpts > 0:
                if num_checkpoints >= args.max_ckpts:
                    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, 'tfmr')
            os.makedirs(output_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(output_dir, save_embedding_layers=False)
            tokenizer.save_pretrained(output_dir)
            print(f'LoRA adapter saved in {output_dir}')

        accelerator.wait_for_everyone()
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    model.train()

    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids=batch['input_ids']
            labels=batch['labels']

            output = model(input_ids=input_ids, labels=labels, return_dict=True,use_cache=False)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                model.eval()
                val_metric = SFTMetric(device=torch.cuda.current_device())
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_out = model(input_ids=val_batch['input_ids'], labels=val_batch['labels'], return_dict=True, use_cache=False)
                        val_metric(val_out.logits, val_batch['labels'], val_out.loss)
                val_acc, val_loss = val_metric.get_metric()
                if accelerator.is_main_process:
                    accelerator.print(f'[eval] step={global_step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
                    tb_writer.add_scalar('val/loss', val_loss, global_step)
                    tb_writer.add_scalar('val/acc', val_acc, global_step)
                    wandb.log({'val/loss': val_loss, 'val/acc': val_acc}, step=global_step)
                model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
                accelerator.print(f'Reached max_steps={args.max_steps}, stopping.')
                return

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)
                tb_writer.add_scalar('train/loss', train_loss, global_step)
                tb_writer.add_scalar('train/acc', acc, global_step)
                tb_writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], global_step)

        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str,default='sft_stage1')

    # Model Args
    parser.add_argument('--model_path', required=True, type=str)

    # Data Args
    parser.add_argument('--data_path', required=True, type=str)

    # Training Args
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=2, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=1, type=int)
    parser.add_argument('--lora_rank', default=64, type=int)
    parser.add_argument('--max_steps', default=-1, type=int, help='Stop after this many steps (-1 = no limit)')
    parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='Path to LoRA adapter dir (tfmr/) to resume from')
    parser.add_argument('--eval_steps', default=50, type=int, help='Run validation every N steps (0 = disable)')
    parser.add_argument('--val_ratio', default=0.02, type=float, help='Fraction of data used for validation')

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)
