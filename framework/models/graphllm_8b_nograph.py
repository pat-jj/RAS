import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from models.gnns import load_gnn_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math


class GraphLLM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            # "max_memory": {0: '7GiB', 1: '7GiB', 2: '7GiB', 3: '7GiB', 4: '7GiB', 5: '7GiB', 6: '7GiB', 7: '7GiB'},
            "max_memory": {0: '30GiB', 1: '30GiB'},
            # "max_memory": {0: '0GiB', 1: '30GiB', 2: '10GiB', 3: '10GiB', 4: '10GiB'},
            # "max_memory": {0: '0GiB', 1: '10GiB', 2: '10GiB', 3: '20GiB', 4: '20GiB'},
            # "max_memory": {0: '30GiB', 1: '30GiB', 2: '30GiB', 3: '30GiB'},
            # "max_memory": {1: '40GiB', 2: '40GiB', 3: '40GiB', 4: '40GiB', 5: '40GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        
        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get special tokens from tokenizer
        self.eos_token = self.tokenizer.eos_token
        
        # Construct chat format tokens
        self.BOS = '<|begin_of_text|>'
        self.EOS = '<|end_of_text|>'
        self.IGNORE_INDEX = -100

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for param in model.parameters():
                param.requires_grad = False
        else:
            if args.finetune_method == 'full':
                print("Full-parameter finetuning of LLAMA!")
                model.gradient_checkpointing_enable()
                for param in model.parameters():
                    param.requires_grad = True
            elif args.finetune_method == 'lora':
                print("Training LLAMA with LORA!")
                model = prepare_model_for_kbit_training(model)
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')


        self.word_embedding = self.model.model.get_input_embeddings()
        

    
    def forward(self, samples):
        # Tokenize inputs and labels
        inputs = self.tokenizer(samples['input'], add_special_tokens=False)
        labels = self.tokenizer(samples['label'], add_special_tokens=False)

        # Get special token ids
        eos_tokens = self.tokenizer(self.EOS, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
        ).to(self.model.device)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).to(self.model.device)

        batch_size = len(samples['input'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            # Prepare label sequence
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids

            # Include label in input sequence
            input_ids = (inputs.input_ids[i][:self.max_txt_len] + 
                        label_input_ids)

            # Create embeddings
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([
                bos_embeds,
                inputs_embeds
            ], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [self.IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # Get maximum length
        max_length = max([x.shape[0] for x in batch_inputs_embeds])

        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [self.IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        # Stack all tensors
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        # Forward pass with autocast
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):
        # encode inputs
        inputs = self.tokenizer(samples['input'], add_special_tokens=False)
        
        # encode special tokens
        bos_embeds = self.word_embedding(
            self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
        ).to(self.model.device)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0).to(self.model.device)

        batch_size = len(samples['input'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        for i in range(batch_size):
            # Add special tokens and create input embeddings
            input_ids = inputs.input_ids[i][:self.max_txt_len]
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # Concatenate embeddings
            inputs_embeds = torch.cat([
                bos_embeds,
                inputs_embeds
            ], dim=0)
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # Pad inputs to max length
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            if pad_length > 0:
                batch_inputs_embeds[i] = torch.cat([
                    pad_embeds.repeat(pad_length, 1), 
                    batch_inputs_embeds[i]
                ])
                batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        # Stack tensors
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        # Generate with autocast
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True, 
            )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        return {
            'input': samples['input'],
            'pred': predictions,
            'label': samples['label'],
        }
        
    def maybe_autocast(self, dtype=torch.bfloat16):
        """Helper for handling autocast"""
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def print_trainable_params(self):
        """Print trainable parameter stats"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f'trainable params: {trainable_params:,d} || '
            f'all params: {all_param:,d} || '
            f'trainable%: {100 * trainable_params / all_param:.2f}%'
        )

    @property
    def device(self):
        return list(self.parameters())[0].device