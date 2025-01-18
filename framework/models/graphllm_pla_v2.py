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
            # "max_memory": {0: '20GiB', 1: '20GiB'},
            "max_memory": {0: '80GiB', 1: '80GiB', 2: '80GiB', 3: '80GiB'},
            # "max_memory": {0: '80GiB', 1: '80GiB', 2: '80GiB'},
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

        # Define special tokens
        special_tokens = {
            'additional_special_tokens': [
                '[NO_RETRIEVAL]',
                '[SUBQ]',
                '[SUFFICIENT]',
                'Question:',  # New
                'Retrieved Graph Information:'  # New
            ]
        }
        
        # Add special tokens to tokenizer
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        
        # Store special token ids
        self.no_retrieval_token_id = self.tokenizer.convert_tokens_to_ids('[NO_RETRIEVAL]')
        self.subq_token_id = self.tokenizer.convert_tokens_to_ids('[SUBQ]')
        self.sufficient_token_id = self.tokenizer.convert_tokens_to_ids('[SUFFICIENT]')

        # Get special tokens from tokenizer
        self.eos_token = self.tokenizer.eos_token  
        
        # Construct chat format tokens
        self.BOS = '<s>[INST]'
        self.EOS_USER = '[/INST]'
        self.EOS = '</s>'
        self.IGNORE_INDEX = -100

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )

        # Resize token embeddings after model is loaded
        model.resize_token_embeddings(len(self.tokenizer))

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

        # Graph encoder setup
        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()
        
        self.no_graph_embedding = nn.Parameter(
        torch.randn(1, 1, args.gnn_hidden_dim) / math.sqrt(args.gnn_hidden_dim)
        )
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=args.gnn_hidden_dim,
            num_heads=4,
            batch_first=True
        )

    def encode_graphs(self, graphs_list):
        """
        Encode graphs for the planner, handling empty graph lists
        """
        if not graphs_list:
            # Return zero tensor if no graphs
            return torch.zeros((1, 1, self.projector[0].in_features), device=self.model.device)
            
        graph_embeds = []
        for graph in graphs_list:
            try:
                graph = graph.to(self.model.device)
                n_embeds, _ = self.graph_encoder(graph.x, graph.edge_index.long(), graph.edge_attr)
                # Mean pooling for each graph
                g_embed = scatter(n_embeds, 
                                torch.zeros(n_embeds.size(0), dtype=torch.long, device=self.model.device),
                                dim=0,
                                reduce='mean')
                graph_embeds.append(g_embed)
            except (AttributeError, ValueError) as e:
                print(f"Warning: Error processing graph: {e}")
                continue
        
        if not graph_embeds:  # If all graphs failed processing
            return torch.zeros((1, 1, self.projector[0].in_features), device=self.model.device)
        
        # Stack and mean pool across graphs
        g_embeds = torch.stack(graph_embeds).mean(dim=0)  # [1, hidden_dim]
        return g_embeds.unsqueeze(0)  # [1, 1, hidden_dim]
    
    def forward(self, samples):
        # Verify special token handling
        for input_text in samples['input']:
            if '[NO_RETRIEVAL]' in input_text:
                assert self.tokenizer.convert_tokens_to_ids('[NO_RETRIEVAL]') in \
                    self.tokenizer(input_text, add_special_tokens=False).input_ids, \
                    "NO_RETRIEVAL token not correctly tokenized"

        # Tokenize inputs and labels
        inputs = self.tokenizer(samples['input'], add_special_tokens=False)
        labels = self.tokenizer(samples['label'], add_special_tokens=False)

        # Get special token ids
        eos_tokens = self.tokenizer(self.EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
        ).to(self.model.device)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).to(self.model.device)

        batch_size = len(samples['input'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        # First pass - get all embeddings and lengths
        for i in range(batch_size):
            # Encode graphs
            graph_embeds = self.encode_graphs(samples['graphs'][i])  # [1, 1, hidden_dim]
            assert graph_embeds.size(-1) == self.projector[0].in_features, \
                f"Graph embedding dimension mismatch: {graph_embeds.size(-1)} vs {self.projector[0].in_features}"
            graph_embeds = self.projector(graph_embeds.squeeze(1))  # [1, proj_dim]

            # Prepare label sequence first (following G-Retriever)
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids

            # Now include label in input sequence (G-Retriever style)
            input_ids = (inputs.input_ids[i][:self.max_txt_len] + 
                        eos_user_tokens.input_ids + 
                        label_input_ids)  # Include labels in input

            # Create embeddings
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([
                bos_embeds,
                graph_embeds,
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
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
        ).to(self.model.device)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0).to(self.model.device)

        batch_size = len(samples['input'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        for i in range(batch_size):
            # Encode graphs for this sample
            graph_embeds = self.encode_graphs(samples['graphs'][i])  # [1, 1, hidden_dim]
            graph_embeds = self.projector(graph_embeds.squeeze(1))  # [1, proj_dim]
            
            # Add special tokens and create input embeddings
            input_ids = inputs.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # Concatenate all embeddings
            inputs_embeds = torch.cat([
                bos_embeds,
                graph_embeds,
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
                use_cache=True,  # IMPORTANT: keep this as in G-Retriever
                # pad_token_id=self.tokenizer.pad_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
            )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
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