import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from models.gnns import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphLLM(torch.nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '40GiB', 1: '40GiB', 2: '40GiB', 3: '40GiB', 4: '40GiB'},
            # "max_memory": {0: '80GiB', 1: '80GiB', 2: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            if args.finetune_method == 'full':
                print("Full-parameter finetuning of LLAMA!")
                model.gradient_checkpointing_enable()
                for name, param in model.named_parameters():
                    param.requires_grad = True
            elif args.finetune_method == 'lora':
                print("Training LLAMA with LORA!")
                model = prepare_model_for_kbit_training(model)
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=[
                        "q_proj",
                        "v_proj",
                    ],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)
            else:
                raise ValueError(f"Unknown finetune_method: {args.finetune_method}")

        self.model = model
        print('Finish loading LLAMA!')

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

    def print_trainable_params(self):
        """Print number of trainable parameters."""
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

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        """
        Encode multiple graphs per query with proper dimension handling
        Args:
            samples: Dictionary containing 'graphs' key with list of graphs per query
        Returns:
            Tensor of graph embeddings with shape [batch_size, 1, hidden_dim]
        """
        batch_size = len(samples['id'])
        g_embeds_list = []
        
        # Process each query in the batch
        for i in range(batch_size):
            query_graphs = samples['graphs'][i]
            graph_embeds = []
            
            # Encode each graph for this query
            for graph in query_graphs:
                graph = graph.to(self.model.device)
                n_embeds, _ = self.graph_encoder(graph.x, graph.edge_index.long(), graph.edge_attr)
                # Mean pooling for each graph
                g_embed = scatter(n_embeds, 
                                torch.zeros(n_embeds.size(0), dtype=torch.long, device=self.model.device), 
                                dim=0, 
                                reduce='mean')
                graph_embeds.append(g_embed)
            
            # Combine embeddings of all graphs for this query
            if graph_embeds:
                # Stack and mean pool across graphs
                query_embed = torch.stack(graph_embeds).mean(dim=0)  # [1, hidden_dim]
            else:
                # Fallback if no graphs
                query_embed = torch.zeros((1, self.projector[0].in_features), 
                                    device=self.model.device)
            
            g_embeds_list.append(query_embed)
        
        # Stack all query embeddings and add sequence dimension
        g_embeds = torch.stack(g_embeds_list)  # [batch_size, hidden_dim]
        g_embeds = g_embeds.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        return g_embeds

    def forward(self, samples):
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]).unsqueeze(0)  # Add sequence dim
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs with proper dimensions
        graph_embeds = self.encode_graphs(samples)  # [batch_size, 1, hidden_dim]
        graph_embeds = self.projector(graph_embeds.squeeze(1)).unsqueeze(1)  # Apply projector and restore sequence dim

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))  # [seq_len, hidden_dim]
            inputs_embeds = inputs_embeds.unsqueeze(0)  # Add batch dim: [1, seq_len, hidden_dim]
            
            # Concatenate along sequence dimension (dim=1)
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i], inputs_embeds], dim=1)  # [1, total_seq_len, hidden_dim]

            batch_inputs_embeds.append(inputs_embeds.squeeze(0))  # Remove batch dim for padding
            batch_attention_mask.append([1] * inputs_embeds.size(1))
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.size(1)-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param