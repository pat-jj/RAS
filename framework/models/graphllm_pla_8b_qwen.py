import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from models.gnns import load_gnn_model
from peft import LoraConfig, get_peft_model
# prepare_model_for_kbit_training removed - only needed for 8-bit/4-bit quantization
import math


class GraphLLM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading QWEN')
        kwargs = {
            "max_memory": {0: '30GiB', 1: '30GiB', 2: '30GiB', 3: '30GiB', 4: '30GiB', 5: '30GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"], trust_remote_code=True)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        
        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Get special tokens from tokenizer
        self.eos_token = self.tokenizer.eos_token
        
        # Construct chat format tokens for Qwen
        # NOTE: Using simple format to match what the model was trained with
        # The model was trained with: <|im_start|> + content + <|im_end|>
        # (Not the full Qwen3 chat template format)
        self.BOS = '<|im_start|>'
        self.EOS = '<|im_end|>'
        self.IGNORE_INDEX = -100

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing QWEN!")
            for param in model.parameters():
                param.requires_grad = False
        else:
            if args.finetune_method == 'full':
                print("Full-parameter finetuning of QWEN!")
                model.gradient_checkpointing_enable()
                for param in model.parameters():
                    param.requires_grad = True
            elif args.finetune_method == 'lora':
                print("Training QWEN with LORA!")
                # Only prepare for kbit training if using quantization (8-bit/4-bit)
                # For full precision LoRA, we don't need this and it causes bitsandbytes import issues
                # model = prepare_model_for_kbit_training(model)  # Commented out - not needed for full precision
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
        print('Finish loading QWEN!')

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
        # Handle empty list or list of empty lists
        if not graphs_list or (isinstance(graphs_list, list) and len(graphs_list) > 0 and all(not g or (isinstance(g, list) and len(g) == 0) for g in graphs_list)):
            # Return zero tensor if no graphs
            return torch.zeros((1, 1, self.projector[0].in_features), device=self.model.device)
            
        graph_embeds = []
        for graph in graphs_list:
            # Skip empty graphs
            if graph is None or (isinstance(graph, list) and len(graph) == 0):
                continue
            try:
                # Check if graph is a PyG Data object
                if hasattr(graph, 'to'):
                    graph = graph.to(self.model.device)
                    n_embeds, _ = self.graph_encoder(graph.x, graph.edge_index.long(), graph.edge_attr)
                    # Mean pooling for each graph
                    g_embed = scatter(n_embeds, 
                                    torch.zeros(n_embeds.size(0), dtype=torch.long, device=self.model.device),
                                    dim=0,
                                    reduce='mean')
                    graph_embeds.append(g_embed)
                else:
                    # Skip if not a valid graph object
                    continue
            except (AttributeError, ValueError, TypeError) as e:
                # Silently skip invalid graphs instead of printing warning
                continue
        
        if not graph_embeds:  # If all graphs failed processing
            return torch.zeros((1, 1, self.projector[0].in_features), device=self.model.device)
        
        # Stack and mean pool across graphs
        g_embeds = torch.stack(graph_embeds).mean(dim=0)  # [1, hidden_dim]
        return g_embeds.unsqueeze(0)  # [1, 1, hidden_dim]
    
    def forward(self, samples):
        # Tokenize inputs and labels
        inputs = self.tokenizer(samples['input'], add_special_tokens=False)
        labels = self.tokenizer(samples['label'], add_special_tokens=False)

        # Get special token ids - match training format
        eos_tokens = self.tokenizer(self.EOS, add_special_tokens=False)
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
            # Training format: BOS + content + label + EOS
            input_ids = (inputs.input_ids[i][:self.max_txt_len] + 
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
        # encode inputs - match training format
        inputs = self.tokenizer(samples['input'], add_special_tokens=False)
        
        # encode special tokens - use same format as training
        eos_tokens = self.tokenizer(self.EOS, add_special_tokens=False)
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
            # Match training format: BOS + content + EOS (for marking end of input)
            input_ids = inputs.input_ids[i][:self.max_txt_len] + eos_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # Concatenate all embeddings: BOS + graph_embeds + content + EOS
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

        # Set model to eval mode for inference
        self.model.eval()
        
        # CRITICAL FIX: model.generate() with inputs_embeds doesn't work correctly for Qwen
        # when custom graph embeddings are inserted. The output is shorter than input.
        # Solution: Use manual generation loop that properly handles inputs_embeds
        
        # First, do a forward pass to get initial hidden states and past_key_values
        # With device_map="auto", the model handles device placement automatically
        # We need to ensure inputs_embeds are on the device where the model's embedding expects them
        # The embedding layer is typically on the same device as the first layer
        # But to be safe, let's check where the model expects inputs
        try:
            # Try to get device from model's embedding or first layer
            if hasattr(self.model.model, 'embed_tokens'):
                embedding_device = next(self.model.model.embed_tokens.parameters()).device
            else:
                embedding_device = next(self.model.model.layers[0].parameters()).device
        except:
            # Fallback: use the device of inputs_embeds (should already be correct)
            embedding_device = inputs_embeds.device
        
        inputs_embeds = inputs_embeds.to(embedding_device)
        attention_mask = attention_mask.to(embedding_device)
        first_layer_device = embedding_device  # Store for later use
        
        with self.maybe_autocast():
            model_outputs = self.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
        
        # Get actual input lengths (excluding padding)
        input_lengths = attention_mask.sum(dim=1).cpu().tolist()  # [batch_size]
        
        # Get the device where the model's main parameters are (for device_map="auto", this might vary)
        # We'll use the device of the last hidden state
        main_device = model_outputs.last_hidden_state.device
        
        # Manual generation for each sample in batch
        all_generated_ids = []
        for batch_idx in range(batch_size):
            # Get the last hidden state for this sequence (where generation starts)
            seq_len = input_lengths[batch_idx]
            last_hidden = model_outputs.last_hidden_state[batch_idx:batch_idx+1, seq_len-1:seq_len, :]  # [1, 1, hidden_dim]
            # Ensure it's on the correct device
            last_hidden = last_hidden.to(main_device)
            
            # Extract past_key_values for this sequence
            # CRITICAL: With device_map="auto", past_key_values are on different devices per layer
            # We must NOT move them - keep them on their original devices
            if model_outputs.past_key_values:
                # past_key_values structure: tuple of tuples, each layer has (key, value) tensors
                # Extract batch item but keep on original device for each layer
                past_kv = tuple(
                    tuple(kv[batch_idx:batch_idx+1] for kv in layer_past)  # Keep on original device
                    for layer_past in model_outputs.past_key_values
                )
            else:
                past_kv = None
            
            # Generation loop
            generated_ids = []
            current_hidden = last_hidden
            
            for step in range(self.max_new_tokens):
                # Get logits from current hidden state
                # lm_head might be on a different device, so we need to handle that
                with self.maybe_autocast():
                    # Ensure current_hidden is on the same device as lm_head
                    lm_head_device = next(self.model.lm_head.parameters()).device
                    current_hidden_for_logits = current_hidden.to(lm_head_device)
                    logits = self.model.lm_head(current_hidden_for_logits)  # [1, 1, vocab_size]
                    # Move logits back to main_device for sampling
                    logits = logits.to(main_device)
                
                # Apply sampling parameters
                logits = logits / 0.7  # temperature
                
                # Top-k filtering
                top_k = 20
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > 0.8
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs[0, -1], 1).item()
                generated_ids.append(next_token_id)
                
                # Stop if EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                # Get embedding for next token
                # word_embedding might be on a different device
                word_embedding_device = next(self.word_embedding.parameters()).device
                next_token_tensor = torch.tensor([[next_token_id]], device=word_embedding_device)
                next_token_embed = self.word_embedding(next_token_tensor)  # [1, 1, hidden_dim]
                
                # Forward pass with next token
                # With device_map="auto", the model handles device placement internally
                # We need to ensure inputs_embeds is on the same device as the first forward pass
                # Use the same device we used for the initial forward pass
                next_token_embed = next_token_embed.to(first_layer_device)
                
                with self.maybe_autocast():
                    step_outputs = self.model.model(
                        inputs_embeds=next_token_embed,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True
                    )
                
                # Get the new hidden state - ensure it's on the correct device
                # The output device might be different from input device
                current_hidden = step_outputs.last_hidden_state
                # Move to main_device for consistency
                current_hidden = current_hidden.to(main_device)
                past_kv = step_outputs.past_key_values
            
            all_generated_ids.append(generated_ids)
        
        # Decode generated tokens
        outputs = [torch.tensor(ids, device=self.model.device) for ids in all_generated_ids]

        # Decode generated tokens (these are already only the generated tokens, not input+generated)
        predictions = []
        for i, generated_ids in enumerate(outputs):
            if len(generated_ids) > 0:
                pred_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                
                # Remove EOS token if present
                if self.EOS in pred_text:
                    pred_text = pred_text.split(self.EOS)[0]
                
                pred_text = pred_text.strip()
            else:
                pred_text = ""
            
            # Debug for first sample
            if i == 0:
                print(f"DEBUG: Generated {len(generated_ids)} tokens")
                print(f"DEBUG: Generated IDs: {generated_ids.tolist()[:20] if len(generated_ids) > 0 else '[]'}")
                print(f"DEBUG: Decoded text: {repr(pred_text[:100])}")
            
            predictions.append(pred_text)
        
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

