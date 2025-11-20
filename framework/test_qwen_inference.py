#!/usr/bin/env python3
"""
Simple inference test script for Qwen3-8B GraphLLM checkpoint.
Demonstrates iterative planning and answering capability.
"""

import argparse
import torch
from models.graphllm_pla_8b_qwen import GraphLLM
from utils import get_planner_instruction, get_answerer_instruction
from safetensors.torch import load_model
import sys


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint using safetensors"""
    try:
        load_model(model, checkpoint_path)
        print(f"✅ Successfully loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def test_planner(model, question, iteration=0, previous_subqueries=None, graph_info=None):
    """Test planner component"""
    planner_instruction = get_planner_instruction('qwen3-8b')
    
    # Build planner input
    if iteration == 0:
        # Initial planning: just question
        planner_input = planner_instruction + "\n" + question
        graphs = [[]]  # Empty graphs for initial planning
    else:
        # Iterative planning: previous subqueries + graph info + question
        planner_input = ""
        if previous_subqueries:
            for i, subq in enumerate(previous_subqueries):
                planner_input += subq + "\n"
                if graph_info and i < len(graph_info):
                    planner_input += "Retrieved Graph Information: " + str(graph_info[i]) + '\n'
        planner_input += "Question: " + question
        planner_input = planner_instruction + "\n" + planner_input
        graphs = [[]]  # Can add actual graphs here if available
    
    print(f"\n{'='*60}")
    print(f"PLANNER - Iteration {iteration + 1}")
    print(f"{'='*60}")
    print(f"Input:\n{planner_input}\n")
    
    # Run inference
    with torch.no_grad():
        result = model.inference({
            'input': [planner_input],
            'graphs': [graphs],
            'label': ['']
        })
    
    planner_output = result['pred'][0]
    print(f"Output: {planner_output}\n")
    
    return planner_output


def test_answerer(model, question, subqueries=None, graph_info=None):
    """Test answerer component"""
    answerer_instruction = get_answerer_instruction('qwen3-8b')
    
    # Build answerer input
    answerer_input = ""
    if subqueries:
        for i, subq in enumerate(subqueries):
            answerer_input += subq + "\n"
            if graph_info and i < len(graph_info):
                answerer_input += "Retrieved Graph Information: " + str(graph_info[i]) + '\n'
    answerer_input += "Question: " + question
    answerer_input = answerer_instruction + "\n" + answerer_input
    
    graphs = [[]]  # Can add actual graphs here if available
    
    print(f"\n{'='*60}")
    print(f"ANSWERER")
    print(f"{'='*60}")
    print(f"Input:\n{answerer_input}\n")
    
    # Run inference
    with torch.no_grad():
        result = model.inference({
            'input': [answerer_input],
            'graphs': [graphs],
            'label': ['']
        })
    
    answerer_output = result['pred'][0]
    print(f"Output: {answerer_output}\n")
    
    return answerer_output


def iterative_planning_and_answering(model, question, max_iterations=3):
    """Demonstrate iterative planning and answering"""
    print(f"\n{'#'*60}")
    print(f"ITERATIVE PLANNING AND ANSWERING DEMO")
    print(f"{'#'*60}")
    print(f"\nQuestion: {question}\n")
    
    subqueries = []
    graph_info = []  # Simulated graph information
    end_iteration = False
    
    # Iteration 0: Initial planning
    planner_output = test_planner(model, question, iteration=0)
    
    # Check if retrieval is needed
    if '[NO_RETRIEVAL]' in planner_output.upper() or 'SUFFICIENT' in planner_output.upper():
        print("✅ Model determined: NO_RETRIEVAL or SUFFICIENT - can answer directly")
        end_iteration = True
    else:
        # Extract subquery if present
        if '[SUBQ]' in planner_output.upper():
            # Try to extract subquery (simple extraction)
            subq = planner_output.replace('[SUBQ]', '').replace('[subq]', '').strip()
            if subq:
                subqueries.append(subq)
                # Simulate retrieved graph information
                graph_info.append(f"Graph info for: {subq}")
                print(f"📝 Extracted subquery: {subq}")
        else:
            # Use question as first subquery
            subqueries.append(question)
            graph_info.append(f"Graph info for: {question}")
    
    # Iterative planning (if needed)
    iteration = 1
    while not end_iteration and iteration < max_iterations:
        planner_output = test_planner(
            model, 
            question, 
            iteration=iteration,
            previous_subqueries=subqueries,
            graph_info=graph_info
        )
        
        if 'SUFFICIENT' in planner_output.upper() or '[NO_RETRIEVAL]' in planner_output.upper():
            print("✅ Model determined: SUFFICIENT - ready to answer")
            end_iteration = True
        elif '[SUBQ]' in planner_output.upper():
            # Extract new subquery
            subq = planner_output.replace('[SUBQ]', '').replace('[subq]', '').strip()
            if subq and subq not in subqueries:
                subqueries.append(subq)
                graph_info.append(f"Graph info for: {subq}")
                print(f"📝 Extracted new subquery: {subq}")
            iteration += 1
        else:
            # No clear signal, assume sufficient
            end_iteration = True
    
    # Final answering
    if subqueries or graph_info:
        final_answer = test_answerer(model, question, subqueries, graph_info)
    else:
        # Direct answer without retrieval
        final_answer = test_answerer(model, question)
    
    print(f"\n{'#'*60}")
    print(f"FINAL ANSWER")
    print(f"{'#'*60}")
    print(f"{final_answer}\n")
    
    return {
        'question': question,
        'subqueries': subqueries,
        'graph_info': graph_info,
        'final_answer': final_answer
    }


def main():
    parser = argparse.ArgumentParser(description='Test Qwen3-8B GraphLLM inference')
    parser.add_argument('--checkpoint', type=str, 
                       default='/shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen/checkpoint_10_of_20_qwen.safetensors',
                       help='Path to checkpoint file')
    parser.add_argument('--llm_model_path', type=str, default='Qwen/Qwen3-8B',
                       help='Base Qwen model path')
    parser.add_argument('--question', type=str, 
                       default='What is the capital of France?',
                       help='Test question')
    parser.add_argument('--max_iterations', type=int, default=3,
                       help='Maximum number of planning iterations')
    parser.add_argument('--max_txt_len', type=int, default=2500,
                       help='Maximum text length')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum new tokens to generate')
    parser.add_argument('--llm_frozen', type=str, default='False',
                       help='Whether LLM is frozen')
    parser.add_argument('--finetune_method', type=str, default='lora',
                       help='Finetune method (lora or full)')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank (must match training: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha (must match training: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    parser.add_argument('--gnn_model_name', type=str, default='gt',
                       help='GNN model name')
    parser.add_argument('--gnn_in_dim', type=int, default=1024,
                       help='GNN input dimension')
    parser.add_argument('--gnn_hidden_dim', type=int, default=1024,
                       help='GNN hidden dimension')
    parser.add_argument('--gnn_num_layers', type=int, default=3,
                       help='GNN number of layers')
    parser.add_argument('--gnn_dropout', type=float, default=0.1,
                       help='GNN dropout')
    parser.add_argument('--gnn_num_heads', type=int, default=8,
                       help='GNN number of heads')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Loading Qwen3-8B GraphLLM Model")
    print("="*60)
    
    # Initialize model
    model = GraphLLM(args)
    model.eval()
    
    # Load checkpoint
    if not load_checkpoint(model, args.checkpoint):
        print("❌ Failed to load checkpoint. Exiting.")
        sys.exit(1)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        print(f"Moving model to GPU...")
        model = model.cuda()
        print(f"✅ Model moved to GPU: {next(model.parameters()).device}")
    else:
        print(f"⚠️  No GPU available, using CPU (inference may be slow or have issues)")
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Checkpoint: {args.checkpoint}\n")
    
    # Run iterative planning and answering
    result = iterative_planning_and_answering(
        model, 
        args.question, 
        max_iterations=args.max_iterations
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Question: {result['question']}")
    print(f"Subqueries generated: {len(result['subqueries'])}")
    for i, sq in enumerate(result['subqueries']):
        print(f"  {i+1}. {sq}")
    print(f"\nFinal Answer:\n{result['final_answer']}\n")


if __name__ == '__main__':
    main()

