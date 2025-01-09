import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple
import re
from claude_api import get_claude_response  # Using your existing API module

class TripleFormatter:
    def __init__(self, input_file: str, output_dir: str, llm: str = "sonnet"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.llm = llm
        
        # Output files
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, "standardized_triples.json")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'formatter.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'formatted_locally': 0,
            'sent_to_llm': 0,
            'errors': 0
        }

    def parse_triples(self, triples_str: str) -> List[str]:
        """Parse triples string into individual triples."""
        # Extract all triples enclosed in parentheses
        triples = re.findall(r'\((.*?)\)', triples_str)
        return [t.strip() for t in triples if t.strip()]

    def is_valid_triple(self, triple: str) -> bool:
        """Check if a triple string follows the exact required format."""
        # Extract prefixes
        prefixes = re.findall(r'[A-Z]>', triple)
        if len(prefixes) != 3:
            return False
            
        # Check if we have exactly one of each required prefix
        return set(prefixes) == {'S>', 'P>', 'O>'}

    def create_prompt(self, text: str, triples: str) -> str:
        return f"""Task: Reformat the relationship triples to ensure each triple has exactly one subject (S>), one predicate (P>), and one object (O>). Convert any other types of relationships (like temporal) into this format.

Text: {text}
Current triples: {triples}

Output the reformatted triples in exactly this format: (S> subject| P> predicate| O> object), (S> subject2| P> predicate2| O> object2)
Each triple must:
- Be enclosed in parentheses
- Contain exactly one S>, one P>, and one O>
- Use | as separator
- Multiple triples should be separated by commas
Do not include any explanations or additional text."""

    def format_triple(self, triple: str) -> List[str]:
        """Try to format a single triple locally. Return None if it needs LLM processing."""
        # Check if it already has the correct format
        if self.is_valid_triple(triple):
            return [triple]
            
        # Try to fix simple cases of multiple objects
        if triple.count('O>') > 1:
            parts = triple.split('|')
            subject = next((p for p in parts if 'S>' in p), None)
            predicate = next((p for p in parts if 'P>' in p), None)
            objects = [p for p in parts if 'O>' in p]
            
            if subject and predicate and objects:
                return [f"{subject.strip()}| {predicate.strip()}| {obj.strip()}" 
                       for obj in objects]
                
        return None

    def process_triples(self, sample: Dict) -> Dict:
        """Process a single sample's triples, either locally or via LLM if needed."""
        try:
            # Parse individual triples
            current_triples = self.parse_triples(sample['generated_triple'])
            
            # Try to fix locally first
            new_triples = []
            needs_llm = False
            
            for triple in current_triples:
                formatted = self.format_triple(triple)
                if formatted:
                    new_triples.extend(formatted)
                else:
                    needs_llm = True
                    break
            
            if not needs_llm and all(self.is_valid_triple(t) for t in new_triples):
                self.stats['formatted_locally'] += 1
                formatted_str = ', '.join(f"({t})" for t in new_triples)
                return {
                    'text': sample['text'],
                    'generated_triple': formatted_str
                }
            
            # If local formatting failed, use LLM
            self.stats['sent_to_llm'] += 1
            prompt = self.create_prompt(
                text=sample['text'],
                triples=sample['generated_triple']
            )
            response = get_claude_response(llm=self.llm, prompt=prompt)
            
            # Validate LLM response
            response = response.strip()
            if not response.startswith('(') or not response.endswith(')'):
                self.logger.warning(f"Invalid LLM response format: {response}")
                return sample
                
            return {
                'text': sample['text'],
                'generated_triple': response
            }
            
        except Exception as e:
            self.logger.error(f"Error processing sample: {str(e)}")
            self.stats['errors'] += 1
            return sample

    def process_dataset(self):
        """Process the entire dataset and save results."""
        try:
            # Load input data
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.stats['total_samples'] = len(data)
            self.logger.info(f"Processing {self.stats['total_samples']} samples...")
            
            # Process each sample
            processed_data = []
            for i, sample in enumerate(data, 1):
                if i % 100 == 0:
                    self.logger.info(f"Processed {i}/{self.stats['total_samples']} samples...")
                processed_sample = self.process_triples(sample)
                processed_data.append(processed_sample)
            
            # Save results
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            # Log statistics
            self.logger.info(f"""
Processing completed:
- Total samples: {self.stats['total_samples']}
- Formatted locally: {self.stats['formatted_locally']}
- Sent to LLM: {self.stats['sent_to_llm']}
- Errors: {self.stats['errors']}
Results saved to: {self.output_file}
            """)
            
        except Exception as e:
            self.logger.error(f"Error in process_dataset: {str(e)}")
            raise

def main():
    formatter = TripleFormatter(
        input_file="/shared/eng/pj20/firas_data/datasets/hotpotqa/verified_triples/verified_triples_clean.json",
        output_dir="/shared/eng/pj20/firas_data/datasets/hotpotqa/verified_triples/standardized",
        llm="sonnet"
    )
    formatter.process_dataset()

if __name__ == "__main__":
    main()