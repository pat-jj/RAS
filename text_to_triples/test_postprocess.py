# import unittest

# def postprocess_generation(text):
#     """Post-process generated text to extract triples in the correct format"""
#     # Handle None or non-string input
#     if text is None or not isinstance(text, str):
#         return ""
        
#     # Find the position after the instruction
#     instruction_prefix = "Convert this text to a list of triples:\n"
#     if instruction_prefix in text:
#         try:
#             # Split on instruction prefix and take the last part
#             parts = text.split(instruction_prefix)
#             if len(parts) > 1:
#                 generated_text = parts[-1].strip()
#                 # Look for the first occurrence of a triple pattern
#                 triple_start = generated_text.find("(S>")
#                 if triple_start != -1:
#                     # Extract from the first triple to the end
#                     triples = generated_text[triple_start:].strip()
#                     # Clean up any extra whitespace between triples
#                     triples = ', '.join(t.strip() for t in triples.split(','))
#                     return triples
#                 else:
#                     # Fallback: look for any parenthesis-enclosed content
#                     triple_start = generated_text.find("(")
#                     if triple_start != -1:
#                         triples = generated_text[triple_start:].strip()
#                         return triples
#                     else:
#                         return generated_text
#         except Exception as e:
#             print(f"Error processing prediction: {e}")
#             return text
#     return text

# class TestTriplePostprocessing(unittest.TestCase):
#     def setUp(self):
#         # Test cases with expected inputs and outputs
#         self.test_cases = [
#             # Basic case with correct format
#             {
#                 "input": "Convert this text to a list of triples:\nJohn works at Google.\n(S> John| P> Employer| O> Google)",
#                 "expected": "(S> John| P> Employer| O> Google)"
#             },
#             # Multiple triples with extra whitespace
#             {
#                 "input": "Convert this text to a list of triples:\nJohn is a software engineer at Google.\n(S> John| P> Occupation| O> Software Engineer),    (S> John| P> Employer| O> Google)",
#                 "expected": "(S> John| P> Occupation| O> Software Engineer), (S> John| P> Employer| O> Google)"
#             },
#             # Case with text before triples
#             {
#                 "input": "Convert this text to a list of triples:\nBased on the text, here are the triples:\n(S> John| P> Role| O> Developer)",
#                 "expected": "(S> John| P> Role| O> Developer)"
#             },
#             # Case with multiple newlines and spacing
#             {
#                 "input": "Convert this text to a list of triples:\nHere's the text:\n\nJohn works at Google.\n\nHere are the triples:\n(S> John| P> Employer| O> Google)",
#                 "expected": "(S> John| P> Employer| O> Google)"
#             },
#             # Case without instruction prefix
#             {
#                 "input": "(S> John| P> Employer| O> Google)",
#                 "expected": "(S> John| P> Employer| O> Google)"
#             },
#             # Case with malformed triples
#             {
#                 "input": "Convert this text to a list of triples:\nHere are the triples: (S>John|P>Works|O>Google",
#                 "expected": "(S>John|P>Works|O>Google"
#             }
#         ]

#     def test_postprocessing(self):
#         for i, test_case in enumerate(self.test_cases):
#             result = postprocess_generation(test_case["input"])
#             self.assertEqual(
#                 result, 
#                 test_case["expected"],
#                 f"Test case {i + 1} failed:\nInput: {test_case['input']}\nExpected: {test_case['expected']}\nGot: {result}"
#             )

#     def test_error_handling(self):
#         # Test with None input
#         self.assertIsNotNone(postprocess_generation(None))
        
#         # Test with empty string
#         self.assertEqual(postprocess_generation(""), "")
        
#         # Test with malformed input
#         malformed_input = "Convert this text to a list of triples:\n(S> incomplete triple"
#         result = postprocess_generation(malformed_input)
#         self.assertIsNotNone(result)

# def print_test_results(text, expected_output):
#     """Helper function to manually test and visualize the post-processing"""
#     print("\nTesting post-processing:")
#     print(f"Input text:\n{text}")
#     result = postprocess_generation(text)
#     print(f"\nProcessed output:\n{result}")
#     print(f"\nExpected output:\n{expected_output}")
#     print(f"\nMatch: {result == expected_output}")
#     print("-" * 80)

# if __name__ == "__main__":
#     # Run unit tests
#     unittest.main(argv=[''], exit=False)
    
#     print("\nRunning manual test cases...")
    
#     # Example from your data
#     example_text = '''Convert this text to a list of triples:
# Goertz also served as a social media producer for @midnight.
# (S> Goertz| P> Employer| O> @midnight), (S> Goertz| P> Occupation| O> Social media producer)'''
    
#     example_expected = "(S> Goertz| P> Employer| O> @midnight), (S> Goertz| P> Occupation| O> Social media producer)"
    
#     print_test_results(example_text, example_expected)
    
#     # Test with some variations
#     variant_text = '''Convert this text to a list of triples:
# Here's the text: Goertz works at @midnight.

# Generated triples:
# (S> Goertz| P> Employer| O> @midnight)'''
    
#     variant_expected = "(S> Goertz| P> Employer| O> @midnight)"
    
#     print_test_results(variant_text, variant_expected)


import unittest
import torch
from torch.utils.data import Dataset
import json
import os
from transformers import AutoTokenizer
import tempfile

class TripletDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                input_text = f"Convert this text to a list of triples:\n{item['text']}"
                self.data.append({
                    'input': input_text,
                    'label': item['triplet']
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input and label separately
        input_ids = self.tokenizer(
            f"{self.tokenizer.bos_token}{item['input']}",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids[0]
        
        label_ids = self.tokenizer(
            f"{item['label']}{self.tokenizer.eos_token}",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids[0]
        
        # Create combined input_ids with padding
        combined_length = min(len(input_ids) + len(label_ids), self.max_length)
        padded_input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id)
        padded_input_ids[:len(input_ids)] = input_ids[:len(padded_input_ids)]
        
        # Create labels: -100 for input portion, actual labels for output portion
        labels = torch.full((self.max_length,), -100)
        label_start = min(len(input_ids), self.max_length - len(label_ids))
        labels[label_start:label_start + len(label_ids)] = label_ids[:self.max_length - label_start]
        
        # Create attention mask
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'raw_input': item['input'],
            'raw_label': item['label']
        }

class TestTripletDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize tokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.2-3B")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        
        # Create a temporary file with test data
        cls.test_data = [
            {
                "text": "Goertz also served as a social media producer for @midnight.",
                "triplet": "(S> Goertz| P> Employer| O> @midnight), (S> Goertz| P> Occupation| O> Social media producer)"
            },
            {
                "text": "Short text.",
                "triplet": "(S> Text| P> Length| O> Short)"
            }
        ]
        
        # Create temporary file
        cls.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        for item in cls.test_data:
            cls.temp_file.write(json.dumps(item) + '\n')
        cls.temp_file.close()
        
        # Create dataset
        cls.dataset = TripletDataset(cls.temp_file.name, cls.tokenizer, max_length=2048)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary file
        os.unlink(cls.temp_file.name)

    def test_dataset_length(self):
        """Test if dataset length matches input data"""
        self.assertEqual(len(self.dataset), len(self.test_data))

    def test_item_structure(self):
        """Test if dataset items have correct structure"""
        item = self.dataset[0]
        expected_keys = {'input_ids', 'attention_mask', 'labels', 'raw_input', 'raw_label'}
        self.assertEqual(set(item.keys()), expected_keys)

    def test_tensor_shapes(self):
        """Test if tensors have correct shapes"""
        item = self.dataset[0]
        self.assertEqual(item['input_ids'].shape, torch.Size([2048]))
        self.assertEqual(item['attention_mask'].shape, torch.Size([2048]))
        self.assertEqual(item['labels'].shape, torch.Size([2048]))

    def test_special_tokens(self):
        """Test if special tokens are properly placed"""
        item = self.dataset[0]
        # Check if sequence starts with BOS token
        self.assertEqual(item['input_ids'][0].item(), self.tokenizer.bos_token_id)
        # Check if padding tokens are present in padded area
        self.assertTrue(torch.any(item['input_ids'] == self.tokenizer.pad_token_id))

    def test_attention_mask(self):
        """Test if attention mask correctly matches non-pad tokens"""
        item = self.dataset[0]
        # Check if attention mask is 1 for non-pad tokens and 0 for pad tokens
        self.assertTrue(torch.all(item['attention_mask'] == (item['input_ids'] != self.tokenizer.pad_token_id)))

    def test_labels(self):
        """Test if labels are properly formatted"""
        item = self.dataset[0]
        # Check if labels contain -100 values
        self.assertTrue(torch.any(item['labels'] == -100))
        # Check if there are some non-negative labels (actual token ids)
        self.assertTrue(torch.any(item['labels'] >= 0))

    def test_raw_text_preservation(self):
        """Test if raw text is preserved correctly"""
        item = self.dataset[0]
        expected_input = f"Convert this text to a list of triples:\n{self.test_data[0]['text']}"
        self.assertEqual(item['raw_input'], expected_input)
        self.assertEqual(item['raw_label'], self.test_data[0]['triplet'])

    def visualize_sample(self, idx=0):
        """Helper function to visualize a sample (not a test)"""
        item = self.dataset[idx]
        print("\nSample Visualization:")
        print(f"Raw input: {item['raw_input']}")
        print(f"Raw label: {item['raw_label']}")
        
        # Find where actual content starts and ends
        non_pad_mask = item['input_ids'] != self.tokenizer.pad_token_id
        content_start = torch.nonzero(non_pad_mask)[0].item()
        content_end = torch.nonzero(non_pad_mask)[-1].item()
        
        # Find where labels start (first non -100 value)
        label_start = torch.nonzero(item['labels'] != -100)[0].item()
        
        print("\nSequence Analysis:")
        print(f"Content starts at position: {content_start}")
        print(f"Labels start at position: {label_start}")
        print(f"Content ends at position: {content_end}")
        print(f"Total sequence length: {len(item['input_ids'])}")
        
        print("\nToken Breakdown:")
        print("Input portion:")
        input_tokens = self.tokenizer.convert_ids_to_tokens(item['input_ids'][content_start:label_start])
        print(f"First 10 tokens: {input_tokens[:10]}")
        
        print("\nLabel portion:")
        label_tokens = self.tokenizer.convert_ids_to_tokens(
            item['input_ids'][label_start:content_end+1]
        )
        print(f"First 10 tokens: {label_tokens[:10]}")
        
        # Show attention mask distribution
        attention_sum = item['attention_mask'].sum().item()
        print(f"\nAttention mask covers {attention_sum} tokens out of {len(item['attention_mask'])}")
        
        # Verify label alignment
        print("\nLabel Verification:")
        label_ids = item['labels'][item['labels'] != -100]
        decoded_labels = self.tokenizer.decode(label_ids)
        print(f"Decoded labels: {decoded_labels}")
        
        # Show full sequence
        print("\nFull sequence decoded:")
        print(self.tokenizer.decode(item['input_ids'][item['input_ids'] != self.tokenizer.pad_token_id]))

if __name__ == '__main__':
    # Run tests
    unittest.main(argv=[''], exit=False)
    
    # Visualize a sample
    test_dataset = TestTripletDataset()
    test_dataset.setUpClass()
    test_dataset.visualize_sample()
    test_dataset.tearDownClass()