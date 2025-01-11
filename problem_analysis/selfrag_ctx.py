import json
import logging
from tqdm import tqdm
from collections import OrderedDict

class DocumentExtractor:
    def __init__(self, 
                 input_file: str,
                 output_file: str = "selfrag_documents.json"):
        self.input_file = input_file
        self.output_file = output_file
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Use OrderedDict to maintain insertion order and ensure uniqueness
        self.documents = OrderedDict()

    def extract_paragraphs(self, output: str) -> list:
        """Extract paragraphs from SelfRAG output format"""
        paragraphs = []
        
        if '[No Retrieval]' in output:
            return paragraphs
            
        current_pos = 0
        while True:
            start_idx = output.find('<paragraph>', current_pos)
            if start_idx == -1:
                break
            end_idx = output.find('</paragraph>', start_idx)
            if end_idx == -1:
                break
            paragraph = output[start_idx + len('<paragraph>'):end_idx].strip()
            paragraphs.append(paragraph)
            current_pos = end_idx + 1

        return paragraphs

    def process_dataset(self):
        try:
            # Load JSONL data
            self.logger.info(f"Processing file: {self.input_file}")
            doc_count = 0
            
            with open(self.input_file, 'r') as f:
                for line in tqdm(f, desc="Extracting documents"):
                    sample = json.loads(line)
                    paragraphs = self.extract_paragraphs(sample['output'])
                    
                    # Add each paragraph to documents dict with a unique ID
                    for para in paragraphs:
                        if para not in self.documents:
                            doc_id = f"doc_{doc_count}"
                            self.documents[para] = {
                                'id': doc_id,
                                'text': para
                            }
                            doc_count += 1

            # Convert to list format for output
            documents_list = []
            for d in self.documents.values():
                documents_list.append(d['text'])
            
            # Save documents
            with open(self.output_file, 'w') as f:
                json.dump(documents_list, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Extracted {len(documents_list)} unique documents")
            self.logger.info(f"Results saved to {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Error in process_dataset: {str(e)}")
            raise

def main():
    extractor = DocumentExtractor(
        input_file="/shared/eng/pj20/firas_data/datasets/selfrag/train.jsonl",
        output_file="/shared/eng/pj20/firas_data/datasets/selfrag/documents.json"
    )
    extractor.process_dataset()

if __name__ == "__main__":
    main()