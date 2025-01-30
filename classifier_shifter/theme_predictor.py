import torch
from sentence_transformers import SentenceTransformer
from doc_classifier_train import MultilabelClassifier, load_trained_model
from dist_shifter import DistributionMapper
import json
from pathlib import Path
import numpy as np

class ThemePredictor:
    def __init__(self, classifier_path, shifter_path):
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classifier and encoder
        self.classifier, self.encoder, checkpoint_info = load_trained_model(
            classifier_path,
            self.device
        )
        self.classifier.to(self.device)  # Ensure classifier is on correct device
        self.encoder.to(self.device)     # Ensure encoder is on correct device
        self.classifier.eval()
        self.encoder.eval()
        
        # Load label mapping
        self.label_mapping = checkpoint_info['label_mapping']
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Load distribution shifter
        self.shifter = DistributionMapper(input_dim=len(self.label_mapping))
        self.shifter.load_state_dict(torch.load(shifter_path, map_location=self.device))
        self.shifter.to(self.device)     # Ensure shifter is on correct device
        self.shifter.eval()
        
    def predict(self, text: str, top_k: int = 10, mode: str = 'q'):
        # Get text embedding
        with torch.no_grad():
            # Encode and ensure it's on the correct device
            embedding = self.encoder.encode(
                text,
                convert_to_tensor=True,
                device=self.device  # Specify device during encoding
            )
            
            # Get query distribution
            if mode == 'q':
                query_dist = self.classifier(embedding.unsqueeze(0))
                # Shift distribution
                doc_dist = self.shifter(query_dist)
            elif mode == 'd':
                query_dist = self.classifier(embedding.unsqueeze(0))
                doc_dist = query_dist
        
        # Convert to numpy
        doc_dist = doc_dist.squeeze().cpu().numpy()
        
        # Get top k predictions
        top_indices = np.argsort(doc_dist)[-top_k:][::-1]
        top_themes = [(self.idx_to_label[idx], float(doc_dist[idx])) for idx in top_indices]
        
        return {
            'full_distribution': doc_dist,
            'top_themes': top_themes
        }

def main():
    # Initialize predictor
    predictor = ThemePredictor(
        classifier_path='/shared/eng/pj20/firas_data/classifiers/best_model',
        shifter_path='/shared/eng/pj20/firas_data/classifiers/best_distribution_mapper.pt'
    )
    
    # Get input from user
    while True:
        text = input("\nSelect mode: q: query, d: doc (or 'b' to quit): ")
        if text.lower() == 'b':
            break
            
        # Get predictions
        if text.lower() == 'q':
            mode = 'q'
            text = input("\nEnter a sentence: ")
        elif text.lower() == 'd':
            mode = 'd'
            text = input("\nEnter a document: ")
        
        if text.lower() == 'b':
            break
        
        results = predictor.predict(text, mode=mode)
        
        # Print results
        print("\nTop 10 predicted themes:")
        print("-" * 50)
        for theme, prob in results['top_themes']:
            print(f"{theme:<40} {prob:.4f}")

if __name__ == "__main__":
    main()