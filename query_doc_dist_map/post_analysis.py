import torch
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from kneed import KneeLocator

def load_trained_model(model_path, input_dim):
    model = DistributionMapper(input_dim)
    model.load_state_dict(torch.load(model_path))
    return model

def analyze_distances(model, val_loader, device):
    model.eval()
    js_distances = []
    wasserstein_distances = []
    
    with torch.no_grad():
        for query_dist, doc_dist in val_loader:
            query_dist, doc_dist = query_dist.to(device), doc_dist.to(device)
            pred_dist = model(query_dist)
            
            batch_metrics = compute_distribution_metrics(pred_dist, doc_dist)
            js_distances.extend(batch_metrics['js_distances'])
            wasserstein_distances.extend(batch_metrics['wasserstein_distances'])
    
    distances = {
        'js': np.array(js_distances),
        'wasserstein': np.array(wasserstein_distances)
    }
    
    # Print statistics for each distance metric
    for metric_name, metric_distances in distances.items():
        print(f"\n{metric_name.upper()} Distance Statistics:")
        print(f"Mean: {np.mean(metric_distances):.4f}")
        print(f"Std: {np.std(metric_distances):.4f}")
        print(f"Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"{p}th: {np.percentile(metric_distances, p):.4f}")
    
    return distances

def select_thresholds(distances):
    thresholds = {}
    
    for metric_name, metric_distances in distances.items():
        thresholds[metric_name] = {
            'percentile_90': np.percentile(metric_distances, 90),
            'mean_2std': np.mean(metric_distances) + 2 * np.std(metric_distances),
        }
        
        # Elbow method
        sorted_distances = np.sort(metric_distances)
        kneedle = KneeLocator(range(len(sorted_distances)), 
                            sorted_distances, 
                            S=1.0, 
                            curve='convex', 
                            direction='increasing')
        if kneedle.elbow is not None:
            thresholds[metric_name]['elbow'] = sorted_distances[kneedle.elbow]
        else:
            thresholds[metric_name]['elbow'] = np.mean([thresholds[metric_name]['percentile_90'], 
                                                       thresholds[metric_name]['mean_2std']])
    
    return thresholds

def plot_distance_distributions(distances, thresholds):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, (metric_name, metric_distances) in enumerate(distances.items()):
        axes[idx].hist(metric_distances, bins=50, density=True, alpha=0.7)
        axes[idx].set_title(f'{metric_name.capitalize()} Distance Distribution')
        axes[idx].set_xlabel('Distance')
        axes[idx].set_ylabel('Density')
        
        # Plot threshold lines
        for threshold_name, threshold_value in thresholds[metric_name].items():
            axes[idx].axvline(threshold_value, 
                            linestyle='--', 
                            label=f'{threshold_name}: {threshold_value:.4f}')
        
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('distance_distributions.png')
    plt.close()

def main():
    # Load your trained model and validation data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    query_path = '/shared/eng/pj20/firas_data/datasets/classifier_labeling_data/query_class_probabilities.csv'
    doc_path = '/shared/eng/pj20/firas_data/datasets/classifier_labeling_data/document_class_probabilities.csv'
    
    X, y = load_data(query_path, doc_path)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    val_dataset = DistributionDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Load model
    model = load_trained_model('best_distribution_mapper.pt', X.shape[1])
    model = model.to(device)
    
    # Analyze distances
    distances = analyze_distances(model, val_loader, device)
    
    # Select thresholds
    thresholds = select_thresholds(distances)
    
    # Print selected thresholds
    print("\nSelected Thresholds:")
    for metric_name, metric_thresholds in thresholds.items():
        print(f"\n{metric_name.upper()}:")
        for threshold_name, threshold_value in metric_thresholds.items():
            print(f"{threshold_name}: {threshold_value:.4f}")
    
    # Plot distributions and thresholds
    plot_distance_distributions(distances, thresholds)
    
    # Estimate corpus size for each threshold
    print("\nEstimated corpus size (percentage of total corpus):")
    for metric_name, metric_distances in distances.items():
        print(f"\n{metric_name.upper()}:")
        total_docs = len(metric_distances)
        for threshold_name, threshold_value in thresholds[metric_name].items():
            selected_docs = np.sum(metric_distances < threshold_value)
            percentage = (selected_docs / total_docs) * 100
            print(f"{threshold_name}: {percentage:.2f}% of corpus")

if __name__ == "__main__":
    main()