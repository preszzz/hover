import os
import argparse
from data.processed.audio_to_features import process_dataset_to_csv
from data.features.dataset_analysis import load_and_analyze_features, visualize_feature_distributions, visualize_feature_importance

def main():
    parser = argparse.ArgumentParser(description='Process audio dataset to features CSV and analyze')
    
    parser.add_argument('--input_dir', type=str, default='../../data/raw',
                        help='Directory with raw audio files organized by class')
    
    parser.add_argument('--output_dir', type=str, default='../../data/processed',
                        help='Directory to save processed data')
    
    parser.add_argument('--segment_duration', type=float, default=1.0,
                        help='Duration of each audio segment in seconds')
    
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap between audio segments (0.0-1.0)')
    
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of files to process in each batch')
    
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (None for all)')
    
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip feature extraction and use existing CSV')
    
    parser.add_argument('--skip_analysis', action='store_true',
                        help='Skip dataset analysis')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Define output CSV path
    features_csv = os.path.join(args.output_dir, 'features_dataset.csv')
    
    # Extract features
    if not args.skip_extraction:
        print("Starting feature extraction...")
        
        # Feature configuration
        feature_config = {
            'mfcc': {'n_mfcc': 20, 'include': True, 'deltas': True},
            'spectral': {'include': True},
            'temporal': {'include': True},
            'chroma': {'include': False},
            'mel': {'include': True, 'n_mels': 128},
        }
        
        process_dataset_to_csv(
            input_dir=args.input_dir,
            output_csv=features_csv,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            feature_config=feature_config,
            batch_size=args.batch_size,
            max_files=args.max_files
        )
    
    # Analyze dataset
    if not args.skip_analysis:
        print("\nAnalyzing features dataset...")
        try:
            df = load_and_analyze_features(features_csv)
            visualize_feature_distributions(df, analysis_dir)
            visualize_feature_importance(df, analysis_dir)
            print(f"Analysis complete. Visualizations saved to {analysis_dir}")
        except Exception as e:
            print(f"Error during analysis: {e}")
    
    print("\nPreprocessing pipeline complete!")

if __name__ == "__main__":
    main() 