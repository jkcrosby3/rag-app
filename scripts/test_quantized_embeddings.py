"""Script to test the performance of quantized embeddings.

This script compares the performance of standard and quantized embedding generators
for the RAG system.
"""
import sys
import logging
import time
import numpy as np
from pathlib import Path
import argparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.quantized_generator import QuantizedEmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_test_data():
    """Load test data from the embedded documents directory."""
    data_dir = Path("data/embedded")
    
    # Check if directory exists
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return []
    
    # Load text from embedded documents
    texts = []
    for file_path in data_dir.glob("*.embedded*.json"):
        try:
            # Just use the filename as a test string
            texts.append(file_path.stem)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    # Add some synthetic test data
    texts.extend([
        "What were the key provisions of the Glass-Steagall Act?",
        "How did the Glass-Steagall Act separate commercial and investment banking?",
        "What was the purpose of the SEC during the Great Depression?",
        "Compare the New Deal programs with Glass-Steagall regulations.",
        "What were the major financial reforms after the Great Depression?"
    ])
    
    return texts

def test_embedding_performance(
    texts,
    quantized=False,
    quantization_type="int8",
    batch_size=5,
    num_runs=3
):
    """Test the performance of embedding generation.
    
    Args:
        texts: List of texts to generate embeddings for
        quantized: Whether to use quantized embeddings
        quantization_type: Type of quantization to use
        batch_size: Batch size for embedding generation
        num_runs: Number of runs to average over
        
    Returns:
        Dict with performance metrics
    """
    # Initialize the appropriate embedding generator
    if quantized:
        logger.info(f"Testing quantized embedding generator with {quantization_type} quantization")
        generator = QuantizedEmbeddingGenerator(quantization_type=quantization_type)
    else:
        logger.info("Testing standard embedding generator")
        generator = EmbeddingGenerator()
    
    # Warm up the model
    logger.info("Warming up the model...")
    generator.generate_embedding("Warm up text")
    
    # Test single embedding generation
    single_times = []
    for _ in range(num_runs):
        # Select a random text
        text = np.random.choice(texts)
        
        # Time the embedding generation
        start_time = time.time()
        embedding = generator.generate_embedding(text)
        generation_time = time.time() - start_time
        
        single_times.append(generation_time)
        
    # Test batch embedding generation
    batch_times = []
    for _ in range(num_runs):
        # Select random texts
        batch_texts = np.random.choice(texts, size=batch_size, replace=True).tolist()
        
        # Time the batch embedding generation
        start_time = time.time()
        embeddings = generator.generate_embeddings(batch_texts)
        generation_time = time.time() - start_time
        
        batch_times.append(generation_time)
    
    # Calculate metrics
    avg_single_time = sum(single_times) / len(single_times)
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_per_text_in_batch = avg_batch_time / batch_size
    
    return {
        "avg_single_time": avg_single_time,
        "avg_batch_time": avg_batch_time,
        "avg_per_text_in_batch": avg_per_text_in_batch,
        "embedding_dimension": len(embedding),
        "num_runs": num_runs,
        "batch_size": batch_size
    }

def main():
    """Run the embedding performance test."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test embedding performance')
    parser.add_argument('--quantized', action='store_true', help='Test quantized embeddings')
    parser.add_argument('--quantization-type', type=str, choices=['int8', 'int4'], default='int8',
                        help='Type of quantization to use')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for embedding generation')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs to average over')
    parser.add_argument('--compare', action='store_true', help='Compare standard and quantized embeddings')
    
    args = parser.parse_args()
    
    # Load test data
    texts = load_test_data()
    if not texts:
        logger.error("No test data found")
        return 1
    
    logger.info(f"Loaded {len(texts)} test texts")
    
    # Run the tests
    if args.compare:
        # Test standard embeddings
        standard_metrics = test_embedding_performance(
            texts,
            quantized=False,
            batch_size=args.batch_size,
            num_runs=args.num_runs
        )
        
        # Test quantized embeddings
        quantized_metrics = test_embedding_performance(
            texts,
            quantized=True,
            quantization_type=args.quantization_type,
            batch_size=args.batch_size,
            num_runs=args.num_runs
        )
        
        # Print comparison
        print("\n" + "="*80)
        print("EMBEDDING PERFORMANCE COMPARISON")
        print("="*80)
        print(f"Standard Embedding Generator:")
        print(f"  - Average time for single embedding: {standard_metrics['avg_single_time']:.4f} seconds")
        print(f"  - Average time for batch of {args.batch_size}: {standard_metrics['avg_batch_time']:.4f} seconds")
        print(f"  - Average time per text in batch: {standard_metrics['avg_per_text_in_batch']:.4f} seconds")
        print(f"  - Embedding dimension: {standard_metrics['embedding_dimension']}")
        print("\nQuantized Embedding Generator:")
        print(f"  - Quantization type: {args.quantization_type}")
        print(f"  - Average time for single embedding: {quantized_metrics['avg_single_time']:.4f} seconds")
        print(f"  - Average time for batch of {args.batch_size}: {quantized_metrics['avg_batch_time']:.4f} seconds")
        print(f"  - Average time per text in batch: {quantized_metrics['avg_per_text_in_batch']:.4f} seconds")
        print(f"  - Embedding dimension: {quantized_metrics['embedding_dimension']}")
        print("\nPerformance Improvement:")
        single_speedup = standard_metrics['avg_single_time'] / quantized_metrics['avg_single_time']
        batch_speedup = standard_metrics['avg_batch_time'] / quantized_metrics['avg_batch_time']
        print(f"  - Single embedding speedup: {single_speedup:.2f}x")
        print(f"  - Batch embedding speedup: {batch_speedup:.2f}x")
        print("="*80)
    else:
        # Test only one type of embeddings
        metrics = test_embedding_performance(
            texts,
            quantized=args.quantized,
            quantization_type=args.quantization_type,
            batch_size=args.batch_size,
            num_runs=args.num_runs
        )
        
        # Print results
        print("\n" + "="*80)
        print(f"{'QUANTIZED' if args.quantized else 'STANDARD'} EMBEDDING PERFORMANCE")
        print("="*80)
        print(f"Average time for single embedding: {metrics['avg_single_time']:.4f} seconds")
        print(f"Average time for batch of {args.batch_size}: {metrics['avg_batch_time']:.4f} seconds")
        print(f"Average time per text in batch: {metrics['avg_per_text_in_batch']:.4f} seconds")
        print(f"Embedding dimension: {metrics['embedding_dimension']}")
        print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
