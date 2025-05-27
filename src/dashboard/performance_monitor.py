"""
Performance monitoring dashboard for the RAG system.

This module provides a dashboard to track cache hit rates and response times
for the RAG system over time, allowing for continuous performance monitoring
and optimization.
"""
import os
import json
import logging
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESULTS_DIR = os.path.join("data", "test_results")
DEFAULT_OUTPUT_DIR = os.path.join("data", "dashboard")


def load_test_results(results_dir: str = DEFAULT_RESULTS_DIR) -> List[Dict[str, Any]]:
    """
    Load all test results from the specified directory.
    
    Args:
        results_dir: Directory containing test result JSON files
        
    Returns:
        List of test result dictionaries
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory {results_dir} does not exist")
        return results
    
    for file_path in results_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                # Add filename and timestamp
                result["filename"] = file_path.name
                result["timestamp"] = datetime.datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")
                results.append(result)
                logger.info(f"Loaded test results from {file_path}")
        except Exception as e:
            logger.error(f"Error loading test results from {file_path}: {str(e)}")
    
    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""))
    
    return results


def create_performance_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a performance summary dataframe from test results.
    
    Args:
        results: List of test result dictionaries
        
    Returns:
        DataFrame with performance summary
    """
    summary_data = []
    
    for result in results:
        config = result.get("system_config", {})
        summary = result.get("summary", {})
        
        row = {
            "timestamp": result.get("timestamp", ""),
            "filename": result.get("filename", ""),
            "use_quantized": config.get("use_quantized", False),
            "similarity_threshold": config.get("semantic_similarity_threshold", 0.0),
            "avg_processing_time": summary.get("avg_processing_time", 0.0),
            "avg_original_query_time": summary.get("avg_original_query_time", 0.0),
            "avg_similar_query_time": summary.get("avg_similar_query_time", 0.0),
            "time_improvement_percentage": summary.get("time_improvement_percentage", 0.0),
            "cache_hit_rate": summary.get("cache_hit_rate", 0.0),
            "semantic_cache_hit_rate": summary.get("semantic_cache_hit_rate", 0.0),
            "total_queries": summary.get("total_queries", 0),
            "cache_hits": summary.get("cache_hits", 0),
            "semantic_cache_hits": summary.get("semantic_cache_hits", 0)
        }
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def plot_response_times(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Plot response times for different configurations.
    
    Args:
        df: DataFrame with performance summary
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by configuration
    grouped = df.groupby(["use_quantized", "similarity_threshold"])
    
    # Plot bars for each configuration
    bar_width = 0.25
    positions = np.arange(len(grouped))
    
    # Plot original query time
    ax.bar(
        positions - bar_width, 
        [group["avg_original_query_time"].mean() for _, group in grouped], 
        width=bar_width, 
        label="Original Query Time"
    )
    
    # Plot similar query time
    ax.bar(
        positions, 
        [group["avg_similar_query_time"].mean() for _, group in grouped], 
        width=bar_width, 
        label="Similar Query Time"
    )
    
    # Plot average processing time
    ax.bar(
        positions + bar_width, 
        [group["avg_processing_time"].mean() for _, group in grouped], 
        width=bar_width, 
        label="Average Processing Time"
    )
    
    # Set labels and title
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Response Time (seconds)")
    ax.set_title("Response Times by Configuration")
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{'Quantized' if q else 'Standard'}, Threshold={t:.2f}" 
        for (q, t) in grouped.groups.keys()
    ], rotation=45, ha="right")
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = output_path / "response_times.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Response times plot saved to {plot_path}")
    
    return str(plot_path)


def plot_cache_hit_rates(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Plot cache hit rates for different configurations.
    
    Args:
        df: DataFrame with performance summary
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by configuration
    grouped = df.groupby(["use_quantized", "similarity_threshold"])
    
    # Plot bars for each configuration
    bar_width = 0.4
    positions = np.arange(len(grouped))
    
    # Plot cache hit rate
    ax.bar(
        positions - bar_width/2, 
        [group["cache_hit_rate"].mean() * 100 for _, group in grouped], 
        width=bar_width, 
        label="Overall Cache Hit Rate (%)"
    )
    
    # Plot semantic cache hit rate
    ax.bar(
        positions + bar_width/2, 
        [group["semantic_cache_hit_rate"].mean() * 100 for _, group in grouped], 
        width=bar_width, 
        label="Semantic Cache Hit Rate (%)"
    )
    
    # Set labels and title
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Cache Hit Rates by Configuration")
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{'Quantized' if q else 'Standard'}, Threshold={t:.2f}" 
        for (q, t) in grouped.groups.keys()
    ], rotation=45, ha="right")
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = output_path / "cache_hit_rates.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Cache hit rates plot saved to {plot_path}")
    
    return str(plot_path)


def plot_time_improvement(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Plot time improvement percentage for different configurations.
    
    Args:
        df: DataFrame with performance summary
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by configuration
    grouped = df.groupby(["use_quantized", "similarity_threshold"])
    
    # Plot bars for each configuration
    positions = np.arange(len(grouped))
    
    # Plot time improvement percentage
    bars = ax.bar(
        positions, 
        [min(group["time_improvement_percentage"].mean(), 100) for _, group in grouped], 
        width=0.6
    )
    
    # Color bars based on improvement (green for positive, red for negative)
    for i, (_, group) in enumerate(grouped):
        improvement = min(group["time_improvement_percentage"].mean(), 100)
        bars[i].set_color("green" if improvement > 0 else "red")
    
    # Set labels and title
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Time Improvement (%)")
    ax.set_title("Time Improvement by Configuration")
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{'Quantized' if q else 'Standard'}, Threshold={t:.2f}" 
        for (q, t) in grouped.groups.keys()
    ], rotation=45, ha="right")
    
    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    
    # Add data labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f"{height:.1f}%",
            ha="center", 
            va="bottom",
            fontweight="bold"
        )
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = output_path / "time_improvement.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Time improvement plot saved to {plot_path}")
    
    return str(plot_path)


def generate_html_report(
    df: pd.DataFrame, 
    response_times_plot: str, 
    cache_hit_rates_plot: str, 
    time_improvement_plot: str,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    """
    Generate an HTML report with performance metrics and plots.
    
    Args:
        df: DataFrame with performance summary
        response_times_plot: Path to response times plot
        cache_hit_rates_plot: Path to cache hit rates plot
        time_improvement_plot: Path to time improvement plot
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report
    """
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG System Performance Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            .plot-container {{
                margin-top: 20px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .plot {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .summary-card {{
                background-color: #e9f7ef;
                border-left: 5px solid #27ae60;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 3px;
            }}
            .summary-value {{
                font-size: 24px;
                font-weight: bold;
                color: #27ae60;
            }}
            .summary-label {{
                font-size: 14px;
                color: #555;
            }}
            .summary-row {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .summary-item {{
                flex: 1;
                min-width: 200px;
                margin: 10px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.05);
            }}
            .timestamp {{
                color: #777;
                font-style: italic;
                text-align: right;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG System Performance Dashboard</h1>
            <p class="timestamp">Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-card">
                <h2>Performance Summary</h2>
                <p>This dashboard shows the performance metrics for the RAG system with different configurations.</p>
            </div>
            
            <div class="summary-row">
                <div class="summary-item">
                    <div class="summary-value">{df["avg_processing_time"].min():.2f}s</div>
                    <div class="summary-label">Best Average Response Time</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{df["cache_hit_rate"].max() * 100:.1f}%</div>
                    <div class="summary-label">Best Cache Hit Rate</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{min(df["time_improvement_percentage"].max(), 100):.1f}%</div>
                    <div class="summary-label">Best Time Improvement</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{df["similarity_threshold"].min():.2f}</div>
                    <div class="summary-label">Lowest Similarity Threshold</div>
                </div>
            </div>
            
            <h2>Response Times</h2>
            <div class="plot-container">
                <img src="{os.path.basename(response_times_plot)}" alt="Response Times" class="plot">
            </div>
            
            <h2>Cache Hit Rates</h2>
            <div class="plot-container">
                <img src="{os.path.basename(cache_hit_rates_plot)}" alt="Cache Hit Rates" class="plot">
            </div>
            
            <h2>Time Improvement</h2>
            <div class="plot-container">
                <img src="{os.path.basename(time_improvement_plot)}" alt="Time Improvement" class="plot">
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Configuration</th>
                    <th>Avg. Processing Time (s)</th>
                    <th>Cache Hit Rate (%)</th>
                    <th>Semantic Cache Hit Rate (%)</th>
                    <th>Time Improvement (%)</th>
                </tr>
    """
    
    # Add rows for each configuration
    for (use_quantized, similarity_threshold), group in df.groupby(["use_quantized", "similarity_threshold"]):
        config_name = f"{'Quantized' if use_quantized else 'Standard'}, Threshold={similarity_threshold:.2f}"
        avg_time = group["avg_processing_time"].mean()
        cache_hit_rate = group["cache_hit_rate"].mean() * 100
        semantic_hit_rate = group["semantic_cache_hit_rate"].mean() * 100
        time_improvement = min(group["time_improvement_percentage"].mean(), 100)
        
        html_content += f"""
                <tr>
                    <td>{config_name}</td>
                    <td>{avg_time:.2f}</td>
                    <td>{cache_hit_rate:.1f}%</td>
                    <td>{semantic_hit_rate:.1f}%</td>
                    <td>{time_improvement:.1f}%</td>
                </tr>
        """
    
    # Complete HTML content
    html_content += """
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Use the quantized embedding model for better performance</li>
                <li>Lower the similarity threshold to increase semantic cache hit rates</li>
                <li>Monitor cache size and performance over time to optimize settings</li>
            </ul>
            
            <div class="timestamp">
                <p>RAG System Performance Dashboard - Cascade AI</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save HTML report
    report_path = output_path / "performance_dashboard.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {report_path}")
    
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Generate performance dashboard for RAG system")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory containing test results")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save dashboard")
    
    args = parser.parse_args()
    
    # Load test results
    results = load_test_results(args.results_dir)
    
    if not results:
        logger.error(f"No test results found in {args.results_dir}")
        return
    
    # Create performance summary
    df = create_performance_summary(results)
    
    # Generate plots
    response_times_plot = plot_response_times(df, args.output_dir)
    cache_hit_rates_plot = plot_cache_hit_rates(df, args.output_dir)
    time_improvement_plot = plot_time_improvement(df, args.output_dir)
    
    # Generate HTML report
    report_path = generate_html_report(
        df, 
        response_times_plot, 
        cache_hit_rates_plot, 
        time_improvement_plot,
        args.output_dir
    )
    
    logger.info(f"Performance dashboard generated at {report_path}")
    print(f"\nPerformance dashboard generated at {report_path}")
    print(f"Open this file in a web browser to view the dashboard.")


if __name__ == "__main__":
    main()
