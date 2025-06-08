import csv
import os
from datetime import import datetime
from pathlib import Path
from dotenv import load_dotenv

from core.agents.response_rewriter.agent import RewriterAgent
from core.agents.response_rewriter.states import RewriteState
from prompts import PROMPT_MAP, get_prompt_for_situation, HYBRID_PROMPT

# Load environment variables
load_dotenv()

def evaluate_tsv_reader(reader: csv.DictReader, writer: csv.DictWriter) -> list:
    """Process TSV reader and save results to writer. Returns list of errors."""
    errors = []
    for i, row in enumerate(reader, 1):
        state = RewriteState(user_text=row["Query"], bixby_text=row["BixbyResponse"])
        try:
            # First, get classification
            state = agent.invoke_graph(state, agent.config)
            
            # Now use smart prompt selection based on classification
            selected_prompt = get_prompt_for_situation(
                query_type=state.type,
                difficulty=state.difficulty,
                special_case=detect_special_case(row["Query"], row["BixbyResponse"])
            )
            
            # Apply the selected prompt
            enhanced_config = agent.config.copy()
            enhanced_config['selected_prompt'] = selected_prompt
            enhanced_config['prompt_strategy'] = get_strategy_name(selected_prompt)
            
            # Re-invoke with selected prompt
            state = agent.invoke_graph(state, enhanced_config)
            
            row["RewrittenResponse"] = state.rewritten_text if hasattr(state, 'rewritten_text') else ""
            row["QueryType"] = state.type if hasattr(state, 'type') else ""
            row["Difficulty"] = state.difficulty if hasattr(state, 'difficulty') else ""
            row["PromptStrategy"] = enhanced_config.get('prompt_strategy', 'default')
            
            print(f"Processed row {i}: {row['Query'][:30]}... -> {row['PromptStrategy']}")
        except Exception as e:
            errors.append((i, str(e)))
            row["RewrittenResponse"] = f"Error: {str(e)}"
            row["QueryType"] = "ERROR"
            row["Difficulty"] = "ERROR"
            row["PromptStrategy"] = "ERROR"
        
        writer.writerow(row)
    return errors

def detect_special_case(query: str, response: str) -> str:
    """Detect if this needs special handling."""
    # Error patterns
    error_keywords = ["error", "not found", "cannot", "failed", "unable"]
    if any(keyword in response.lower() for keyword in error_keywords):
        return "error"
    
    # Sensitive topics
    sensitive_keywords = ["health", "medical", "personal", "private", "death", "suicide"]
    if any(keyword in query.lower() for keyword in sensitive_keywords):
        return "sensitive"
    
    # Ambiguous queries
    if len(query.split()) < 3 and "?" not in query:
        return "ambiguous"
    
    return None

def get_strategy_name(prompt_template: str) -> str:
    """Get the strategy name from the prompt template."""
    for name, template in PROMPT_MAP.items():
        if template == prompt_template:
            return name
    return "custom"

def evaluate_tsv_file(input_path: str, output_dir: str):
    """Process TSV file and save results with enhanced prompt selection."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"rewritten_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
    
    with open(input_path, "r", newline="") as infile, open(output_path, "w", newline="") as outfile:
        reader = csv.DictReader(infile, delimiter="\t")
        fieldnames = reader.fieldnames + ["RewrittenResponse", "QueryType", "Difficulty", "PromptStrategy"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        
        errors = evaluate_tsv_reader(reader, writer)
        
    if errors:
        print(f"\nErrors encountered:")
        for i, error in errors:
            print(f"Row {i}: {error}")
    
    print(f"\nOutput saved to: {output_path}")
    
    # Generate statistics
    generate_prompt_statistics(output_path)

def generate_prompt_statistics(file_path: str):
    """Generate statistics about which prompts were used."""
    stats = {}
    total = 0
    
    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            strategy = row.get("PromptStrategy", "unknown")
            if strategy not in stats:
                stats[strategy] = {
                    "count": 0,
                    "types": {},
                    "difficulties": {}
                }
            
            stats[strategy]["count"] += 1
            
            # Track query types
            query_type = row.get("QueryType", "unknown")
            stats[strategy]["types"][query_type] = stats[strategy]["types"].get(query_type, 0) + 1
            
            # Track difficulties
            difficulty = row.get("Difficulty", "unknown")
            stats[strategy]["difficulties"][difficulty] = stats[strategy]["difficulties"].get(difficulty, 0) + 1
            
            total += 1
    
    print("\n=== Prompt Strategy Statistics ===")
    for strategy, data in sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"\n{strategy.upper()}: {data['count']} ({data['count']/total*100:.1f}%)")
        print(f"  Query Types: {dict(sorted(data['types'].items()))}")
        print(f"  Difficulties: {dict(sorted(data['difficulties'].items()))}")

def interactive_mode(agent):
    """Run interactive mode with enhanced prompt selection."""
    while True:
        user = input("User: ")
        if user == "exit":
            break
        
        bixby = input("Bixby: ")
        
        # Create initial state
        state = RewriteState(user_text=user, bixby_text=bixby)
        
        # Get classification first
        state = agent.invoke_graph(state, agent.config)
        
        # Detect special cases
        special_case = detect_special_case(user, bixby)
        
        # Smart prompt selection
        selected_prompt = get_prompt_for_situation(
            query_type=getattr(state, 'type', 'GENERAL_KNOWLEDGE'),
            difficulty=getattr(state, 'difficulty', 'MIDDLE_SCHOOL'),
            special_case=special_case
        )
        
        strategy_name = get_strategy_name(selected_prompt)
        print(f"\n[Using {strategy_name} strategy for {state.type}/{state.difficulty}]")
        
        # Apply selected prompt
        enhanced_config = agent.config.copy()
        enhanced_config['selected_prompt'] = selected_prompt
        
        # Get rewritten response
        state = agent.invoke_graph(state, enhanced_config)
        
        if hasattr(state, 'rewritten_text'):
            print(f"Rewritten: {state.rewritten_text}")
        else:
            print("No rewritten response generated")
        print()

def benchmark_prompts(agent, test_cases_file: str):
    """Benchmark different prompt strategies on the same inputs."""
    results = []
    
    with open(test_cases_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        test_cases = list(reader)[:10]  # Limit to 10 for benchmarking
    
    print("=== Benchmarking Prompt Strategies ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['Query'][:50]}...")
        results_row = {
            "Query": test_case["Query"],
            "Original": test_case["BixbyResponse"]
        }
        
        # Test each prompt strategy
        for strategy_name, prompt_template in PROMPT_MAP.items():
            state = RewriteState(
                user_text=test_case["Query"],
                bixby_text=test_case["BixbyResponse"]
            )
            
            config = agent.config.copy()
            config['selected_prompt'] = prompt_template
            
            try:
                state = agent.invoke_graph(state, config)
                results_row[strategy_name] = getattr(state, 'rewritten_text', 'Error')
                print(f"  - {strategy_name}: ✓")
            except Exception as e:
                results_row[strategy_name] = f"Error: {str(e)}"
                print(f"  - {strategy_name}: ✗ ({str(e)[:30]}...)")
        
        results.append(results_row)
        print()
    
    # Save benchmark results
    output_path = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ["Query", "Original"] + list(PROMPT_MAP.keys())
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Benchmark results saved to: {output_path}")

def test_hybrid_prompt(agent):
    """Test the hybrid prompt approach."""
    print("=== Testing Hybrid Prompt Strategy ===\n")
    
    test_queries = [
        ("What is photosynthesis?", "Plants make food from sunlight"),
        ("How are you feeling today?", "I am functioning normally"),
        ("Calculate the area of a circle with radius 5", "The area is 78.5"),
        ("What is the meaning of life?", "Life's meaning varies by perspective")
    ]
    
    for query, original in test_queries:
        print(f"Query: {query}")
        print(f"Original: {original}")
        
        state = RewriteState(user_text=query, bixby_text=original)
        
        # Use hybrid prompt
        config = agent.config.copy()
        config['selected_prompt'] = HYBRID_PROMPT
        
        state = agent.invoke_graph(state, config)
        print(f"Hybrid Rewrite: {getattr(state, 'rewritten_text', 'No output')}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    load_dotenv()
    
    provider = os.getenv("provider")
    model = os.getenv("model")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    
    agent = RewriterAgent("rewriter_agent", provider, model)
    agent.build_graph()
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Input TSV file path",
        default="core/agents/response_rewriter/resource/ResponseRewriterEvaluationDB.tsv",
    )
    parser.add_argument("--output", help="Output directory", default="core/agents/response_rewriter/resource")
    parser.add_argument("--interactive", help="Run in interactive mode", action="store_true")
    parser.add_argument("--web", help="Run web interface", action="store_true")
    parser.add_argument("--benchmark", help="Benchmark all prompt strategies", action="store_true")
    parser.add_argument("--hybrid", help="Test hybrid prompt approach", action="store_true")
    
    args = parser.parse_args()
    
    if args.web:
        # Launch the Gradio app
        import subprocess
        import sys
        print("Launching web interface...")
        subprocess.run([sys.executable, "app.py"])
    elif args.interactive:
        interactive_mode(agent)
    elif args.benchmark:
        benchmark_prompts(agent, args.input)
    elif args.hybrid:
        test_hybrid_prompt(agent)
    else:
        evaluate_tsv_file(args.input, args.output)
