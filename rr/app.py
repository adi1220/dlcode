import gradio as gr
import requests
import random
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import pandas as pd
from config import Config
from ratings_db import RatingsDatabase
from prompts import PROMPT_MAP, get_prompt_for_situation, HYBRID_PROMPT

# Initialize ratings database
ratings_db = RatingsDatabase(Config.RATINGS_DB_PATH)

def detect_special_case(query: str, response: str) -> str:
    """Detect if this needs special handling."""
    # Error patterns
    error_keywords = ["error", "not found", "cannot", "failed", "unable", "no device"]
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

def send_to_backend(user_text: str, bix_response: str, prompt_strategy: str, 
                   prompt_template: str = None) -> Tuple[bool, str, Dict]:
    """Send request to Flask backend with enhanced prompt selection."""
    try:
        request_data = {
            "user_text": user_text,
            "bix_response": bix_response,
            "prompt_type": prompt_strategy
        }
        
        # If using smart selection, include the actual prompt template
        if prompt_template:
            request_data["prompt_template"] = prompt_template
        
        response = requests.post(
            Config.FLASK_URL,
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get('rewritten_response', 'No response received'), data
        else:
            return False, f"Backend error: {response.status_code}", {}
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}", {}

def process_rewrite(user_text: str, bix_response: str, use_smart_selection: bool,
                   manual_strategy: str, state: Dict) -> Tuple[str, str, Dict, gr.update, gr.update, gr.update]:
    """Process the rewrite request with enhanced prompt selection."""
    if not user_text or not bix_response:
        return (
            "Please provide both user text and bix response.",
            "",
            state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    # Detect special cases
    special_case = detect_special_case(user_text, bix_response)
    
    if use_smart_selection:
        # First, get classification by calling backend with default prompt
        _, _, classification_data = send_to_backend(user_text, bix_response, "default")
        
        classification = classification_data.get('classification', {})
        query_type = classification.get('type', 'GENERAL_KNOWLEDGE')
        difficulty = classification.get('difficulty', 'MIDDLE_SCHOOL')
        
        # Smart prompt selection
        selected_prompt = get_prompt_for_situation(
            query_type=query_type,
            difficulty=difficulty,
            special_case=special_case
        )
        
        # Find which strategy this prompt belongs to
        prompt_strategy = "custom"
        for name, template in PROMPT_MAP.items():
            if template == selected_prompt:
                prompt_strategy = name
                break
        
        # Send with selected prompt
        success, response, backend_data = send_to_backend(
            user_text, bix_response, prompt_strategy, selected_prompt
        )
    else:
        # Use manual strategy selection or random
        if manual_strategy == "random":
            prompt_strategy = random.choice(list(PROMPT_MAP.keys()))
        else:
            prompt_strategy = manual_strategy
        
        success, response, backend_data = send_to_backend(
            user_text, bix_response, prompt_strategy
        )
        
        classification = backend_data.get('classification', {})
        query_type = classification.get('type', 'Unknown')
        difficulty = classification.get('difficulty', 'Unknown')
    
    if success:
        # Update state with current session info
        state.update({
            'prompt_strategy': prompt_strategy,
            'user_text': user_text,
            'bix_response': bix_response,
            'rewritten_response': response,
            'query_type': query_type,
            'difficulty': difficulty,
            'special_case': special_case or "none",
            'has_response': True
        })
        
        prompt_info = f"Strategy: {prompt_strategy} | Type: {query_type} | Difficulty: {difficulty}"
        if special_case:
            prompt_info += f" | Special: {special_case}"
        
        return (
            response,
            prompt_info,
            state,
            gr.update(visible=True, interactive=True),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    else:
        return (
            f"Error: {response}",
            "",
            state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def submit_rating(rating: int, state: Dict) -> Tuple[str, gr.update, pd.DataFrame, pd.DataFrame]:
    """Submit rating for the current response."""
    if not state.get('has_response'):
        return "No response to rate!", gr.update(value=None), get_stats_dataframe(), get_difficulty_stats()
    
    # Add rating to database with additional metadata
    ratings_db.add_rating(
        prompt_id=f"{state['prompt_strategy']}_{state['query_type']}_{state['difficulty']}",
        prompt_template=state['prompt_strategy'],
        user_text=state['user_text'],
        bix_response=state['bix_response'],
        rewritten_response=state['rewritten_response'],
        rating=rating
    )
    
    # Reset state
    state['has_response'] = False
    
    return (
        f"Thank you! Rating {rating} stars submitted successfully.",
        gr.update(value=None),
        get_stats_dataframe(),
        get_difficulty_stats()
    )

def get_stats_dataframe() -> pd.DataFrame:
    """Get prompt statistics as a dataframe."""
    top_prompts = ratings_db.get_top_prompts(15)
    
    if not top_prompts:
        return pd.DataFrame({
            "Prompt Strategy": ["No data yet"],
            "Average Rating": [0],
            "Total Ratings": [0],
            "Best For": ["N/A"]
        })
    
    # Analyze which strategies work best for what
    stats_data = []
    for p in top_prompts:
        prompt_id = p['prompt_id']
        parts = prompt_id.split('_')
        
        if len(parts) >= 3:
            strategy = parts[0]
            query_type = parts[1]
            difficulty = '_'.join(parts[2:]) if len(parts) > 3 else parts[2]
            
            stats_data.append({
                "Prompt Strategy": strategy,
                "Average Rating": f"⭐ {p['average_rating']:.2f}",
                "Total Ratings": p['total_ratings'],
                "Best For": f"{query_type} ({difficulty})"
            })
        else:
            stats_data.append({
                "Prompt Strategy": p['template'],
                "Average Rating": f"⭐ {p['average_rating']:.2f}",
                "Total Ratings": p['total_ratings'],
                "Best For": "General"
            })
    
    return pd.DataFrame(stats_data)

def get_difficulty_stats() -> pd.DataFrame:
    """Get statistics by difficulty level."""
    all_stats = ratings_db.get_prompt_stats()
    
    difficulty_data = {}
    for prompt_id, data in all_stats.items():
        parts = prompt_id.split('_')
        if len(parts) >= 3:
            difficulty = '_'.join(parts[2:]) if len(parts) > 3 else parts[2]
            strategy = parts[0]
            
            if difficulty not in difficulty_data:
                difficulty_data[difficulty] = {}
            
            if strategy not in difficulty_data[difficulty]:
                difficulty_data[difficulty][strategy] = {
                    'ratings': [],
                    'count': 0
                }
            
            difficulty_data[difficulty][strategy]['ratings'].extend(data['ratings'])
            difficulty_data[difficulty][strategy]['count'] += data['count']
    
    # Format for display
    rows = []
    for difficulty, strategies in difficulty_data.items():
        for strategy, data in strategies.items():
            if data['ratings']:
                avg_rating = sum(data['ratings']) / len(data['ratings'])
                rows.append({
                    "Difficulty Level": difficulty,
                    "Best Strategy": strategy,
                    "Avg Rating": f"⭐ {avg_rating:.2f}",
                    "Sample Size": data['count']
                })
    
    if not rows:
        return pd.DataFrame({
            "Difficulty Level": ["No data yet"],
            "Best Strategy": ["N/A"],
            "Avg Rating": ["N/A"],
            "Sample Size": [0]
        })
    
    return pd.DataFrame(rows).sort_values("Avg Rating", ascending=False)

def benchmark_strategies(user_text: str, bix_response: str) -> pd.DataFrame:
    """Benchmark all strategies on the same input."""
    results = []
    
    for strategy in PROMPT_MAP.keys():
        success, response, data = send_to_backend(user_text, bix_response, strategy)
        
        if success:
            # Truncate response for display
            display_response = response[:100] + "..." if len(response) > 100 else response
            results.append({
                "Strategy": strategy.upper(),
                "Response Preview": display_response,
                "Length": len(response),
                "Status": "✓ Success"
            })
        else:
            results.append({
                "Strategy": strategy.upper(),
                "Response Preview": "Failed to generate",
                "Length": 0,
                "Status": "✗ Error"
            })
    
    return pd.DataFrame(results)

# Create Gradio interface
with gr.Blocks(title="Response Rewriter Evaluation", theme=gr.themes.Soft()) as demo:
    state = gr.State({})
    
    gr.Markdown("""
    # 📝 Response Rewriter Evaluation - Enhanced Edition
    
    This tool evaluates different prompt strategies for rewriting responses using:
    - **Smart Selection**: AI chooses the best strategy based on query analysis
    - **Manual Testing**: Compare specific strategies
    - **Special Handling**: Automatic detection of errors, sensitive topics, and edge cases
    """)
    
    with gr.Tabs():
        with gr.TabItem("🚀 Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    user_text_input = gr.Textbox(
                        label="User Query",
                        placeholder="Enter the user's question or request...",
                        lines=3
                    )
                    
                    bix_response_input = gr.Textbox(
                        label="Bixby Response",
                        placeholder="Enter the original Bixby response to improve...",
                        lines=3
                    )
                    
                    with gr.Row():
                        use_smart = gr.Checkbox(
                            label="Use Smart Prompt Selection",
                            value=True,
                            info="AI selects best strategy based on query analysis"
                        )
                        
                        manual_strategy = gr.Dropdown(
                            choices=["random"] + list(PROMPT_MAP.keys()) + ["hybrid"],
                            value="random",
                            label="Manual Strategy Selection",
                            info="Used when Smart Selection is OFF"
                        )
                    
                    submit_btn = gr.Button("🚀 Rewrite Response", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="Rewritten Response",
                        lines=6,
                        interactive=False
                    )
                    
                    prompt_info = gr.Textbox(
                        label="Classification & Strategy Info",
                        interactive=False,
                        lines=2
                    )
                    
                    with gr.Row():
                        rating_radio = gr.Radio(
                            choices=[("⭐", 1), ("⭐⭐", 2), ("⭐⭐⭐", 3), ("⭐⭐⭐⭐", 4), ("⭐⭐⭐⭐⭐", 5)],
                            label="Rate the rewritten response",
                            visible=False
                        )
                        
                        rate_btn = gr.Button("Submit Rating", visible=False, variant="secondary")
                    
                    rating_feedback = gr.Textbox(
                        label="Feedback",
                        interactive=False,
                        lines=1,
                        visible=False
                    )
        
        with gr.TabItem("📊 Statistics"):
            gr.Markdown("## 📈 Performance Analytics")
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Statistics", size="sm")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Strategy Performance")
                    stats_table = gr.DataFrame(
                        value=get_stats_dataframe(),
                        label="Top Performing Strategies",
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### Performance by Difficulty")
                    difficulty_table = gr.DataFrame(
                        value=get_difficulty_stats(),
                        label="Best Strategies by Difficulty Level",
                        interactive=False
                    )
        
        with gr.TabItem("🔬 Benchmark"):
            gr.Markdown("## 🔬 Strategy Comparison")
            gr.Markdown("Compare all strategies on the same input to see differences")
            
            with gr.Row():
                with gr.Column():
                    bench_user = gr.Textbox(
                        label="Test Query",
                        placeholder="Enter a test query...",
                        lines=2
                    )
                    bench_bixby = gr.Textbox(
                        label="Test Response", 
                        placeholder="Enter original response...",
                        lines=2
                    )
                    bench_btn = gr.Button("🔬 Run Benchmark", variant="primary")
            
            benchmark_results = gr.DataFrame(
                label="Benchmark Results",
                interactive=False
            )
        
        with gr.TabItem("ℹ️ Guide"):
            gr.Markdown("""
            ## 📚 Prompt Strategies Guide
            
            ### Available Strategies:
            
            1. **Default** 🎯
               - Adaptive personality based on difficulty level
               - Best for: General queries across all levels
               - Adjusts tone from playful (PRIMARY) to scholarly (PHD)
            
            2. **Chain of Thought (CoT)** 🧠
               - Step-by-step reasoning approach
               - Best for: Complex queries requiring analysis
               - Shows thinking process transparently
            
            3. **Few-Shot** 📖
               - Learning from high-quality examples
               - Best for: Queries similar to trained patterns
               - Maintains consistency across similar inputs
            
            4. **Step-by-Step** 📋
               - Systematic, structured responses
               - Best for: Technical or instructional content
               - Clear phases: Analysis → Planning → Execution
            
            5. **Edge Case** ⚠️
               - Special handling for difficult situations
               - Auto-triggered for: Errors, sensitive topics, ambiguous queries
               - Focuses on being helpful despite challenges
            
            6. **Hybrid** 🔄
               - Combines best aspects of all strategies
               - Experimental approach for maximum quality
               - Analyzes → References → Structures → Adapts
            
            ### Smart Selection Logic:
            - **Errors** → Edge Case handler
            - **Sensitive Topics** → Edge Case with extra care
            - **Philosophy + University/PHD** → Chain of Thought
            - **Primary/Middle School** → Few-Shot examples
            - **STEM Queries** → Step-by-Step structure
            - **General** → Adaptive default
            
            ### Tips for Best Results:
            1. Use Smart Selection for optimal automatic strategy
            2. Test edge cases with error messages or sensitive topics
            3. Compare strategies using the Benchmark tab
            4. Rate responses to improve the system
            """)
    
    # Event handlers
    submit_btn.click(
        fn=process_rewrite,
        inputs=[user_text_input, bix_response_input, use_smart, manual_strategy, state],
        outputs=[output_text, prompt_info, state, rating_radio, rate_btn, rating_feedback]
    )
    
    rate_btn.click(
        fn=submit_rating,
        inputs=[rating_radio, state],
        outputs=[rating_feedback, rating_radio, stats_table, difficulty_table]
    )
    
    refresh_btn.click(
        fn=lambda: (get_stats_dataframe(), get_difficulty_stats()),
        outputs=[stats_table, difficulty_table]
    )
    
    bench_btn.click(
        fn=benchmark_strategies,
        inputs=[bench_user, bench_bixby],
        outputs=[benchmark_results]
    )
    
    # Smart selection toggle
    use_smart.change(
        fn=lambda x: gr.update(interactive=not x),
        inputs=[use_smart],
        outputs=[manual_strategy]
    )
    
    # Refresh stats on load
    demo.load(
        fn=lambda: (get_stats_dataframe(), get_difficulty_stats()),
        outputs=[stats_table, difficulty_table]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=Config.GRADIO_HOST,
        server_port=Config.GRADIO_PORT,
        share=False
    )
