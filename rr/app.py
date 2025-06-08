import gradio as gr
import requests
import random
from datetime import datetime
from typing import Dict, Tuple, Optional
import pandas as pd
from config import Config
from ratings_db import RatingsDatabase
from prompts import PROMPT_MAP  # Import your PROMPT_MAP from prompts.py

# Initialize ratings database
ratings_db = RatingsDatabase(Config.RATINGS_DB_PATH)

def send_to_backend(user_text: str, bix_response: str, prompt_type: str) -> Tuple[bool, str, Dict]:
    """Send request to Flask backend and get response with classification."""
    try:
        response = requests.post(
            Config.FLASK_URL,
            json={
                "user_text": user_text,
                "bix_response": bix_response,
                "prompt_type": prompt_type
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get('rewritten_response', 'No response received'), data
        else:
            return False, f"Backend error: {response.status_code}", {}
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}", {}

def process_rewrite(user_text: str, bix_response: str, 
                   state: Dict) -> Tuple[str, str, Dict, gr.update, gr.update]:
    """Process the rewrite request."""
    if not user_text or not bix_response:
        return (
            "Please provide both user text and bix response.",
            "",
            state,
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    # Randomly select a prompt type from PROMPT_MAP
    prompt_types = list(PROMPT_MAP.keys())
    selected_prompt_type = random.choice(prompt_types)
    
    # Send to backend
    success, response, backend_data = send_to_backend(user_text, bix_response, selected_prompt_type)
    
    if success:
        # Extract classification info if available
        classification = backend_data.get('classification', {})
        query_type = classification.get('type', 'Unknown')
        difficulty = classification.get('difficulty', 'Unknown')
        
        # Update state with current session info
        state.update({
            'prompt_type': selected_prompt_type,
            'user_text': user_text,
            'bix_response': bix_response,
            'rewritten_response': response,
            'query_type': query_type,
            'difficulty': difficulty,
            'has_response': True
        })
        
        prompt_info = f"Prompt Type: {selected_prompt_type} | Query Type: {query_type} | Difficulty: {difficulty}"
        
        return (
            response,
            prompt_info,
            state,
            gr.update(visible=True, interactive=True),
            gr.update(visible=True)
        )
    else:
        return (
            f"Error: {response}",
            "",
            state,
            gr.update(visible=False),
            gr.update(visible=False)
        )

def submit_rating(rating: int, state: Dict) -> Tuple[str, gr.update, pd.DataFrame]:
    """Submit rating for the current response."""
    if not state.get('has_response'):
        return "No response to rate!", gr.update(value=None), get_stats_dataframe()
    
    # Add rating to database with additional metadata
    ratings_db.add_rating(
        prompt_id=f"{state['prompt_type']}_{state['query_type']}_{state['difficulty']}",
        prompt_template=state['prompt_type'],
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
        get_stats_dataframe()
    )

def get_stats_dataframe() -> pd.DataFrame:
    """Get prompt statistics as a dataframe."""
    top_prompts = ratings_db.get_top_prompts(10)
    
    if not top_prompts:
        return pd.DataFrame({
            "Prompt Strategy": ["No data yet"],
            "Average Rating": [0],
            "Total Ratings": [0]
        })
    
    return pd.DataFrame([
        {
            "Prompt Strategy": p['template'],
            "Average Rating": f"⭐ {p['average_rating']:.2f}",
            "Total Ratings": p['total_ratings'],
            "Prompt ID": p['prompt_id']
        }
        for p in top_prompts
    ])

# Create Gradio interface
with gr.Blocks(title="Response Rewriter Evaluation", theme=gr.themes.Soft()) as demo:
    state = gr.State({})
    
    gr.Markdown("""
    # 📝 Response Rewriter Evaluation
    
    This tool helps evaluate different prompt strategies for rewriting responses.
    It uses intelligent classification to determine query type and difficulty, then applies
    various prompt strategies (default, CoT, few-shot, step-by-step) to generate improved responses.
    """)
    
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
                lines=1
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
    
    gr.Markdown("## 📊 Prompt Strategy Performance")
    
    with gr.Row():
        stats_table = gr.DataFrame(
            value=get_stats_dataframe(),
            label="Top Performing Strategies",
            interactive=False
        )
    
    with gr.Row():
        gr.Markdown("""
        ### 🎯 Prompt Strategies:
        - **Default**: Adaptive personality based on difficulty level
        - **CoT (Chain of Thought)**: Step-by-step reasoning approach
        - **Few-shot**: Learning from examples
        - **Step-by-Step**: Structured, methodical responses
        """)
    
    # Event handlers
    submit_btn.click(
        fn=process_rewrite,
        inputs=[user_text_input, bix_response_input, state],
        outputs=[output_text, prompt_info, state, rating_radio, rate_btn]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[rating_feedback]
    )
    
    rate_btn.click(
        fn=submit_rating,
        inputs=[rating_radio, state],
        outputs=[rating_feedback, rating_radio, stats_table]
    )
    
    # Refresh stats on load
    demo.load(fn=get_stats_dataframe, outputs=[stats_table])

if __name__ == "__main__":
    demo.launch(
        server_name=Config.GRADIO_HOST,
        server_port=Config.GRADIO_PORT,
        share=False
    )
