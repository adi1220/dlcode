"""
Response Rewriter Arena - Modern Gradio Interface
Inspired by TTS-Arena-V2 design
"""

import gradio as gr
import requests
import random
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import pandas as pd
from config import Config
from ratings_db import RatingsDatabase
from prompts import PROMPT_MAP

# Initialize ratings database
ratings_db = RatingsDatabase(Config.RATINGS_DB_PATH)

# Arena battle history
arena_battles = []

# Custom CSS for modern design
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 2rem;
}

/* Arena container */
.arena-container {
    display: flex;
    gap: 20px;
    margin: 20px 0;
}

.model-output {
    flex: 1;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 12px;
    border: 2px solid #e9ecef;
    min-height: 200px;
}

.model-output.selected {
    border-color: #667eea;
    background: #f0f4ff;
}

/* Vote buttons */
.vote-button {
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.3s;
}

.vote-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Leaderboard styling */
.leaderboard-table {
    margin-top: 20px;
}

.rank-1 { color: #FFD700; font-weight: bold; }
.rank-2 { color: #C0C0C0; font-weight: bold; }
.rank-3 { color: #CD7F32; font-weight: bold; }

/* Tab styling */
.tab-nav {
    border-bottom: 2px solid #e9ecef;
    margin-bottom: 20px;
}

/* Stats cards */
.stats-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}

.stats-number {
    font-size: 2.5rem;
    font-weight: bold;
    color: #667eea;
}

/* Model badges */
.model-badge {
    display: inline-block;
    padding: 6px 12px;
    background: #667eea;
    color: white;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* History item */
.history-item {
    padding: 15px;
    margin: 10px 0;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

/* Animated elements */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}
"""

def send_to_backend(user_text: str, bix_response: str, strategy: str) -> Tuple[bool, str, Dict]:
    """Send request to Flask backend."""
    try:
        response = requests.post(
            Config.FLASK_URL,
            json={
                "user_text": user_text,
                "bix_response": bix_response,
                "prompt_type": strategy
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get('rewritten_response', 'No response'), data
        else:
            return False, f"Error: {response.status_code}", {}
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}", {}

def battle_arena(user_text: str, bix_response: str) -> Tuple[str, str, str, str, gr.update, gr.update]:
    """Run arena battle between two random strategies."""
    if not user_text or not bix_response:
        return (
            "Please provide both user query and Bixby response.",
            "",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    # Select two different strategies randomly
    strategies = list(PROMPT_MAP.keys())
    if len(strategies) < 2:
        return "Not enough strategies available", "", "", "", gr.update(), gr.update()
    
    strategy_a, strategy_b = random.sample(strategies, 2)
    
    # Get responses from both strategies
    success_a, response_a, data_a = send_to_backend(user_text, bix_response, strategy_a)
    success_b, response_b, data_b = send_to_backend(user_text, bix_response, strategy_b)
    
    if not success_a or not success_b:
        return (
            response_a if not success_a else "Error with Model B",
            response_b if not success_b else "Error with Model A",
            strategy_a,
            strategy_b,
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    # Store battle info
    battle_id = len(arena_battles)
    arena_battles.append({
        'id': battle_id,
        'timestamp': datetime.now(),
        'user_text': user_text,
        'bix_response': bix_response,
        'strategy_a': strategy_a,
        'strategy_b': strategy_b,
        'response_a': response_a,
        'response_b': response_b,
        'winner': None
    })
    
    return (
        response_a,
        response_b,
        f"Model A: {strategy_a.upper()}",
        f"Model B: {strategy_b.upper()}",
        gr.update(visible=True, value=battle_id),
        gr.update(visible=True)
    )

def record_vote(choice: str, battle_id: int) -> Tuple[str, pd.DataFrame, str]:
    """Record the vote for arena battle."""
    if battle_id >= len(arena_battles):
        return "Invalid battle ID", get_leaderboard(), ""
    
    battle = arena_battles[battle_id]
    
    # Determine winner
    if choice == "A is better":
        winner = battle['strategy_a']
        loser = battle['strategy_b']
    elif choice == "B is better":
        winner = battle['strategy_b']
        loser = battle['strategy_a']
    elif choice == "Tie":
        winner = "tie"
        loser = "tie"
    else:
        return "Invalid choice", get_leaderboard(), ""
    
    # Update battle record
    battle['winner'] = winner
    battle['vote'] = choice
    
    # Update ratings (ELO-style)
    if winner != "tie":
        update_elo_ratings(winner, loser)
        
        # Also record in ratings_db for persistence
        ratings_db.add_rating(
            prompt_id=f"arena_{winner}_vs_{loser}",
            prompt_template=winner,
            user_text=battle['user_text'],
            bix_response=battle['bix_response'],
            rewritten_response=battle[f'response_{"a" if winner == battle["strategy_a"] else "b"}'],
            rating=5  # Winner gets 5 stars
        )
        
        ratings_db.add_rating(
            prompt_id=f"arena_{loser}_vs_{winner}",
            prompt_template=loser,
            user_text=battle['user_text'],
            bix_response=battle['bix_response'],
            rewritten_response=battle[f'response_{"b" if loser == battle["strategy_b"] else "a"}'],
            rating=2  # Loser gets 2 stars
        )
    
    # Create reveal message
    reveal_msg = f"""
    ### 🎭 Models Revealed!
    
    **Model A** was: **{battle['strategy_a'].upper()}**
    **Model B** was: **{battle['strategy_b'].upper()}**
    
    You voted: **{choice}**
    """
    
    return "Vote recorded! Thank you.", get_leaderboard(), reveal_msg

# ELO rating system
elo_ratings = {strategy: 1500 for strategy in PROMPT_MAP.keys()}

def update_elo_ratings(winner: str, loser: str, k=32):
    """Update ELO ratings based on battle result."""
    if winner not in elo_ratings:
        elo_ratings[winner] = 1500
    if loser not in elo_ratings:
        elo_ratings[loser] = 1500
    
    # Expected scores
    expected_winner = 1 / (1 + 10 ** ((elo_ratings[loser] - elo_ratings[winner]) / 400))
    expected_loser = 1 / (1 + 10 ** ((elo_ratings[winner] - elo_ratings[loser]) / 400))
    
    # Update ratings
    elo_ratings[winner] += k * (1 - expected_winner)
    elo_ratings[loser] += k * (0 - expected_loser)

def get_leaderboard() -> pd.DataFrame:
    """Get current leaderboard based on ELO ratings."""
    # Combine ELO ratings with win statistics
    stats = {}
    
    for strategy in PROMPT_MAP.keys():
        wins = sum(1 for b in arena_battles if b['winner'] == strategy)
        losses = sum(1 for b in arena_battles if 
                    (b['strategy_a'] == strategy and b['winner'] == b['strategy_b']) or
                    (b['strategy_b'] == strategy and b['winner'] == b['strategy_a']))
        ties = sum(1 for b in arena_battles if 
                  (b['strategy_a'] == strategy or b['strategy_b'] == strategy) and b['winner'] == 'tie')
        
        total_battles = wins + losses + ties
        win_rate = (wins / total_battles * 100) if total_battles > 0 else 0
        
        stats[strategy] = {
            'Strategy': strategy.upper(),
            'ELO Rating': round(elo_ratings.get(strategy, 1500)),
            'Battles': total_battles,
            'Wins': wins,
            'Losses': losses,
            'Ties': ties,
            'Win Rate': f"{win_rate:.1f}%"
        }
    
    # Create dataframe and sort by ELO
    df = pd.DataFrame(list(stats.values()))
    df = df.sort_values('ELO Rating', ascending=False).reset_index(drop=True)
    
    # Add rank column
    df.insert(0, 'Rank', ['🥇', '🥈', '🥉'] + [f"{i}" for i in range(4, len(df) + 1)])
    
    return df

def get_statistics() -> Tuple[str, str, str, pd.DataFrame]:
    """Get overall statistics."""
    total_battles = len(arena_battles)
    total_votes = sum(1 for b in arena_battles if b.get('winner') is not None)
    
    # Get most recent battles
    recent_battles = []
    for battle in reversed(arena_battles[-10:]):  # Last 10 battles
        if battle.get('winner'):
            recent_battles.append({
                'Time': battle['timestamp'].strftime('%H:%M:%S'),
                'Query': battle['user_text'][:30] + '...',
                'Model A': battle['strategy_a'].upper(),
                'Model B': battle['strategy_b'].upper(),
                'Winner': battle['winner'].upper() if battle['winner'] != 'tie' else 'TIE'
            })
    
    recent_df = pd.DataFrame(recent_battles) if recent_battles else pd.DataFrame()
    
    return (
        str(total_battles),
        str(total_votes),
        f"{(total_votes/total_battles*100):.1f}%" if total_battles > 0 else "0%",
        recent_df
    )

def side_by_side_comparison(user_text: str, bix_response: str, 
                          strategy_a: str, strategy_b: str) -> Tuple[str, str, str]:
    """Compare two specific strategies side by side."""
    if not user_text or not bix_response:
        return "Please provide inputs", "", ""
    
    # Get responses
    success_a, response_a, _ = send_to_backend(user_text, bix_response, strategy_a)
    success_b, response_b, _ = send_to_backend(user_text, bix_response, strategy_b)
    
    if not success_a:
        response_a = f"Error: {response_a}"
    if not success_b:
        response_b = f"Error: {response_b}"
    
    comparison = f"""
    ### 📊 Comparison Summary
    
    **Strategy A ({strategy_a})**: {len(response_a)} characters
    **Strategy B ({strategy_b})**: {len(response_b)} characters
    
    **Difference**: {abs(len(response_a) - len(response_b))} characters
    """
    
    return response_a, response_b, comparison

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Response Rewriter Arena") as demo:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">⚔️ Response Rewriter Arena</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Battle different prompt strategies to find the best rewriter
        </p>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        # Arena Tab
        with gr.Tab("🏟️ Arena", id=1):
            gr.Markdown("### Enter your query and original response to start a battle")
            
            with gr.Row():
                with gr.Column(scale=1):
                    user_input = gr.Textbox(
                        label="User Query",
                        placeholder="Enter the user's question...",
                        lines=2
                    )
                    bix_input = gr.Textbox(
                        label="Original Bixby Response",
                        placeholder="Enter Bixby's original response...",
                        lines=2
                    )
                    
                    arena_btn = gr.Button(
                        "⚔️ Start Arena Battle",
                        variant="primary",
                        size="lg",
                        elem_classes=["vote-button"]
                    )
            
            # Battle outputs
            with gr.Row():
                with gr.Column():
                    model_a_output = gr.Textbox(
                        label="Model A",
                        lines=6,
                        interactive=False,
                        elem_classes=["model-output"]
                    )
                    model_a_name = gr.Markdown("**Model A**: Hidden")
                
                with gr.Column():
                    model_b_output = gr.Textbox(
                        label="Model B",
                        lines=6,
                        interactive=False,
                        elem_classes=["model-output"]
                    )
                    model_b_name = gr.Markdown("**Model B**: Hidden")
            
            # Voting section
            battle_id_state = gr.State()
            
            with gr.Row(visible=False) as voting_row:
                gr.Markdown("### 🗳️ Which response is better?")
            
            with gr.Row(visible=False) as vote_buttons:
                vote_a = gr.Button("A is better", variant="primary", size="lg")
                vote_tie = gr.Button("Tie", variant="secondary", size="lg")
                vote_b = gr.Button("B is better", variant="primary", size="lg")
            
            vote_result = gr.Markdown("")
            reveal_models = gr.Markdown("")
        
        # Leaderboard Tab
        with gr.Tab("🏆 Leaderboard", id=2):
            gr.Markdown("## 📊 Strategy Rankings")
            gr.Markdown("Rankings based on ELO rating system from arena battles")
            
            leaderboard_table = gr.DataFrame(
                value=get_leaderboard(),
                interactive=False,
                elem_classes=["leaderboard-table"]
            )
            
            refresh_btn = gr.Button("🔄 Refresh Leaderboard", size="sm")
        
        # Side-by-Side Tab
        with gr.Tab("🔬 Compare", id=3):
            gr.Markdown("## Compare Two Strategies Directly")
            
            with gr.Row():
                with gr.Column():
                    compare_user = gr.Textbox(
                        label="User Query",
                        placeholder="Enter query...",
                        lines=2
                    )
                    compare_bix = gr.Textbox(
                        label="Original Response",
                        placeholder="Enter original response...",
                        lines=2
                    )
                
                with gr.Column():
                    strategy_choices = list(PROMPT_MAP.keys())
                    compare_a = gr.Dropdown(
                        choices=strategy_choices,
                        value=strategy_choices[0],
                        label="Strategy A"
                    )
                    compare_b = gr.Dropdown(
                        choices=strategy_choices,
                        value=strategy_choices[1] if len(strategy_choices) > 1 else strategy_choices[0],
                        label="Strategy B"
                    )
                    
                    compare_btn = gr.Button("🔬 Compare", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    compare_output_a = gr.Textbox(
                        label="Strategy A Output",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Column():
                    compare_output_b = gr.Textbox(
                        label="Strategy B Output",
                        lines=8,
                        interactive=False
                    )
            
            compare_summary = gr.Markdown()
        
        # Statistics Tab
        with gr.Tab("📈 Statistics", id=4):
            gr.Markdown("## Arena Statistics")
            
            with gr.Row():
                with gr.Column():
                    total_battles_stat = gr.Textbox(
                        label="Total Battles",
                        value="0",
                        interactive=False,
                        elem_classes=["stats-card"]
                    )
                
                with gr.Column():
                    total_votes_stat = gr.Textbox(
                        label="Total Votes",
                        value="0",
                        interactive=False,
                        elem_classes=["stats-card"]
                    )
                
                with gr.Column():
                    completion_rate_stat = gr.Textbox(
                        label="Completion Rate",
                        value="0%",
                        interactive=False,
                        elem_classes=["stats-card"]
                    )
            
            gr.Markdown("### Recent Battles")
            recent_battles_table = gr.DataFrame(
                interactive=False
            )
            
            stats_refresh_btn = gr.Button("🔄 Refresh Statistics", size="sm")
        
        # About Tab
        with gr.Tab("ℹ️ About", id=5):
            gr.Markdown("""
            ## About Response Rewriter Arena
            
            This arena allows you to compare different prompt engineering strategies for rewriting responses:
            
            ### Available Strategies:
            
            1. **DEFAULT** - Adaptive personality based on difficulty
            2. **COT** (Chain of Thought) - Step-by-step reasoning
            3. **FEWSHOT** - Learning from examples
            4. **STEPBYSTEP** - Systematic structured approach
            5. **EDGE_CASE** - Special handling for errors and edge cases
            
            ### How it Works:
            
            1. **Arena Mode**: Two random strategies compete head-to-head
            2. **Blind Voting**: Models are hidden until after you vote
            3. **ELO Rating**: Strategies are ranked using chess-style ELO system
            4. **Fair Comparison**: Each strategy gets the same input
            
            ### Tips:
            
            - Try different types of queries to see which strategies excel where
            - The leaderboard updates in real-time as more battles complete
            - Use the Compare tab to directly test specific strategies
            """)
    
    # Event handlers
    arena_btn.click(
        fn=battle_arena,
        inputs=[user_input, bix_input],
        outputs=[model_a_output, model_b_output, model_a_name, model_b_name, 
                battle_id_state, voting_row]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[vote_buttons]
    )
    
    # Vote handlers
    vote_a.click(
        fn=lambda bid: record_vote("A is better", bid),
        inputs=[battle_id_state],
        outputs=[vote_result, leaderboard_table, reveal_models]
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[voting_row, vote_buttons]
    )
    
    vote_tie.click(
        fn=lambda bid: record_vote("Tie", bid),
        inputs=[battle_id_state],
        outputs=[vote_result, leaderboard_table, reveal_models]
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[voting_row, vote_buttons]
    )
    
    vote_b.click(
        fn=lambda bid: record_vote("B is better", bid),
        inputs=[battle_id_state],
        outputs=[vote_result, leaderboard_table, reveal_models]
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[voting_row, vote_buttons]
    )
    
    # Refresh handlers
    refresh_btn.click(
        fn=get_leaderboard,
        outputs=[leaderboard_table]
    )
    
    stats_refresh_btn.click(
        fn=get_statistics,
        outputs=[total_battles_stat, total_votes_stat, completion_rate_stat, recent_battles_table]
    )
    
    # Compare handlers
    compare_btn.click(
        fn=side_by_side_comparison,
        inputs=[compare_user, compare_bix, compare_a, compare_b],
        outputs=[compare_output_a, compare_output_b, compare_summary]
    )
    
    # Load initial data
    demo.load(
        fn=lambda: (get_leaderboard(), *get_statistics()),
        outputs=[leaderboard_table, total_battles_stat, total_votes_stat, 
                completion_rate_stat, recent_battles_table]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=Config.GRADIO_HOST,
        server_port=Config.GRADIO_PORT,
        share=False
    )
