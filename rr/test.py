"""
Distributed Response Rewriter Arena - Gradio Only Version
No external database required - uses in-memory shared state
"""

import gradio as gr
import pandas as pd
import random
import threading
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

# Import your agent components
from core.agents.response_rewriter.agent import RewriterAgent
from core.agents.response_rewriter.states import RewriteState
from core.agents.response_rewriter.prompts import PROMPT_MAP

# Global shared state with thread safety
class SharedArenaState:
    """Thread-safe shared state for all users."""
    
    def __init__(self):
        self.lock = threading.RLock()
        
        # Battle data
        self.battles = {}  # battle_id -> battle_info
        self.battle_counter = 0
        
        # Voting data
        self.votes = defaultdict(lambda: defaultdict(list))  # battle_id -> choice -> [voter_ids]
        self.user_votes = defaultdict(dict)  # user_id -> battle_id -> choice
        
        # ELO ratings
        self.elo_ratings = {
            strategy: {'rating': 1500, 'battles': 0, 'wins': 0, 'losses': 0, 'ties': 0}
            for strategy in PROMPT_MAP.keys()
        }
        
        # Session tracking
        self.active_sessions = {}  # session_id -> last_seen
        self.session_counter = 0
        
        # TSV tracking
        self.used_indices = set()
        
        # Statistics
        self.total_votes = 0
        self.completed_battles = 0
    
    def create_battle(self, user_text: str, bixby_text: str,
                     strategy_a: str, strategy_b: str,
                     response_a: str, response_b: str,
                     tsv_index: int = None) -> int:
        """Create a new battle."""
        with self.lock:
            self.battle_counter += 1
            battle_id = self.battle_counter
            
            self.battles[battle_id] = {
                'id': battle_id,
                'timestamp': datetime.now().isoformat(),
                'user_text': user_text,
                'bixby_text': bixby_text,
                'strategy_a': strategy_a,
                'strategy_b': strategy_b,
                'response_a': response_a,
                'response_b': response_b,
                'winner': None,
                'tsv_index': tsv_index,
                'created_at': time.time()
            }
            
            if tsv_index is not None:
                self.used_indices.add(tsv_index)
            
            return battle_id
    
    def record_vote(self, battle_id: int, user_id: str, choice: str) -> Dict:
        """Record a vote from a user."""
        with self.lock:
            # Check if battle exists
            if battle_id not in self.battles:
                return {'success': False, 'message': 'Battle not found'}
            
            # Check if already voted
            if battle_id in self.user_votes.get(user_id, {}):
                prev_choice = self.user_votes[user_id][battle_id]
                return {
                    'success': False, 
                    'message': f'Already voted: {prev_choice}',
                    'already_voted': True
                }
            
            # Record vote
            self.votes[battle_id][choice].append(user_id)
            self.user_votes[user_id][battle_id] = choice
            self.total_votes += 1
            
            # Check if battle is complete (5 votes)
            total_battle_votes = sum(len(voters) for voters in self.votes[battle_id].values())
            
            if total_battle_votes >= 5 and self.battles[battle_id]['winner'] is None:
                # Determine winner
                vote_counts = {
                    choice: len(voters) 
                    for choice, voters in self.votes[battle_id].items()
                }
                
                a_votes = vote_counts.get('A is better', 0)
                b_votes = vote_counts.get('B is better', 0)
                tie_votes = vote_counts.get('Tie', 0)
                
                battle = self.battles[battle_id]
                
                if a_votes > b_votes and a_votes > tie_votes:
                    winner = battle['strategy_a']
                    loser = battle['strategy_b']
                elif b_votes > a_votes and b_votes > tie_votes:
                    winner = battle['strategy_b']
                    loser = battle['strategy_a']
                else:
                    winner = 'tie'
                    loser = 'tie'
                
                # Update battle
                battle['winner'] = winner
                battle['final_votes'] = vote_counts
                self.completed_battles += 1
                
                # Update ELO ratings
                if winner != 'tie':
                    self._update_elo(winner, loser)
            
            return {
                'success': True,
                'message': 'Vote recorded!',
                'total_votes': total_battle_votes,
                'battle_complete': total_battle_votes >= 5
            }
    
    def _update_elo(self, winner: str, loser: str, k: int = 32):
        """Update ELO ratings."""
        w_stats = self.elo_ratings[winner]
        l_stats = self.elo_ratings[loser]
        
        # Calculate expected scores
        expected_w = 1 / (1 + 10 ** ((l_stats['rating'] - w_stats['rating']) / 400))
        expected_l = 1 / (1 + 10 ** ((w_stats['rating'] - l_stats['rating']) / 400))
        
        # Update ratings
        w_stats['rating'] += k * (1 - expected_w)
        l_stats['rating'] += k * (0 - expected_l)
        
        # Update stats
        w_stats['battles'] += 1
        w_stats['wins'] += 1
        l_stats['battles'] += 1
        l_stats['losses'] += 1
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get current leaderboard."""
        with self.lock:
            data = []
            for strategy, stats in self.elo_ratings.items():
                win_rate = (stats['wins'] / stats['battles'] * 100) if stats['battles'] > 0 else 0
                data.append({
                    'Strategy': strategy.upper(),
                    'ELO Rating': round(stats['rating']),
                    'Battles': stats['battles'],
                    'Wins': stats['wins'],
                    'Losses': stats['losses'],
                    'Win Rate': f"{win_rate:.1f}%"
                })
            
            df = pd.DataFrame(data).sort_values('ELO Rating', ascending=False).reset_index(drop=True)
            
            if not df.empty:
                df.insert(0, 'Rank', ['🥇', '🥈', '🥉'] + [str(i) for i in range(4, len(df) + 1)])
            
            return df
    
    def get_battle_status(self, battle_id: int) -> Dict:
        """Get current status of a battle."""
        with self.lock:
            if battle_id not in self.battles:
                return None
            
            battle = self.battles[battle_id]
            vote_counts = {
                choice: len(voters) 
                for choice, voters in self.votes[battle_id].items()
            }
            total = sum(vote_counts.values())
            
            return {
                'battle': battle,
                'vote_counts': vote_counts,
                'total_votes': total,
                'votes_needed': max(0, 5 - total),
                'is_complete': battle['winner'] is not None
            }
    
    def get_active_battles(self) -> List[Dict]:
        """Get battles still accepting votes."""
        with self.lock:
            active = []
            for battle_id, battle in self.battles.items():
                if battle['winner'] is None:
                    status = self.get_battle_status(battle_id)
                    active.append({
                        'id': battle_id,
                        'strategies': f"{battle['strategy_a']} vs {battle['strategy_b']}",
                        'votes': status['total_votes'],
                        'needed': status['votes_needed']
                    })
            
            return sorted(active, key=lambda x: x['votes'], reverse=True)[:10]
    
    def track_session(self, session_id: str):
        """Track active session."""
        with self.lock:
            self.active_sessions[session_id] = time.time()
    
    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        with self.lock:
            # Clean old sessions (5 min timeout)
            current_time = time.time()
            self.active_sessions = {
                sid: last_seen 
                for sid, last_seen in self.active_sessions.items()
                if current_time - last_seen < 300
            }
            
            recent_battles = sorted(
                [b for b in self.battles.values() if b['winner'] is not None],
                key=lambda x: x['timestamp'],
                reverse=True
            )[:10]
            
            return {
                'total_battles': len(self.battles),
                'completed_battles': self.completed_battles,
                'total_votes': self.total_votes,
                'active_users': len(self.active_sessions),
                'recent_battles': recent_battles
            }


# Initialize global state
ARENA_STATE = SharedArenaState()

# Load TSV data
def load_tsv_data(path: str) -> List[Dict]:
    """Load TSV data."""
    import csv
    data = []
    
    if not os.path.exists(path):
        print(f"Warning: TSV file not found at {path}")
        # Create some dummy data
        return [
            {'user_text': 'Hello', 'bixby_text': 'Hi there', 'index': 0},
            {'user_text': 'How are you?', 'bixby_text': 'I am good', 'index': 1},
            {'user_text': 'What is the weather?', 'bixby_text': 'It is sunny', 'index': 2},
        ]
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if 'Query' in row and 'BixbyResponse' in row:
                data.append({
                    'user_text': row['Query'],
                    'bixby_text': row['BixbyResponse'],
                    'index': i
                })
    
    print(f"Loaded {len(data)} entries from TSV")
    return data

# Create the distributed interface
def create_interface(agent, tsv_data: List[Dict]) -> gr.Blocks:
    """Create the Gradio interface."""
    
    def get_session_id(request: gr.Request) -> str:
        """Get or create session ID."""
        session_id = getattr(request, 'session_hash', None) or f"user_{int(time.time() * 1000)}"
        ARENA_STATE.track_session(session_id)
        return session_id
    
    def generate_battle(request: gr.Request) -> Tuple:
        """Generate a new battle."""
        session_id = get_session_id(request)
        
        # Get unused TSV entry
        available_indices = set(range(len(tsv_data))) - ARENA_STATE.used_indices
        if not available_indices:
            # Reset if all used
            ARENA_STATE.used_indices.clear()
            available_indices = set(range(len(tsv_data)))
        
        idx = random.choice(list(available_indices))
        entry = tsv_data[idx]
        
        # Select strategies
        strategies = list(PROMPT_MAP.keys())
        strategy_a, strategy_b = random.sample(strategies, 2)
        
        # Generate responses
        try:
            # Process with agent
            state_a = RewriteState(user_text=entry['user_text'], bixby_text=entry['bixby_text'])
            config_a = agent.config.copy()
            config_a['prompt_strategy'] = strategy_a
            result_a = agent.invoke_graph(state_a, config_a)
            response_a = result_a.rewritten_text if hasattr(result_a, 'rewritten_text') else entry['bixby_text']
            
            state_b = RewriteState(user_text=entry['user_text'], bixby_text=entry['bixby_text'])
            config_b = agent.config.copy()
            config_b['prompt_strategy'] = strategy_b
            result_b = agent.invoke_graph(state_b, config_b)
            response_b = result_b.rewritten_text if hasattr(result_b, 'rewritten_text') else entry['bixby_text']
            
            # Create battle
            battle_id = ARENA_STATE.create_battle(
                entry['user_text'], entry['bixby_text'],
                strategy_a, strategy_b,
                response_a, response_b,
                idx
            )
            
            preview = f"""### Query:\n{entry['user_text']}\n\n### Original:\n{entry['bixby_text']}"""
            
            return (
                preview, response_a, response_b, battle_id,
                gr.update(visible=True), "",
                f"Battle #{battle_id} - Vote now!",
                gr.update(visible=True, value=format_vote_status(battle_id))
            )
            
        except Exception as e:
            return (
                f"Error: {str(e)}", "", "", None,
                gr.update(visible=False), "", "", gr.update(visible=False)
            )
    
    def submit_vote(choice: str, battle_id: int, request: gr.Request) -> Tuple:
        """Submit a vote."""
        if not battle_id:
            return "No active battle!", ARENA_STATE.get_leaderboard(), "", gr.update()
        
        session_id = get_session_id(request)
        result = ARENA_STATE.record_vote(battle_id, session_id, choice)
        
        if not result['success']:
            return f"❌ {result['message']}", ARENA_STATE.get_leaderboard(), "", gr.update()
        
        status = ARENA_STATE.get_battle_status(battle_id)
        
        if status['is_complete']:
            battle = status['battle']
            reveal = f"""### 🎭 Battle Complete!\n\n**Model A**: {battle['strategy_a'].upper()}\n**Model B**: {battle['strategy_b'].upper()}\n\n**Winner**: {battle['winner'].upper() if battle['winner'] != 'tie' else 'TIE'}"""
            vote_status = gr.update(visible=False)
        else:
            reveal = f"Vote recorded! ({status['total_votes']}/5 votes)"
            vote_status = gr.update(visible=True, value=format_vote_status(battle_id))
        
        return "✅ " + result['message'], ARENA_STATE.get_leaderboard(), reveal, vote_status
    
    def format_vote_status(battle_id: int) -> str:
        """Format voting status."""
        status = ARENA_STATE.get_battle_status(battle_id)
        if not status:
            return ""
        
        html = f'<div style="background: #f0f4ff; padding: 1rem; border-radius: 8px; text-align: center;">'
        html += f'<h4>🔴 Live Voting - Battle #{battle_id}</h4>'
        
        for choice in ['A is better', 'B is better', 'Tie']:
            count = status['vote_counts'].get(choice, 0)
            bar = '█' * (count * 4) + '░' * ((5 - count) * 4)
            html += f'<div>{choice}: {bar} {count}</div>'
        
        html += f'<br><strong>{status["votes_needed"]} more votes needed!</strong></div>'
        return html
    
    def get_active_battles_df() -> pd.DataFrame:
        """Get active battles as dataframe."""
        battles = ARENA_STATE.get_active_battles()
        if not battles:
            return pd.DataFrame({"Message": ["No active battles"]})
        
        return pd.DataFrame([{
            "Battle": f"#{b['id']}",
            "Matchup": b['strategies'],
            "Votes": f"{b['votes']}/5",
            "Status": "🔴 Needs votes!" if b['needed'] > 0 else "✅"
        } for b in battles])
    
    def get_stats() -> Tuple:
        """Get statistics."""
        stats = ARENA_STATE.get_statistics()
        
        recent_df = pd.DataFrame([{
            "Time": datetime.fromisoformat(b['timestamp']).strftime("%H:%M"),
            "Battle": f"#{b['id']}",
            "Winner": b['winner'].upper() if b['winner'] != 'tie' else 'TIE',
            "Votes": sum(b.get('final_votes', {}).values())
        } for b in stats['recent_battles']]) if stats['recent_battles'] else pd.DataFrame()
        
        return (
            str(stats['total_battles']),
            str(stats['completed_battles']),
            str(stats['total_votes']),
            str(stats['active_users']),
            recent_df
        )
    
    # Build interface
    with gr.Blocks(theme=gr.themes.Soft(), title="Distributed Arena") as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 2rem;">
            <h1>⚔️ Distributed Response Rewriter Arena</h1>
            <p>Vote with others to find the best strategy!</p>
        </div>
        """)
        
        battle_id_state = gr.State()
        
        with gr.Tabs():
            with gr.Tab("🏟️ Arena"):
                with gr.Row():
                    with gr.Column(scale=3):
                        input_preview = gr.Markdown("Click 'Generate Battle' to start!")
                        generate_btn = gr.Button("⚔️ Generate New Battle", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        battle_header = gr.Markdown("### Ready!")
                        vote_status = gr.HTML(visible=False)
                
                with gr.Row():
                    response_a = gr.Textbox(label="Response A", lines=8, interactive=False)
                    response_b = gr.Textbox(label="Response B", lines=8, interactive=False)
                
                with gr.Row(visible=False) as vote_section:
                    vote_a = gr.Button("A is better", variant="primary", size="lg")
                    vote_tie = gr.Button("Tie", variant="secondary", size="lg")
                    vote_b = gr.Button("B is better", variant="primary", size="lg")
                
                vote_result = gr.Markdown()
                reveal_models = gr.Markdown()
                
                gr.Markdown("### Active Battles")
                active_battles = gr.DataFrame(value=get_active_battles_df())
            
            with gr.Tab("🏆 Leaderboard"):
                leaderboard = gr.DataFrame(value=ARENA_STATE.get_leaderboard())
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            with gr.Tab("📊 Statistics"):
                with gr.Row():
                    total_battles = gr.Textbox(label="Total Battles", value="0")
                    completed_battles = gr.Textbox(label="Completed", value="0")
                    total_votes = gr.Textbox(label="Total Votes", value="0")
                    active_users = gr.Textbox(label="Active Users", value="0")
                
                recent_battles = gr.DataFrame(label="Recent Battles")
                stats_refresh = gr.Button("🔄 Refresh Stats", size="sm")
        
        # Event handlers
        generate_btn.click(
            generate_battle,
            outputs=[input_preview, response_a, response_b, battle_id_state,
                    vote_section, vote_result, battle_header, vote_status]
        )
        
        for btn, choice in [(vote_a, "A is better"), (vote_tie, "Tie"), (vote_b, "B is better")]:
            btn.click(
                lambda bid, c=choice: submit_vote(c, bid),
                inputs=[battle_id_state],
                outputs=[vote_result, leaderboard, reveal_models, vote_status]
            )
        
        refresh_btn.click(lambda: ARENA_STATE.get_leaderboard(), outputs=[leaderboard])
        stats_refresh.click(get_stats, outputs=[total_battles, completed_battles, total_votes, active_users, recent_battles])
        
        # Auto-refresh
        demo.load(lambda: ARENA_STATE.get_leaderboard(), outputs=[leaderboard], every=5)
        demo.load(get_stats, outputs=[total_battles, completed_battles, total_votes, active_users, recent_battles], every=5)
        demo.load(get_active_battles_df, outputs=[active_battles], every=3)
    
    return demo


# Main entry point
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize agent
    provider = os.getenv("provider", "openai")
    model = os.getenv("model", "gpt-3.5-turbo")
    
    print("🤖 Initializing agent...")
    agent = RewriterAgent("rewriter_agent", provider, model)
    agent.build_graph()
    
    # Load TSV data
    tsv_path = "core/agents/response_rewriter/resource/ResponseRewriterEvaluationDB.tsv"
    tsv_data = load_tsv_data(tsv_path)
    
    # Create interface
    demo = create_interface(agent, tsv_data)
    
    # Get your PC's IP address
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n🚀 Launching Distributed Arena...")
    print(f"📍 Local IP: {local_ip}")
    print(f"🌐 Access URL: http://{local_ip}:7860")
    print("\nShare this URL with your users!")
    
    # Launch with queue for concurrent users
    demo.queue(concurrency_count=20)
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,
        share=False,  # Set to True for public URL
        show_error=True,
        max_threads=40
    )
