"""
Response Rewriter Test Script
Supports both arena mode and traditional evaluation
"""

import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

# Import utilities
from arena_utils import ArenaManager
from tsv_utils import TSVDataManager
from ui_components import (
    ARENA_CSS, create_header, create_arena_section, 
    create_battle_section, create_leaderboard_tab,
    create_statistics_tab, create_about_tab,
    format_reveal_message, format_battle_preview
)
from evaluation_utils import (
    evaluate_tsv_file, generate_prompt_statistics,
    evaluate_single_entry, benchmark_strategies
)

# Import agent components
from core.agents.response_rewriter.agent import RewriterAgent
from core.agents.response_rewriter.states import RewriteState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ResponseRewriterApp:
    """Main application class for Response Rewriter."""
    
    def __init__(self, agent):
        self.agent = agent
        self.arena_manager = ArenaManager()
        self.tsv_manager = TSVDataManager()
    
    def create_arena_interface(self) -> gr.Blocks:
        """Create the Gradio arena interface."""
        with gr.Blocks(theme=gr.themes.Soft(), css=ARENA_CSS, title="Response Rewriter Arena") as demo:
            # State
            battle_state = gr.State({})
            
            # Header
            create_header()
            
            with gr.Tabs() as tabs:
                # Arena Tab
                with gr.Tab("🏟️ Arena", id=1):
                    # Input section
                    _, input_preview, generate_btn = create_arena_section()
                    
                    # Battle section
                    battle_components = create_battle_section()
                    
                # Leaderboard Tab
                with gr.Tab("🏆 Leaderboard", id=2):
                    leaderboard_components = create_leaderboard_tab()
                
                # Statistics Tab
                with gr.Tab("📈 Statistics", id=3):
                    stats_components = create_statistics_tab(
                        initial_total_entries=len(self.tsv_manager.data)
                    )
                
                # About Tab
                with gr.Tab("ℹ️ About", id=4):
                    create_about_tab()
            
            # Wire up event handlers
            self._setup_event_handlers(
                generate_btn, input_preview, battle_state,
                battle_components, leaderboard_components, stats_components
            )
            
            # Load initial data
            demo.load(
                fn=self._get_initial_data,
                outputs=[
                    leaderboard_components['leaderboard_table'],
                    stats_components['total_battles'],
                    stats_components['total_votes'],
                    stats_components['completion_rate'],
                    stats_components['used_entries'],
                    stats_components['total_entries'],
                    stats_components['recent_battles_table']
                ]
            )
        
        return demo
    
    def _setup_event_handlers(self, generate_btn, input_preview, battle_state,
                            battle_components, leaderboard_components, stats_components):
        """Setup all event handlers for the interface."""
        
        # Generate battle handler
        generate_btn.click(
            fn=self._generate_battle,
            outputs=[
                input_preview,
                battle_components['response_a'],
                battle_components['response_b'],
                battle_state,
                battle_components['vote_section'],
                battle_components['battle_header'],
                battle_components['vote_result'],
                battle_components['reveal_models']
            ]
        )
        
        # Vote handlers
        vote_handlers = [
            (battle_components['vote_a'], "A is better"),
            (battle_components['vote_tie'], "Tie"),
            (battle_components['vote_b'], "B is better")
        ]
        
        for button, choice in vote_handlers:
            button.click(
                fn=lambda b, c=choice: self._record_vote(c, b),
                inputs=[battle_state],
                outputs=[
                    battle_components['vote_result'],
                    leaderboard_components['leaderboard_table'],
                    battle_components['reveal_models'],
                    battle_components['vote_section'],
                    stats_components['total_battles'],
                    stats_components['total_votes'],
                    stats_components['completion_rate']
                ]
            )
        
        # Refresh handlers
        leaderboard_components['refresh_btn'].click(
            fn=lambda: self.arena_manager.get_leaderboard(),
            outputs=[leaderboard_components['leaderboard_table']]
        )
        
        stats_components['stats_refresh_btn'].click(
            fn=self._refresh_all_stats,
            outputs=[
                stats_components['total_battles'],
                stats_components['total_votes'],
                stats_components['completion_rate'],
                stats_components['used_entries'],
                stats_components['total_entries'],
                stats_components['recent_battles_table']
            ]
        )
    
    def _generate_battle(self):
        """Generate a new arena battle."""
        entry = self.tsv_manager.get_random_unused_entry()
        
        if not entry:
            return (
                "❌ No entries available!",
                "", "", {},
                gr.update(visible=False),
                gr.update(visible=False),
                "", ""
            )
        
        try:
            battle_info = self.arena_manager.create_battle(
                entry['user_text'],
                entry['bixby_text'],
                entry['index'],
                self.agent
            )
            
            preview = format_battle_preview(entry['user_text'], entry['bixby_text'])
            
            return (
                preview,
                battle_info['response_a'],
                battle_info['response_b'],
                battle_info,
                gr.update(visible=True),
                gr.update(visible=True),
                "",  # Clear vote result
                ""   # Clear reveal
            )
            
        except Exception as e:
            logger.error(f"Error generating battle: {e}")
            return (
                f"❌ Error: {str(e)}",
                "", "", {},
                gr.update(visible=False),
                gr.update(visible=False),
                "", ""
            )
    
    def _record_vote(self, choice: str, battle_info: dict):
        """Record a vote for the current battle."""
        if not battle_info or 'id' not in battle_info:
            return (
                "❌ Invalid battle",
                self.arena_manager.get_leaderboard(),
                "", gr.update(visible=False),
                0, 0, 0.0
            )
        
        try:
            vote_result = self.arena_manager.record_vote(battle_info['id'], choice)
            
            reveal_msg = format_reveal_message(
                battle_info['strategy_a'],
                battle_info['strategy_b'],
                choice
            )
            
            stats = self.arena_manager.get_statistics()
            
            return (
                "✅ Vote recorded!",
                self.arena_manager.get_leaderboard(),
                reveal_msg,
                gr.update(visible=False),
                stats['total_battles'],
                stats['completed_battles'],
                stats['completion_rate']
            )
            
        except Exception as e:
            logger.error(f"Error recording vote: {e}")
            return (
                f"❌ Error: {str(e)}",
                self.arena_manager.get_leaderboard(),
                "", gr.update(visible=False),
                0, 0, 0.0
            )
    
    def _get_initial_data(self):
        """Get initial data for interface load."""
        stats = self.arena_manager.get_statistics()
        tsv_stats = self.tsv_manager.get_statistics()
        
        return (
            self.arena_manager.get_leaderboard(),
            stats['total_battles'],
            stats['completed_battles'],
            stats['completion_rate'],
            tsv_stats['used_entries'],
            tsv_stats['total_entries'],
            self.arena_manager.get_recent_battles()
        )
    
    def _refresh_all_stats(self):
        """Refresh all statistics."""
        stats = self.arena_manager.get_statistics()
        tsv_stats = self.tsv_manager.get_statistics()
        
        return (
            stats['total_battles'],
            stats['completed_battles'],
            stats['completion_rate'],
            tsv_stats['used_entries'],
            tsv_stats['total_entries'],
            self.arena_manager.get_recent_battles()
        )
    
    def run_interactive_mode(self):
        """Run interactive CLI mode."""
        print("\n=== Interactive Response Rewriter ===")
        print("Type 'exit' to quit\n")
        
        while True:
            user_text = input("User: ")
            if user_text.lower() == 'exit':
                break
            
            bixby_text = input("Bixby (original): ")
            
            print("\nProcessing...")
            result = evaluate_single_entry(self.agent, user_text, bixby_text)
            
            if result['success']:
                print(f"\n[Strategy: {result['strategy_used']}")
                print(f" Type: {result['query_type']}")
                print(f" Difficulty: {result['difficulty']}]")
                print(f"\nRewritten: {result['rewritten_text']}")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
            
            print("\n" + "-" * 50 + "\n")
    
    def run_tsv_evaluation(self, input_path: str, output_dir: str):
        """Run traditional TSV file evaluation."""
        print(f"\n📂 Processing TSV file: {input_path}")
        
        def progress_callback(current, total):
            if current % 10 == 0:
                print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
        
        output_path, errors = evaluate_tsv_file(
            self.agent, input_path, output_dir, progress_callback
        )
        
        print(f"\n✅ Output saved to: {output_path}")
        
        if errors:
            print(f"\n⚠️ Errors encountered: {len(errors)}")
            for row, error in errors[:5]:  # Show first 5 errors
                print(f"  Row {row}: {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
        
        # Generate statistics
        stats = generate_prompt_statistics(output_path)
        print("\n📊 Prompt Strategy Statistics:")
        
        for strategy, data in stats['strategy_stats'].items():
            print(f"\n{strategy.upper()}:")
            print(f"  Count: {data['count']}")
            print(f"  Success Rate: {data['success_rate']:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Response Rewriter - Arena and Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        help="Input TSV file path",
        default="core/agents/response_rewriter/resource/ResponseRewriterEvaluationDB.tsv"
    )
    parser.add_argument(
        "--output",
        help="Output directory",
        default="core/agents/response_rewriter/resource"
    )
    parser.add_argument("--interactive", help="Run interactive CLI mode", action="store_true")
    parser.add_argument("--arena", help="Run arena interface", action="store_true")
    parser.add_argument("--web", help="Run arena interface (alias for --arena)", action="store_true")
    parser.add_argument("--host", help="Host for web interface", default="0.0.0.0")
    parser.add_argument("--port", help="Port for web interface", type=int, default=7860)
    parser.add_argument("--share", help="Create public Gradio link", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize agent
    provider = os.getenv("provider", "openai")
    model = os.getenv("model", "gpt-3.5-turbo")
    
    print(f"🤖 Initializing RewriterAgent...")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    
    try:
        agent = RewriterAgent("rewriter_agent", provider, model)
        agent.build_graph()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1
    
    # Create app instance
    app = ResponseRewriterApp(agent)
    
    # Handle different modes
    if args.arena or args.web:
        # Arena mode
        print(f"\n📂 Loading TSV data from: {args.input}")
        
        if app.tsv_manager.load_from_file(args.input):
            print(f"✅ Loaded {len(app.tsv_manager.data)} entries")
            
            # Validate data
            issues = app.tsv_manager.validate_data()
            if issues['total_issues'] > 0:
                print(f"⚠️ Data quality issues found: {issues}")
            
            print(f"\n🚀 Launching Response Rewriter Arena...")
            print(f"🌐 Interface will be available at: http://{args.host}:{args.port}\n")
            
            demo = app.create_arena_interface()
            demo.launch(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                show_error=True
            )
        else:
            print("❌ Failed to load TSV data")
            return 1
    
    elif args.interactive:
        # Interactive CLI mode
        app.run_interactive_mode()
    
    else:
        # Default: TSV evaluation mode
        app.run_tsv_evaluation(args.input, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
