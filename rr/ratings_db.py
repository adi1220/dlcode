"""
Enhanced Ratings Database for Response Rewriter Arena
Supports both traditional ratings and ELO-based arena battles
"""

import json
import os
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import statistics
from collections import defaultdict


class RatingsDatabase:
    """Enhanced database for ratings and arena battles."""
    
    def __init__(self, db_path: str = 'data/ratings.json', arena_path: str = 'data/arena.json'):
        self.db_path = db_path
        self.arena_path = arena_path
        self.lock = Lock()
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(arena_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_db()
        self._initialize_arena()
    
    def _initialize_db(self):
        """Initialize the ratings database file if it doesn't exist."""
        if not os.path.exists(self.db_path):
            with self.lock:
                with open(self.db_path, 'w') as f:
                    json.dump({
                        'ratings': [],
                        'prompt_stats': {},
                        'metadata': {
                            'created_at': datetime.now().isoformat(),
                            'version': '2.0'
                        }
                    }, f, indent=2)
    
    def _initialize_arena(self):
        """Initialize the arena database file if it doesn't exist."""
        if not os.path.exists(self.arena_path):
            with self.lock:
                with open(self.arena_path, 'w') as f:
                    json.dump({
                        'battles': [],
                        'elo_ratings': {
                            'default': 1500,
                            'cot': 1500,
                            'fewshot': 1500,
                            'stepbystep': 1500,
                            'edge_case': 1500
                        },
                        'statistics': {},
                        'metadata': {
                            'created_at': datetime.now().isoformat(),
                            'version': '1.0'
                        }
                    }, f, indent=2)
    
    def add_rating(self, prompt_id: str, prompt_template: str, 
                   user_text: str, bix_response: str, 
                   rewritten_response: str, rating: int,
                   metadata: Optional[Dict] = None) -> Dict:
        """Add a new rating to the database."""
        with self.lock:
            # Load current data
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            # Create rating entry
            rating_entry = {
                'id': len(data['ratings']) + 1,
                'timestamp': datetime.now().isoformat(),
                'prompt_id': prompt_id,
                'prompt_template': prompt_template,
                'user_text': user_text,
                'bix_response': bix_response,
                'rewritten_response': rewritten_response,
                'rating': rating,
                'metadata': metadata or {}
            }
            
            # Add to ratings list
            data['ratings'].append(rating_entry)
            
            # Update prompt statistics
            if prompt_id not in data['prompt_stats']:
                data['prompt_stats'][prompt_id] = {
                    'template': prompt_template,
                    'ratings': [],
                    'count': 0,
                    'average': 0.0,
                    'std_dev': 0.0,
                    'min': 5,
                    'max': 1
                }
            
            stats = data['prompt_stats'][prompt_id]
            stats['ratings'].append(rating)
            stats['count'] += 1
            stats['average'] = statistics.mean(stats['ratings'])
            stats['min'] = min(stats['ratings'])
            stats['max'] = max(stats['ratings'])
            
            if len(stats['ratings']) > 1:
                stats['std_dev'] = statistics.stdev(stats['ratings'])
            
            # Save updated data
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return rating_entry
    
    def add_arena_battle(self, strategy_a: str, strategy_b: str,
                        user_text: str, bix_response: str,
                        response_a: str, response_b: str,
                        winner: str, metadata: Optional[Dict] = None) -> Dict:
        """Add an arena battle result."""
        with self.lock:
            # Load current data
            with open(self.arena_path, 'r') as f:
                arena_data = json.load(f)
            
            # Create battle entry
            battle = {
                'id': len(arena_data['battles']) + 1,
                'timestamp': datetime.now().isoformat(),
                'strategy_a': strategy_a,
                'strategy_b': strategy_b,
                'user_text': user_text,
                'bix_response': bix_response,
                'response_a': response_a,
                'response_b': response_b,
                'winner': winner,
                'metadata': metadata or {}
            }
            
            # Add to battles list
            arena_data['battles'].append(battle)
            
            # Update ELO ratings
            if winner != 'tie':
                self._update_elo_ratings(arena_data, strategy_a, strategy_b, winner)
            
            # Update statistics
            self._update_arena_statistics(arena_data, strategy_a, strategy_b, winner)
            
            # Save updated data
            with open(self.arena_path, 'w') as f:
                json.dump(arena_data, f, indent=2)
            
            return battle
    
    def _update_elo_ratings(self, arena_data: Dict, strategy_a: str, 
                           strategy_b: str, winner: str, k: int = 32):
        """Update ELO ratings based on battle result."""
        ratings = arena_data['elo_ratings']
        
        # Ensure strategies exist in ratings
        if strategy_a not in ratings:
            ratings[strategy_a] = 1500
        if strategy_b not in ratings:
            ratings[strategy_b] = 1500
        
        # Current ratings
        rating_a = ratings[strategy_a]
        rating_b = ratings[strategy_b]
        
        # Expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))
        
        # Actual scores
        if winner == strategy_a:
            score_a, score_b = 1, 0
        elif winner == strategy_b:
            score_a, score_b = 0, 1
        else:  # tie
            score_a, score_b = 0.5, 0.5
        
        # Update ratings
        ratings[strategy_a] = rating_a + k * (score_a - expected_a)
        ratings[strategy_b] = rating_b + k * (score_b - expected_b)
    
    def _update_arena_statistics(self, arena_data: Dict, strategy_a: str,
                                strategy_b: str, winner: str):
        """Update arena statistics."""
        stats = arena_data.get('statistics', {})
        
        # Initialize stats for strategies if needed
        for strategy in [strategy_a, strategy_b]:
            if strategy not in stats:
                stats[strategy] = {
                    'battles': 0,
                    'wins': 0,
                    'losses': 0,
                    'ties': 0,
                    'opponents': defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})
                }
        
        # Update battle counts
        stats[strategy_a]['battles'] += 1
        stats[strategy_b]['battles'] += 1
        
        # Update win/loss/tie counts
        if winner == strategy_a:
            stats[strategy_a]['wins'] += 1
            stats[strategy_b]['losses'] += 1
            stats[strategy_a]['opponents'][strategy_b]['wins'] += 1
            stats[strategy_b]['opponents'][strategy_a]['losses'] += 1
        elif winner == strategy_b:
            stats[strategy_b]['wins'] += 1
            stats[strategy_a]['losses'] += 1
            stats[strategy_b]['opponents'][strategy_a]['wins'] += 1
            stats[strategy_a]['opponents'][strategy_b]['losses'] += 1
        else:  # tie
            stats[strategy_a]['ties'] += 1
            stats[strategy_b]['ties'] += 1
            stats[strategy_a]['opponents'][strategy_b]['ties'] += 1
            stats[strategy_b]['opponents'][strategy_a]['ties'] += 1
        
        arena_data['statistics'] = stats
    
    def get_prompt_stats(self) -> Dict[str, Dict]:
        """Get statistics for all prompt templates."""
        with self.lock:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            return data.get('prompt_stats', {})
    
    def get_elo_ratings(self) -> Dict[str, float]:
        """Get current ELO ratings."""
        with self.lock:
            with open(self.arena_path, 'r') as f:
                arena_data = json.load(f)
            return arena_data.get('elo_ratings', {})
    
    def get_arena_statistics(self) -> Dict[str, Dict]:
        """Get detailed arena statistics."""
        with self.lock:
            with open(self.arena_path, 'r') as f:
                arena_data = json.load(f)
            
            stats = arena_data.get('statistics', {})
            
            # Calculate additional metrics
            for strategy, data in stats.items():
                if data['battles'] > 0:
                    data['win_rate'] = data['wins'] / data['battles'] * 100
                    data['loss_rate'] = data['losses'] / data['battles'] * 100
                    data['tie_rate'] = data['ties'] / data['battles'] * 100
                else:
                    data['win_rate'] = 0
                    data['loss_rate'] = 0
                    data['tie_rate'] = 0
            
            return stats
    
    def get_top_prompts(self, n: int = 10) -> List[Dict]:
        """Get the top n performing prompt templates."""
        stats = self.get_prompt_stats()
        
        # Sort by average rating (descending) and then by count
        sorted_prompts = sorted(
            stats.items(),
            key=lambda x: (x[1]['average'], x[1]['count']),
            reverse=True
        )
        
        return [
            {
                'prompt_id': prompt_id,
                'template': data['template'][:100] + '...' if len(data['template']) > 100 else data['template'],
                'average_rating': round(data['average'], 2),
                'total_ratings': data['count'],
                'std_dev': round(data.get('std_dev', 0), 2),
                'rating_range': f"{data.get('min', 1)}-{data.get('max', 5)}"
            }
            for prompt_id, data in sorted_prompts[:n]
        ]
    
    def get_recent_ratings(self, n: int = 20) -> List[Dict]:
        """Get the most recent n ratings."""
        with self.lock:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            return data['ratings'][-n:][::-1]  # Most recent first
    
    def get_recent_battles(self, n: int = 20) -> List[Dict]:
        """Get the most recent n arena battles."""
        with self.lock:
            with open(self.arena_path, 'r') as f:
                arena_data = json.load(f)
            return arena_data['battles'][-n:][::-1]  # Most recent first
    
    def get_head_to_head(self, strategy_a: str, strategy_b: str) -> Dict:
        """Get head-to-head statistics between two strategies."""
        stats = self.get_arena_statistics()
        
        if strategy_a not in stats or strategy_b not in stats:
            return {
                'strategy_a': strategy_a,
                'strategy_b': strategy_b,
                'battles': 0,
                'a_wins': 0,
                'b_wins': 0,
                'ties': 0
            }
        
        # Get head-to-head record
        a_vs_b = stats[strategy_a]['opponents'].get(strategy_b, {'wins': 0, 'losses': 0, 'ties': 0})
        
        return {
            'strategy_a': strategy_a,
            'strategy_b': strategy_b,
            'battles': a_vs_b['wins'] + a_vs_b['losses'] + a_vs_b['ties'],
            'a_wins': a_vs_b['wins'],
            'b_wins': a_vs_b['losses'],
            'ties': a_vs_b['ties']
        }
    
    def export_data(self, output_dir: str = 'exports'):
        """Export all data to CSV files."""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export ratings
        with self.lock:
            with open(self.db_path, 'r') as f:
                ratings_data = json.load(f)
            
            with open(self.arena_path, 'r') as f:
                arena_data = json.load(f)
        
        # Convert to CSV
        import csv
        
        # Export ratings
        ratings_file = Path(output_dir) / f'ratings_{timestamp}.csv'
        with open(ratings_file, 'w', newline='', encoding='utf-8') as f:
            if ratings_data['ratings']:
                writer = csv.DictWriter(f, fieldnames=ratings_data['ratings'][0].keys())
                writer.writeheader()
                writer.writerows(ratings_data['ratings'])
        
        # Export battles
        battles_file = Path(output_dir) / f'battles_{timestamp}.csv'
        with open(battles_file, 'w', newline='', encoding='utf-8') as f:
            if arena_data['battles']:
                writer = csv.DictWriter(f, fieldnames=arena_data['battles'][0].keys())
                writer.writeheader()
                writer.writerows(arena_data['battles'])
        
        # Export ELO ratings
        elo_file = Path(output_dir) / f'elo_ratings_{timestamp}.csv'
        with open(elo_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Strategy', 'ELO Rating'])
            for strategy, rating in sorted(arena_data['elo_ratings'].items(), 
                                         key=lambda x: x[1], reverse=True):
                writer.writerow([strategy, round(rating, 2)])
        
        return {
            'ratings': str(ratings_file),
            'battles': str(battles_file),
            'elo': str(elo_file)
        }
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for the entire system."""
        with self.lock:
            with open(self.db_path, 'r') as f:
                ratings_data = json.load(f)
            
            with open(self.arena_path, 'r') as f:
                arena_data = json.load(f)
        
        total_ratings = len(ratings_data['ratings'])
        total_battles = len(arena_data['battles'])
        
        # Calculate overall statistics
        all_ratings = [r['rating'] for r in ratings_data['ratings']]
        
        summary = {
            'total_ratings': total_ratings,
            'total_battles': total_battles,
            'average_rating': round(statistics.mean(all_ratings), 2) if all_ratings else 0,
            'rating_std_dev': round(statistics.stdev(all_ratings), 2) if len(all_ratings) > 1 else 0,
            'unique_prompts_rated': len(ratings_data['prompt_stats']),
            'strategies_in_arena': len(arena_data['elo_ratings']),
            'highest_elo': max(arena_data['elo_ratings'].items(), key=lambda x: x[1]) if arena_data['elo_ratings'] else ('N/A', 0),
            'lowest_elo': min(arena_data['elo_ratings'].items(), key=lambda x: x[1]) if arena_data['elo_ratings'] else ('N/A', 0),
            'database_created': ratings_data['metadata']['created_at'],
            'last_activity': max(
                ratings_data['ratings'][-1]['timestamp'] if ratings_data['ratings'] else '1970-01-01',
                arena_data['battles'][-1]['timestamp'] if arena_data['battles'] else '1970-01-01'
            )
        }
        
        return summary
