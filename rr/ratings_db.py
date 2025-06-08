import json
import os
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional
import statistics

class RatingsDatabase:
    def __init__(self, db_path: str = 'ratings.json'):
        self.db_path = db_path
        self.lock = Lock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database file if it doesn't exist."""
        if not os.path.exists(self.db_path):
            with self.lock:
                with open(self.db_path, 'w') as f:
                    json.dump({
                        'ratings': [],
                        'prompt_stats': {}
                    }, f, indent=2)
    
    def add_rating(self, prompt_id: str, prompt_template: str, 
                   user_text: str, bix_response: str, 
                   rewritten_response: str, rating: int) -> Dict:
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
                'rating': rating
            }
            
            # Add to ratings list
            data['ratings'].append(rating_entry)
            
            # Update prompt statistics
            if prompt_id not in data['prompt_stats']:
                data['prompt_stats'][prompt_id] = {
                    'template': prompt_template,
                    'ratings': [],
                    'count': 0,
                    'average': 0.0
                }
            
            data['prompt_stats'][prompt_id]['ratings'].append(rating)
            data['prompt_stats'][prompt_id]['count'] += 1
            data['prompt_stats'][prompt_id]['average'] = statistics.mean(
                data['prompt_stats'][prompt_id]['ratings']
            )
            
            # Save updated data
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return rating_entry
    
    def get_prompt_stats(self) -> Dict[str, Dict]:
        """Get statistics for all prompt templates."""
        with self.lock:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            return data.get('prompt_stats', {})
    
    def get_top_prompts(self, n: int = 5) -> List[Dict]:
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
                'total_ratings': data['count']
            }
            for prompt_id, data in sorted_prompts[:n]
        ]
    
    def get_recent_ratings(self, n: int = 10) -> List[Dict]:
        """Get the most recent n ratings."""
        with self.lock:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            return data['ratings'][-n:][::-1]  # Most recent first
