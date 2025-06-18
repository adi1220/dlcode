"""
Memory Tool for TSV-based memory storage and retrieval
"""

import csv
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryTool:
    """A tool for managing memories stored in a TSV database."""
    
    def __init__(self, db_path: str = "memory_db.tsv"):
        """
        Initialize the MemoryTool with a database file path.
        
        Args:
            db_path: Path to the TSV database file
        """
        self.db_path = Path(db_path)
        self._ensure_database_exists()
    
    def _ensure_database_exists(self) -> None:
        """Ensure the database file exists with correct headers."""
        if not self.db_path.exists():
            logger.info(f"Creating new database at {self.db_path}")
            with open(self.db_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['ID', 'Timestamp', 'Content'])
        else:
            # Verify headers
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    headers = next(reader, None)
                    if headers != ['ID', 'Timestamp', 'Content']:
                        logger.warning("Invalid headers detected, recreating database")
                        self._create_backup()
                        with open(self.db_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f, delimiter='\t')
                            writer.writerow(['ID', 'Timestamp', 'Content'])
            except Exception as e:
                logger.error(f"Error verifying database: {e}")
                self._create_backup()
                with open(self.db_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['ID', 'Timestamp', 'Content'])
    
    def _create_backup(self) -> None:
        """Create a backup of the existing database."""
        if self.db_path.exists():
            backup_path = self.db_path.with_suffix('.tsv.backup')
            try:
                self.db_path.rename(backup_path)
                logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
    
    def _generate_id(self) -> str:
        """Generate a unique ID for a memory entry."""
        # Use timestamp with high precision combined with a short UUID
        timestamp = str(time.time()).replace('.', '')
        unique_suffix = str(uuid.uuid4()).split('-')[0]
        return f"{timestamp}_{unique_suffix}"
    
    def _load_memories(self) -> List[Dict[str, str]]:
        """Load all memories from the TSV file."""
        memories = []
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    memories.append(row)
        except IOError as e:
            logger.error(f"Error loading memories: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading memories: {e}")
            return []
        return memories
    
    def _save_memories(self, memories: List[Dict[str, str]]) -> None:
        """Save memories back to the TSV file."""
        try:
            with open(self.db_path, 'w', newline='', encoding='utf-8') as f:
                if memories:
                    writer = csv.DictWriter(f, fieldnames=['ID', 'Timestamp', 'Content'], delimiter='\t')
                    writer.writeheader()
                    writer.writerows(memories)
                else:
                    # Just write headers for empty database
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['ID', 'Timestamp', 'Content'])
        except IOError as e:
            logger.error(f"Error saving memories: {e}")
            raise
    
    def add_memory(self, content: str) -> str:
        """
        Add a new memory to the database.
        
        Args:
            content: The content to store as a memory
            
        Returns:
            Confirmation message with the assigned ID
        """
        memory_id = self._generate_id()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # Append to file
            with open(self.db_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow([memory_id, timestamp, content])
            
            logger.info(f"Added memory with ID: {memory_id}")
            return f"Memory stored successfully with ID: {memory_id}"
        
        except IOError as e:
            logger.error(f"Failed to add memory: {e}")
            return f"Error storing memory: {e}"
    
    def fetch_memories(self, query: str = "", limit: int = 5) -> List[Dict[str, str]]:
        """
        Fetch memories based on a query or get recent memories.
        
        Args:
            query: Search query (empty for recent memories)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries
        """
        memories = self._load_memories()
        
        if not memories:
            return []
        
        # Check if query is asking for recent memories
        if not query or query.lower() in ['recent', 'latest', 'last']:
            # Sort by timestamp (assuming ISO format allows string sorting)
            memories.sort(key=lambda x: x['Timestamp'], reverse=True)
            return memories[:limit]
        
        # Perform case-insensitive search
        matched_memories = []
        query_lower = query.lower()
        
        for memory in memories:
            if query_lower in memory['Content'].lower():
                matched_memories.append(memory)
        
        # Sort matched memories by timestamp (most recent first)
        matched_memories.sort(key=lambda x: x['Timestamp'], reverse=True)
        
        return matched_memories[:limit]
    
    def delete_memory(self, memory_id: str) -> str:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            Success or error message
        """
        memories = self._load_memories()
        
        # Find and remove the memory
        original_count = len(memories)
        memories = [m for m in memories if m['ID'] != memory_id]
        
        if len(memories) == original_count:
            return f"Memory not found with ID: {memory_id}"
        
        try:
            self._save_memories(memories)
            logger.info(f"Deleted memory with ID: {memory_id}")
            return f"Memory with ID {memory_id} has been deleted successfully"
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return f"Error deleting memory: {e}"
    
    def get_memory_count(self) -> int:
        """Get the total number of memories in the database."""
        return len(self._load_memories())
    
    def clear_all_memories(self) -> str:
        """Clear all memories from the database (use with caution)."""
        try:
            self._save_memories([])
            return "All memories have been cleared"
        except Exception as e:
            return f"Error clearing memories: {e}"


# Example usage
if __name__ == "__main__":
    # Create memory tool instance
    memory_tool = MemoryTool("test_memory_db.tsv")
    
    # Add some test memories
    print(memory_tool.add_memory("My favorite programming language is Python"))
    print(memory_tool.add_memory("I enjoy hiking on weekends"))
    print(memory_tool.add_memory("The meeting is scheduled for tomorrow at 3 PM"))
    
    # Fetch recent memories
    print("\n--- Recent Memories ---")
    recent = memory_tool.fetch_memories(limit=2)
    for mem in recent:
        print(f"[{mem['ID']}] {mem['Timestamp']}: {mem['Content']}")
    
    # Search for specific memory
    print("\n--- Search Results for 'programming' ---")
    results = memory_tool.fetch_memories("programming")
    for mem in results:
        print(f"[{mem['ID']}] {mem['Content']}")
    
    # Delete a memory (if results exist)
    if results:
        memory_id = results[0]['ID']
        print(f"\n--- Deleting memory {memory_id} ---")
        print(memory_tool.delete_memory(memory_id))
    
    print(f"\nTotal memories: {memory_tool.get_memory_count()}")
