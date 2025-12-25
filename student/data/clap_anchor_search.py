"""
CLAP-based anchor search to find similar songs.
Used to extract context (genres, moods, instruments) for text generation.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class CLAPAnchorSearch:
    """CLAP-based similar song search for context extraction."""
    
    def __init__(self, config: Dict):
        """
        Initialize CLAP anchor search.
        
        Args:
            config: Configuration dictionary with keys:
                    top_k, similarity_threshold
        """
        self.top_k = config.get('top_k', 10)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # Cache for embeddings
        self.embeddings_cache = {}
        
        logger.info(f"Initialized CLAP anchor search with top_k={self.top_k}")
    
    def set_embeddings(self, embeddings_dict: Dict[str, np.ndarray]):
        """
        Set the embeddings cache for search.
        
        Args:
            embeddings_dict: Dictionary mapping item_id to CLAP embedding
        """
        self.embeddings_cache = embeddings_dict
        logger.info(f"Set {len(self.embeddings_cache)} embeddings for anchor search")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (0 to 1, higher is more similar)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        # Cosine similarity (1 - cosine distance)
        similarity = 1.0 - cosine(embedding1_normalized, embedding2_normalized)
        return float(similarity)
    
    def find_similar_songs(self, query_embedding: np.ndarray, 
                          song_metadata: Dict[str, Dict],
                          exclude_item_id: str = None) -> List[Tuple[str, float, Dict]]:
        """
        Find similar songs based on CLAP embedding.
        
        Args:
            query_embedding: CLAP embedding to search for
            song_metadata: Dictionary mapping item_id to song metadata
            exclude_item_id: Item ID to exclude from results (e.g., the query song itself)
            
        Returns:
            List of tuples (item_id, similarity_score, metadata) sorted by similarity
        """
        if not self.embeddings_cache:
            logger.warning("No embeddings loaded for anchor search")
            return []
        
        similarities = []
        
        for item_id, embedding in self.embeddings_cache.items():
            # Skip excluded item
            if exclude_item_id and item_id == exclude_item_id:
                continue
            
            # Compute similarity
            similarity = self.compute_similarity(query_embedding, embedding)
            
            # Filter by threshold
            if similarity >= self.similarity_threshold:
                metadata = song_metadata.get(item_id, {})
                similarities.append((item_id, similarity, metadata))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return similarities[:self.top_k]
    
    def extract_context_from_similar(self, similar_songs: List[Tuple[str, float, Dict]]) -> Dict:
        """
        Extract common context from similar songs for text generation.
        
        Args:
            similar_songs: List of (item_id, similarity, metadata) tuples
            
        Returns:
            Dictionary with extracted context:
                - common_moods: List of frequently occurring moods
                - mood_keywords: Aggregated mood keywords
                - tempo_range: (min, max) tempo
                - energy_range: (min, max) energy
        """
        if not similar_songs:
            return {
                'common_moods': [],
                'mood_keywords': [],
                'tempo_range': (None, None),
                'energy_range': (None, None)
            }
        
        # Aggregate moods
        mood_counts = {}
        tempos = []
        energies = []
        
        for item_id, similarity, metadata in similar_songs:
            # Parse mood vector
            mood_vector_str = metadata.get('mood_vector', '')
            if mood_vector_str:
                try:
                    # Mood vector is typically stored as comma-separated values or JSON
                    # Handle different formats
                    if isinstance(mood_vector_str, str):
                        mood_vector_str = mood_vector_str.strip()
                        # Try to parse as list
                        if mood_vector_str.startswith('['):
                            import json
                            mood_data = json.loads(mood_vector_str)
                        else:
                            mood_data = mood_vector_str.split(',')
                    else:
                        mood_data = mood_vector_str
                    
                    # Count mood occurrences
                    for mood in mood_data:
                        if isinstance(mood, str):
                            mood = mood.strip()
                            if mood:
                                mood_counts[mood] = mood_counts.get(mood, 0) + 1
                except Exception as e:
                    logger.debug(f"Could not parse mood vector for {item_id}: {e}")
            
            # Collect tempo and energy
            if metadata.get('tempo'):
                try:
                    tempos.append(float(metadata['tempo']))
                except:
                    pass
            
            if metadata.get('energy'):
                try:
                    energies.append(float(metadata['energy']))
                except:
                    pass
        
        # Get most common moods
        common_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)
        common_moods = [mood for mood, count in common_moods[:10]]  # Top 10 moods
        
        # Calculate ranges
        tempo_range = (min(tempos), max(tempos)) if tempos else (None, None)
        energy_range = (min(energies), max(energies)) if energies else (None, None)
        
        context = {
            'common_moods': common_moods,
            'mood_keywords': common_moods,  # Same as common_moods for now
            'tempo_range': tempo_range,
            'energy_range': energy_range
        }
        
        logger.debug(f"Extracted context: {len(common_moods)} moods, "
                    f"tempo {tempo_range}, energy {energy_range}")
        
        return context
