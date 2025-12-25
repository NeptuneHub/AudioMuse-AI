"""
OpenAI text generation for creating diverse music descriptions.
Generates multiple descriptions per song using metadata and CLAP anchor context.
"""

import logging
import json
import os
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class TextGenerator:
    """OpenAI-based text generator for music descriptions."""
    
    def __init__(self, config: Dict):
        """
        Initialize text generator.
        
        Args:
            config: OpenAI configuration dictionary with keys:
                    api_key, model, temperature, max_tokens
        """
        self.api_key = config['api_key']
        self.model = config['model']
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 200)
        self.client = OpenAI(api_key=self.api_key)
        
        self.system_prompt = config.get('system_prompt', 
            "You are a music expert who creates concise, diverse descriptions of songs. "
            "Focus on different aspects: mood, genre, instrumentation, tempo, energy, and style.")
        
        logger.info(f"Initialized OpenAI text generator with model {self.model}")
    
    def generate_descriptions(self, 
                            song_metadata: Dict,
                            anchor_context: Dict,
                            num_descriptions: int = 5) -> List[str]:
        """
        Generate diverse text descriptions for a song.
        
        Args:
            song_metadata: Dictionary with song information:
                          title, author, tempo, energy, mood_vector, other_features
            anchor_context: Context from CLAP anchor search:
                          common_moods, tempo_range, energy_range
            num_descriptions: Number of descriptions to generate
            
        Returns:
            List of generated text descriptions
        """
        # Build user prompt
        user_prompt = self._build_prompt(song_metadata, anchor_context, num_descriptions)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            descriptions = self._parse_descriptions(content, num_descriptions)
            
            logger.info(f"Generated {len(descriptions)} descriptions for {song_metadata.get('title', 'unknown')}")
            return descriptions
            
        except Exception as e:
            logger.error(f"Failed to generate descriptions: {e}")
            # Return fallback descriptions
            return self._generate_fallback_descriptions(song_metadata, num_descriptions)
    
    def _build_prompt(self, song_metadata: Dict, anchor_context: Dict, num_descriptions: int) -> str:
        """Build the user prompt for OpenAI."""
        
        # Extract metadata
        title = song_metadata.get('title', 'Unknown')
        author = song_metadata.get('author', 'Unknown')
        tempo = song_metadata.get('tempo')
        energy = song_metadata.get('energy')
        mood_vector = song_metadata.get('mood_vector', '')
        other_features = song_metadata.get('other_features', '')
        
        # Extract anchor context
        common_moods = anchor_context.get('common_moods', [])
        tempo_range = anchor_context.get('tempo_range', (None, None))
        energy_range = anchor_context.get('energy_range', (None, None))
        
        # Build prompt
        prompt_parts = [
            f"Generate {num_descriptions} diverse, concise descriptions (10-20 words each) for this song:",
            f"\nTitle: {title}",
            f"Artist: {author}"
        ]
        
        if tempo:
            prompt_parts.append(f"Tempo: {tempo} BPM")
        
        if energy:
            prompt_parts.append(f"Energy: {energy:.2f}")
        
        if mood_vector:
            prompt_parts.append(f"Mood vector: {mood_vector}")
        
        if other_features:
            prompt_parts.append(f"Other features: {other_features}")
        
        if common_moods:
            prompt_parts.append(f"\nSimilar songs often have these moods: {', '.join(common_moods[:5])}")
        
        if tempo_range[0] and tempo_range[1]:
            prompt_parts.append(f"Similar songs have tempo range: {tempo_range[0]:.0f}-{tempo_range[1]:.0f} BPM")
        
        if energy_range[0] and energy_range[1]:
            prompt_parts.append(f"Similar songs have energy range: {energy_range[0]:.2f}-{energy_range[1]:.2f}")
        
        prompt_parts.extend([
            f"\nGenerate {num_descriptions} different descriptions focusing on:",
            "1. Overall mood and emotion",
            "2. Genre and style",
            "3. Instrumentation and sound",
            "4. Tempo and energy characteristics",
            "5. Unique qualities or atmosphere",
            "\nProvide one description per line, without numbering."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_descriptions(self, content: str, expected_count: int) -> List[str]:
        """Parse descriptions from API response."""
        # Split by newlines
        lines = content.strip().split('\n')
        
        descriptions = []
        for line in lines:
            line = line.strip()
            # Remove numbering if present (1., 2., etc.)
            if line and len(line) > 3:
                # Remove leading numbers and punctuation
                if line[0].isdigit():
                    line = line.split('.', 1)[-1].strip()
                if line[0] == '-':
                    line = line[1:].strip()
                
                if line:
                    descriptions.append(line)
        
        # If we got fewer descriptions than expected, pad with variations
        while len(descriptions) < expected_count and descriptions:
            descriptions.append(descriptions[0])  # Repeat first description
        
        return descriptions[:expected_count]
    
    def _generate_fallback_descriptions(self, song_metadata: Dict, num_descriptions: int) -> List[str]:
        """Generate fallback descriptions when API fails."""
        title = song_metadata.get('title', 'Unknown')
        author = song_metadata.get('author', 'Unknown')
        tempo = song_metadata.get('tempo')
        energy = song_metadata.get('energy')
        
        descriptions = [
            f"A song by {author}",
            f"Music track titled {title}",
        ]
        
        if tempo:
            if tempo < 90:
                descriptions.append("Slow tempo ballad with emotional depth")
            elif tempo < 120:
                descriptions.append("Mid-tempo track with steady rhythm")
            else:
                descriptions.append("Fast-paced energetic music")
        
        if energy:
            if energy < 0.3:
                descriptions.append("Calm and relaxing musical piece")
            elif energy < 0.7:
                descriptions.append("Moderately energetic song")
            else:
                descriptions.append("High energy dynamic track")
        
        descriptions.append(f"Musical composition by {author}")
        
        # Pad to requested count
        while len(descriptions) < num_descriptions:
            descriptions.append(descriptions[0])
        
        return descriptions[:num_descriptions]
    
    def batch_generate(self, 
                      songs_metadata: List[Dict],
                      anchor_contexts: List[Dict],
                      num_descriptions: int = 5,
                      cache_file: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Generate descriptions for multiple songs with caching.
        
        Args:
            songs_metadata: List of song metadata dictionaries
            anchor_contexts: List of anchor context dictionaries (same order as songs)
            num_descriptions: Number of descriptions per song
            cache_file: Optional path to cache file for storing/loading descriptions
            
        Returns:
            Dictionary mapping item_id to list of descriptions
        """
        # Load from cache if available
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_descriptions = json.load(f)
                logger.info(f"Loaded {len(cached_descriptions)} descriptions from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                cached_descriptions = {}
        else:
            cached_descriptions = {}
        
        all_descriptions = {}
        
        for i, (song_metadata, anchor_context) in enumerate(zip(songs_metadata, anchor_contexts)):
            item_id = song_metadata.get('item_id')
            
            if not item_id:
                logger.warning(f"Song metadata missing item_id at index {i}")
                continue
            
            # Check cache first
            if item_id in cached_descriptions:
                all_descriptions[item_id] = cached_descriptions[item_id]
                logger.debug(f"Using cached descriptions for {item_id}")
                continue
            
            # Generate new descriptions
            logger.info(f"Generating descriptions {i+1}/{len(songs_metadata)}: {item_id}")
            descriptions = self.generate_descriptions(song_metadata, anchor_context, num_descriptions)
            all_descriptions[item_id] = descriptions
            
            # Update cache
            if cache_file:
                cached_descriptions[item_id] = descriptions
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(cached_descriptions, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")
        
        logger.info(f"Generated/loaded descriptions for {len(all_descriptions)} songs")
        return all_descriptions
