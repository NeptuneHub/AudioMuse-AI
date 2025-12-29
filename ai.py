import requests
import json
import re
import ftfy # Import the ftfy library
import time # Import the time library
import logging
import unicodedata
import google.generativeai as genai # Import Gemini library
from mistralai import Mistral
import os # Import os to potentially read GEMINI_API_CALL_DELAY_SECONDS
from config import MAX_SONGS_IN_AI_PROMPT

logger = logging.getLogger(__name__)

# creative_prompt_template is imported in tasks.py, so it should be defined here
creative_prompt_template = (
    "You're an expert of music and you need to give a title to this playlist.\n"
    "The title need to represent the mood and the activity of when you listening the playlist.\n"
    "The title MUST use ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "No special fonts or emojis.\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '𝑯𝒘𝒆 𝒂𝒓𝒐𝒏𝒊 𝒅𝒆𝒕𝒔' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n"
    "This is the playlist: {song_list_sample}\n\n" # {song_list_sample} will contain the full list

)

def clean_playlist_name(name):
    if not isinstance(name, str):
        return ""
    # print(f"DEBUG CLEAN AI: Input name: '{name}'") # Print name before cleaning

    name = ftfy.fix_text(name)

    name = unicodedata.normalize('NFKC', name)
    # Stricter regex: only allows characters explicitly mentioned in the prompt.
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s\-\&\'!\.\,\?\(\)\[\]]', '', name)
    # Also remove trailing number in parentheses, e.g., "My Playlist (2)" -> "My Playlist", to prevent AI from interfering with disambiguation logic.
    cleaned_name = re.sub(r'\s\(\d+\)$', '', cleaned_name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    return cleaned_name


# --- OpenAI-Compatible API Function (used for both Ollama and OpenAI/OpenRouter) ---
def get_openai_compatible_playlist_name(server_url, model_name, full_prompt, api_key="no-key-needed", skip_delay=False):
    """
    Calls an OpenAI-compatible API endpoint to get a playlist name.
    This works for Ollama (no API key needed) and OpenAI/OpenRouter (API key required).
    This version handles streaming responses and extracts only the non-think part.

    Args:
        server_url (str): The URL of the API endpoint (e.g., "http://192.168.3.15:11434/api/generate" for Ollama,
                         or "https://openrouter.ai/api/v1/chat/completions" for OpenRouter).
        model_name (str): The model to use (e.g., "deepseek-r1:1.5b" for Ollama, "openai/gpt-4" for OpenRouter).
        full_prompt (str): The complete prompt text to send to the model.
        api_key (str): API key for authentication. Use "no-key-needed" for Ollama.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Detect which format to use based on API key and URL
    is_openai_format = api_key != "no-key-needed" or "openai" in server_url.lower() or "openrouter" in server_url.lower()

    headers = {
        "Content-Type": "application/json"
    }

    # Add Authorization header if API key is provided and not the default "no-key-needed"
    if api_key and api_key != "no-key-needed":
        headers["Authorization"] = f"Bearer {api_key}"

    # Add OpenRouter-specific headers if using OpenRouter
    if "openrouter" in server_url.lower():
        headers["HTTP-Referer"] = "https://github.com/NeptuneHub/AudioMuse-AI"
        headers["X-Title"] = "AudioMuse-AI"

    # Prepare payload based on format
    if is_openai_format:
        # OpenAI/OpenRouter format uses chat completions
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 8000
        }
    else:
        # Ollama format uses generate endpoint
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "num_predict": 8000,
                "temperature": 0.7
            }
        }

    max_retries = 3
    base_delay = 5
    tried_max_tokens_fallback = False
    tried_aggressive_fallback = False  # NEW: Track aggressive parameter fallback state
    tried_ultra_minimal_fallback = False  # NEW: Track ultra-minimal fallback state

    for attempt in range(max_retries + 1):
        try:
            # Add delay for OpenAI/OpenRouter to respect rate limits (only on first attempt or if not 429 retry)
            if is_openai_format and attempt == 0:
                openai_call_delay = int(os.environ.get("OPENAI_API_CALL_DELAY_SECONDS", "7"))
                if openai_call_delay > 0:
                    logger.debug("Waiting for %ss before OpenAI/OpenRouter API call to respect rate limits.", openai_call_delay)
                    time.sleep(openai_call_delay)

            logger.debug("Starting API call for model '%s' at '%s' (format: %s). Attempt %d/%d", model_name, server_url, "OpenAI" if is_openai_format else "Ollama", attempt + 1, max_retries + 1)

            response = requests.post(server_url, headers=headers, data=json.dumps(payload), stream=True, timeout=960)
            response.raise_for_status()
            full_raw_response_content = ""

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8', errors='ignore').strip()

                # Skip SSE comments (lines starting with :)
                if line_str.startswith(':'):
                    continue

                # Handle SSE data format (OpenRouter/OpenAI)
                if line_str.startswith('data: '):
                    line_str = line_str[6:]  # Remove 'data: ' prefix

                    # Check for end of stream marker
                    if line_str == '[DONE]':
                        break

                # Try to parse JSON
                try:
                    chunk = json.loads(line_str)

                    # Extract content based on format
                    if is_openai_format:
                        # OpenAI/OpenRouter format
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]

                            # Check for finish
                            finish_reason = choice.get('finish_reason')
                            if finish_reason == 'stop':
                                break
                            elif finish_reason == 'length':
                                logger.warning("Response truncated due to max_tokens limit")
                                break

                            # Extract text from delta.content or text field
                            if 'delta' in choice:
                                content = choice['delta'].get('content')
                                if content is not None:
                                    full_raw_response_content += content
                            elif 'text' in choice:
                                text = choice.get('text')
                                if text is not None:
                                    full_raw_response_content += text
                    else:
                        # Ollama format
                        if 'response' in chunk:
                            full_raw_response_content += chunk['response']
                        if chunk.get('done'):
                            break

                except json.JSONDecodeError:
                    logger.debug("Could not decode JSON line from stream: %s", line_str)
                    continue

            # Extract text after common thought tags
            thought_enders = ["</think>", "[/INST]", "[/THOUGHT]"]
            extracted_text = full_raw_response_content.strip()
            for end_tag in thought_enders:
                 if end_tag in extracted_text:
                     extracted_text = extracted_text.split(end_tag, 1)[-1].strip()

            # Log the raw response for debugging (consistent with Gemini/Mistral)
            if extracted_text:
                logger.info("OpenAI/OpenRouter API returned: '%s'", extracted_text)
                return extracted_text
            else:
                logger.warning("OpenAI/OpenRouter returned empty content. Full raw response: %s", full_raw_response_content)
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info("Retrying in %s seconds due to empty content...", sleep_time)
                    time.sleep(sleep_time)
                    continue
                else:
                    return "Error: AI returned empty content after retries."

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded (429). Attempt %d/%d", attempt + 1, max_retries + 1)
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info("Retrying in %s seconds...", sleep_time)
                    time.sleep(sleep_time)
                    continue
            
            # === NEW: Parameter error handling (ADD) ===
            if e.response.status_code == 400 and is_openai_format:
                try:
                    error_body = e.response.json()
                    error_msg = error_body.get('error', {}).get('message', '').lower()
                    
                    # Aggressive fallback: first parameter error
                    if 'unsupported' in error_msg and not tried_aggressive_fallback:
                        logger.info("Unsupported parameter detected, switching to max_completion_tokens and removing temperature")
                        payload.pop('temperature', None)
                        payload.pop('max_tokens', None)
                        payload['max_completion_tokens'] = 8000
                        # Also remove from nested options (Ollama format)
                        if 'options' in payload and isinstance(payload['options'], dict):
                            payload['options'].pop('temperature', None)
                        tried_aggressive_fallback = True
                        continue  # Immediate retry, no delay, no attempt increment
                    
                    # Ultra-minimal fallback: still failing after aggressive
                    elif 'unsupported' in error_msg and tried_aggressive_fallback and not tried_ultra_minimal_fallback:
                        logger.info("Still failing with max_completion_tokens, removing it (ultra-minimal mode)")
                        payload.pop('max_completion_tokens', None)
                        tried_ultra_minimal_fallback = True
                        continue  # Immediate retry, no delay, no attempt increment
                except (json.JSONDecodeError, KeyError, AttributeError):
                    pass  # Can't parse error, fall through
            
            # Check if it's the max_tokens vs max_completion_tokens issue
            if e.response.status_code == 400 and is_openai_format and not tried_max_tokens_fallback:
                try:
                    error_body = e.response.json()
                    error_msg = error_body.get('error', {}).get('message', '')
                    
                    # If error is about max_tokens not supported, try max_completion_tokens
                    if 'max_tokens' in error_msg.lower() and 'not supported' in error_msg.lower():
                        logger.info("Model doesn't support max_tokens, switching to max_completion_tokens")
                        payload.pop('max_tokens', None)
                        payload['max_completion_tokens'] = 8000
                        tried_max_tokens_fallback = True
                        continue
                except (json.JSONDecodeError, KeyError, AttributeError):
                    pass
            
            # Log the response body for better debugging
            error_detail = ""
            try:
                error_detail = e.response.text
                logger.error("Error calling OpenAI-compatible API: %s. Response body: %s", e, error_detail, exc_info=True)
            except:
                logger.error("Error calling OpenAI-compatible API: %s", e, exc_info=True)
            return "Error: AI service is currently unavailable."
            
        except requests.exceptions.RequestException as e:
            logger.error("Error calling OpenAI-compatible API: %s", e, exc_info=True)
            return "Error: AI service is currently unavailable."
        except Exception as e:
            logger.error("An unexpected error occurred in get_openai_compatible_playlist_name", exc_info=True)
            return "Error: AI service is currently unavailable."
    
    return "Error: Max retries exceeded."

# --- Ollama Specific Function (wrapper for backward compatibility) ---
def get_ollama_playlist_name(ollama_url, model_name, full_prompt, skip_delay=False):
    """
    Calls a self-hosted Ollama instance to get a playlist name.
    This is a wrapper around get_openai_compatible_playlist_name for backward compatibility.

    Args:
        ollama_url (str): The URL of your Ollama instance (e.g., "http://192.168.3.15:11434/api/generate").
        model_name (str): The Ollama model to use (e.g., "deepseek-r1:1.5b").
        full_prompt (str): The complete prompt text to send to the model.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    return get_openai_compatible_playlist_name(ollama_url, model_name, full_prompt, api_key="no-key-needed", skip_delay=skip_delay)

# --- Gemini Specific Function ---
def get_gemini_playlist_name(gemini_api_key, model_name, full_prompt, skip_delay=False):
    """
    Calls the Google Gemini API to get a playlist name.

    Args:
        gemini_api_key (str): Your Google Gemini API key.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
        model_name (str): The Gemini model to use (e.g., "gemini-2.5-pro").
        full_prompt (str): The complete prompt text to send to the model.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Allow any provided key, even if it's the placeholder, but check if it's empty/default
    if not gemini_api_key or gemini_api_key == "YOUR-GEMINI-API-KEY-HERE":
         return "Error: Gemini API key is missing or empty. Please provide a valid API key."
    
    try:
        # Read delay from environment/config if needed, otherwise use the default (skip for chat requests)
        if not skip_delay:
            gemini_call_delay = int(os.environ.get("GEMINI_API_CALL_DELAY_SECONDS", "7")) # type: ignore
            if gemini_call_delay > 0:
                logger.debug("Waiting for %ss before Gemini API call to respect rate limits.", gemini_call_delay)
                time.sleep(gemini_call_delay)

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)

        logger.debug("Starting API call for model '%s'.", model_name)
 
        generation_config = genai.types.GenerationConfig(
            temperature=0.9 # Explicitly set temperature for more creative/varied responses
        )
        response = model.generate_content(full_prompt, generation_config=generation_config, request_options={'timeout': 960})
        # Extract text from the response # type: ignore
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            extracted_text = "".join(part.text for part in response.candidates[0].content.parts)
            # Log the raw response for debugging (consistent with OpenAI/OpenRouter)
            logger.info("Gemini API returned: '%s'", extracted_text)
        else:
            logger.warning("Gemini returned no content. Raw response: %s", response)
            return "Error: Gemini returned no content."

        # The final cleaning and length check is done in the general function
        return extracted_text

    except Exception as e:
        logger.error("Error calling Gemini API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."

# --- Mistral Specific Function ---
def get_mistral_playlist_name(mistral_api_key, model_name, full_prompt, skip_delay=False):
    """
    Calls the Mistral API to get a playlist name.

    Args:
        mistral_api_key (str): Your Mistral API key.
        model_name (str): The mistral model to use (e.g., "ministral-3b-latest").
        full_prompt (str): The complete prompt text to send to the model.
        skip_delay (bool): If True, skip the rate limit delay (used for chat requests).
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Allow any provided key, even if it's the placeholder, but check if it's empty/default
    if not mistral_api_key or mistral_api_key == "YOUR-MISTRAL-API-KEY-HERE":
         return "Error: Mistral API key is missing or empty. Please provide a valid API key."

    try:
        # Read delay from environment/config if needed, otherwise use the default (skip for chat requests)
        if not skip_delay:
            mistral_call_delay = int(os.environ.get("MISTRAL_API_CALL_DELAY_SECONDS", "7")) # type: ignore
            if mistral_call_delay > 0:
                logger.debug("Waiting for %ss before mistral API call to respect rate limits.", mistral_call_delay)
                time.sleep(mistral_call_delay)

        client = Mistral(api_key=mistral_api_key)

        logger.debug("Starting API call for model '%s'.", model_name)

        response = client.chat.complete(model=model_name,
                                        temperature=0.9,
                                        timeout_ms=960,
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": full_prompt,
                                            },
                                        ])
        # Extract text from the response # type: ignore
        if response and response.choices[0].message.content:
            extracted_text = response.choices[0].message.content
            # Log the raw response for debugging (consistent with OpenAI/OpenRouter)
            logger.info("Mistral API returned: '%s'", extracted_text)
        else:
            logger.warning("Mistral returned no content. Raw response: %s", response)
            return "Error: mistral returned no content."

        # The final cleaning and length check is done in the general function
        return extracted_text

    except Exception as e:
        logger.error("Error calling Mistral API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."

# --- General AI Call Function (for chat multi-step) ---
def call_ai_for_chat(provider, prompt, ollama_url=None, ollama_model_name=None, 
                     gemini_api_key=None, gemini_model_name=None,
                     mistral_api_key=None, mistral_model_name=None,
                     openai_server_url=None, openai_model_name=None, openai_api_key=None):
    """
    Generic function to call any AI provider with a given prompt.
    Returns the raw text response from the AI.
    Skip delays for chat requests to improve response time.
    """
    if provider == "OLLAMA":
        if not ollama_url or not ollama_model_name:
            return "Error: Ollama configuration missing"
        return get_ollama_playlist_name(ollama_url, ollama_model_name, prompt, skip_delay=True)
    elif provider == "OPENAI":
        if not openai_server_url or not openai_model_name or not openai_api_key:
            return "Error: OpenAI configuration missing"
        return get_openai_compatible_playlist_name(openai_server_url, openai_model_name, prompt, openai_api_key, skip_delay=True)
    elif provider == "GEMINI":
        if not gemini_api_key or not gemini_model_name:
            return "Error: Gemini configuration missing"
        return get_gemini_playlist_name(gemini_api_key, gemini_model_name, prompt, skip_delay=True)
    elif provider == "MISTRAL":
        if not mistral_api_key or not mistral_model_name:
            return "Error: Mistral configuration missing"
        return get_mistral_playlist_name(mistral_api_key, mistral_model_name, prompt, skip_delay=True)
    else:
        return "Error: Invalid AI provider"

# --- General AI Naming Function ---
def get_ai_playlist_name(provider, ollama_url, ollama_model_name, gemini_api_key, gemini_model_name, mistral_api_key, mistral_model_name, prompt_template, feature1, feature2, feature3, song_list, other_feature_scores_dict, openai_server_url=None, openai_model_name=None, openai_api_key=None):
    """
    Selects and calls the appropriate AI model based on the provider.
    Constructs the full prompt including new features.
    Applies length constraints after getting the name.
    """
    MIN_LENGTH = 5
    MAX_LENGTH = 40

    # --- Prepare feature descriptions for the prompt ---
    tempo_description_for_ai = "Tempo is moderate." # Default
    energy_description = "" # Initialize energy description

    if other_feature_scores_dict:
        # Extract energy score first, as it's handled separately
        # Check for 'energy_normalized' first, then fall back to 'energy'
        energy_score = other_feature_scores_dict.get('energy_normalized', other_feature_scores_dict.get('energy', 0.0))

        # Create energy description based on score (example thresholds)
        if energy_score < 0.3:
            energy_description = " It has low energy."
        elif energy_score > 0.7:
            energy_description = " It has high energy."
        # No description if medium energy (between 0.3 and 0.7)

        # Create tempo description
        tempo_normalized_score = other_feature_scores_dict.get('tempo_normalized', 0.5) # Default to moderate if not found
        if tempo_normalized_score < 0.33:
            tempo_description_for_ai = "The tempo is generally slow."
        elif tempo_normalized_score < 0.66:
            tempo_description_for_ai = "The tempo is generally medium."
        else:
            tempo_description_for_ai = "The tempo is generally fast."

        # Note: The logic for 'new_features_description' (which was for 'additional_features_description')
        # has been removed as per the request. If you want to include other features
        # (like danceable, aggressive, etc.) in the prompt, you'd add logic here to create
        # a description for them and a corresponding placeholder in the prompt_template.

    # Format the song list for the prompt
    # Truncate to MAX_SONGS_IN_AI_PROMPT to avoid token limit issues with large playlists
    songs_for_prompt = song_list[:MAX_SONGS_IN_AI_PROMPT]
    formatted_song_list = "\n".join([f"- {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}" for song in songs_for_prompt])
    
    # Log if we truncated the list
    if len(song_list) > MAX_SONGS_IN_AI_PROMPT:
        logger.info("Truncated song list from %d to %d songs for AI prompt to avoid token limits", len(song_list), MAX_SONGS_IN_AI_PROMPT)

    # Construct the full prompt using the template and all features
    # The new prompt only requires the song list sample # type: ignore
    full_prompt = prompt_template.format(song_list_sample=formatted_song_list)

    logger.info("Sending prompt to AI (%s):\n%s", provider, full_prompt)

    # --- Call the AI Model with Retry Logic ---
    max_retries = 3
    current_prompt = full_prompt
    
    for attempt in range(max_retries):
        name = "AI Naming Skipped" # Default if provider is NONE or invalid

        if provider == "OLLAMA":
            name = get_ollama_playlist_name(ollama_url, ollama_model_name, current_prompt)
        elif provider == "OPENAI":
            # Use OpenAI-compatible API with API key
            if not openai_server_url or not openai_model_name or not openai_api_key:
                return "Error: OpenAI configuration is incomplete. Please provide server URL, model name, and API key."
            name = get_openai_compatible_playlist_name(openai_server_url, openai_model_name, current_prompt, openai_api_key)
        elif provider == "GEMINI":
            name = get_gemini_playlist_name(gemini_api_key, gemini_model_name, current_prompt)
        elif provider == "MISTRAL":
            name = get_mistral_playlist_name(mistral_api_key, mistral_model_name, current_prompt)
        # else: provider is NONE or invalid, name remains "AI Naming Skipped"

        # Apply length check and return final name or error
        # Only apply length check if a name was actually generated (not the skip message or an API error message)
        if name not in ["AI Naming Skipped"] and not name.startswith("Error"):
            cleaned_name = clean_playlist_name(name)
            if MIN_LENGTH <= len(cleaned_name) <= MAX_LENGTH:
                return cleaned_name
            else:
                # Name failed length check
                logger.warning(f"AI generated name '{cleaned_name}' ({len(cleaned_name)} chars) outside {MIN_LENGTH}-{MAX_LENGTH} range. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    # Prepare feedback for next attempt
                    feedback = f"\n\nFEEDBACK: The previous title you generated ('{cleaned_name}') was {len(cleaned_name)} characters long. It MUST be between {MIN_LENGTH} and {MAX_LENGTH} characters. Please try again."
                    current_prompt = full_prompt + feedback
                    continue # Try again
                else:
                    # Return an error message indicating the length issue, but include the cleaned name for debugging
                    return f"Error: AI generated name '{cleaned_name}' ({len(cleaned_name)} chars) outside {MIN_LENGTH}-{MAX_LENGTH} range after {max_retries} attempts."
        else:
            # API error or skipped
            return name
            
    return "Error: Max retries exceeded in get_ai_playlist_name"
