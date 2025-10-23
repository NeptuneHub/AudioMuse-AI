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
from typing import Optional
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


# --- Ollama Specific Function ---
def get_ollama_playlist_name(ollama_url, model_name, full_prompt):
    """
    Calls a self-hosted Ollama instance to get a playlist name.
    This version handles streaming responses and extracts only the non-think part.

    Args:
        ollama_url (str): The URL of your Ollama instance (e.g., "http://192.168.3.15:11434/api/generate").
        model_name (str): The Ollama model to use (e.g., "deepseek-r1:1.5b").
        full_prompt (str): The complete prompt text to send to the model.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Ollama API endpoint is usually just the base URL + /api/generate
    options = {
        "num_predict": 5000, # Max tokens to generate
        "temperature": 0.9
    }

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": True, # We handle streaming to get the full response
        "options": options
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        logger.debug("Starting API call for model '%s' at '%s'.", model_name, ollama_url)

        response = requests.post(ollama_url, headers=headers, data=json.dumps(payload), stream=True, timeout=960) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        full_raw_response_content = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_raw_response_content += chunk['response']
                    if chunk.get('done'):
                        break # Stop processing when the 'done' signal is received
                except json.JSONDecodeError:
                    logger.warning("Could not decode JSON line from stream: %s", line.decode('utf-8', errors='ignore'))
                    continue

        # Ollama models often include thought blocks, extract text after common thought tags
        # Using a simple approach: find the last occurrence of common thought block enders
        thought_enders = ["</think>", "[/INST]", "[/THOUGHT]"] # Add other common patterns if needed
        extracted_text = full_raw_response_content.strip()
        for end_tag in thought_enders:
             if end_tag in extracted_text:
                 extracted_text = extracted_text.split(end_tag, 1)[-1].strip() # Take everything after the last tag
        # The final cleaning and length check is done in the general function
        return extracted_text

    except requests.exceptions.RequestException as e:
        # Catch network-related errors, bad HTTP responses, etc.
        logger.error("Error calling Ollama API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."
    except Exception as e:
        # Catch any other unexpected errors.
        logger.error("An unexpected error occurred in get_ollama_playlist_name", exc_info=True)
        return "Error: AI service is currently unavailable."

# --- Gemini Specific Function ---
def get_gemini_playlist_name(gemini_api_key, model_name, full_prompt):
    """
    Calls the Google Gemini API to get a playlist name.

    Args:
        gemini_api_key (str): Your Google Gemini API key.
        model_name (str): The Gemini model to use (e.g., "gemini-2.5-pro").
        full_prompt (str): The complete prompt text to send to the model.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Allow any provided key, even if it's the placeholder, but check if it's empty/default
    if not gemini_api_key or gemini_api_key == "YOUR-GEMINI-API-KEY-HERE":
         return "Error: Gemini API key is missing or empty. Please provide a valid API key."
    
    try:
        # Read delay from environment/config if needed, otherwise use the default
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
        else:
            logger.debug("Gemini returned no content. Raw response: %s", response)
            return "Error: Gemini returned no content."

        # The final cleaning and length check is done in the general function
        return extracted_text

    except Exception as e:
        logger.error("Error calling Gemini API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."

# --- Mistral Specific Function ---
def get_mistral_playlist_name(mistral_api_key, model_name, full_prompt):
    """
    Calls the Mistral API to get a playlist name.

    Args:
        mistral_api_key (str): Your Mistral API key.
        model_name (str): The mistral model to use (e.g., "ministral-3b-latest").
        full_prompt (str): The complete prompt text to send to the model.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Allow any provided key, even if it's the placeholder, but check if it's empty/default
    if not mistral_api_key or mistral_api_key == "YOUR-MISTRAL-API-KEY-HERE":
         return "Error: Mistral API key is missing or empty. Please provide a valid API key."

    try:
        # Read delay from environment/config if needed, otherwise use the default
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
        else:
            logger.debug("Mistral returned no content. Raw response: %s", response)
            return "Error: mistral returned no content."

        # The final cleaning and length check is done in the general function
        return extracted_text

    except Exception as e:
        logger.error("Error calling Mistral API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."
    
# --- OpenAI Specific Function ---

def get_openai_playlist_name(
    openai_model_name: str,
    full_prompt: str,
    openai_api_key: str,
    openai_base_url: Optional[str] = None,
    temperature: float = 0.9,
    max_tokens: int = 500,
    stream: bool = True,
    timeout_seconds: int = 300,
) -> str:
    """
    Calls an OpenAI-compatible chat completions endpoint (can be OpenAI or a self-hosted Open-AI/proxy)
    to get a playlist name. Handles both streaming and non-streaming responses and extracts only
    the "final" assistant content (attempting to remove internal 'thought' blocks).

    Args:
        openai_model_name (str): Model name (e.g., "gpt-4o-mini" or "qwen3-4b").
        full_prompt (str): The prompt to send as a single user message (the function uses a simple messages array).
        openai_api_key (str): API key for Authorization header ("Bearer ...").
        openai_base_url (Optional[str]): If provided, used as base URL (e.g., "http://192.168.20.254:1234").
                                         If None, defaults to the official OpenAI URL "https://api.openai.com".
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum tokens to generate for the completion.
        stream (bool): Whether to request streaming output. If True, the function will parse stream events.
        timeout_seconds (int): HTTP request timeout in seconds.

    Returns:
        str: The cleaned assistant response (intended to be the playlist name) or an error message.
    """

    # If no base URL is provided, use the standard OpenAI REST base.
    # Then append the path for chat completions: /v1/chat/completions
    if openai_base_url:
        # Allow the user to pass either the full url including path or just the base host.
        # If they passed a host only (no /v1/chat/completions), ensure we append the path.
        if openai_base_url.endswith("/v1/chat/completions"):
            endpoint = openai_base_url
        else:
            endpoint = openai_base_url.rstrip("/") + "/v1/chat/completions"
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    # Build the messages array. We send the full_prompt as a single user message.
    # If you want to include a system role or system-level instructions, caller should include them in full_prompt.
    messages = [
        {
            "role": "system",
            "content": "The title needs to represent the mood and the activity of when you're listening to the playlist.",
        },
        {"role": "user", "content": full_prompt}
    ]

    payload = {
        "model": openai_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    try:
        logger.debug("Calling OpenAI-compatible endpoint at %s with model %s", endpoint, openai_model_name)

        # Use streaming if requested; this will allow incremental parsing of assistant content.
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), stream=stream, timeout=timeout_seconds)
        response.raise_for_status()

        full_raw_response_content = ""

        if stream:
            # OpenAI-style streaming: each event line begins with "data: "
            # The stream ends with a line: data: [DONE]
            # Each data: line contains a partial JSON object with "choices"[0]["delta"] fields.
            # We'll iterate over response.iter_lines() to handle chunked events robustly.
            for line_bytes in response.iter_lines(decode_unicode=False):
                if not line_bytes:
                    continue
                try:
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                except Exception:
                    # Fall back to safe decode; continue if we can't decode.
                    try:
                        line = line_bytes.decode("latin-1", errors="ignore").strip()
                    except Exception:
                        logger.warning("Couldn't decode streaming line; skipping a chunk.")
                        continue

                # The streaming protocol uses "data: " prefix. If not present, try to parse raw JSON.
                if line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
                else:
                    data_str = line

                # End-of-stream sentinel for OpenAI-style streaming
                if data_str == "[DONE]":
                    logger.debug("Received [DONE] sentinel from stream.")
                    break

                # Each data_str should be JSON; parse and extract any assistant delta content
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    # Not JSON — ignore and continue. This mirrors your Ollama approach of skipping bad lines.
                    logger.warning("Skipping non-JSON stream line: %s", data_str[:200])
                    continue

                # Two common patterns:
                # - OpenAI-style: chunk["choices"][0]["delta"].get("content")
                # - Some OpenAI proxies may send 'message' or 'content' directly.
                assistant_text_piece = ""
                try:
                    choices = chunk.get("choices")
                    if isinstance(choices, list) and len(choices) > 0:
                        choice0 = choices[0]
                        # OpenAI streaming delta:
                        delta = choice0.get("delta", {})
                        if isinstance(delta, dict):
                            assistant_text_piece = delta.get("content", "") or delta.get("text", "")
                        # In some variants the chunk may include 'message' with 'content'
                        if not assistant_text_piece:
                            message = choice0.get("message")
                            if message and isinstance(message, dict):
                                # message.content can be a string or a dict with parts; handle string primarily
                                content = message.get("content")
                                if isinstance(content, str):
                                    assistant_text_piece = content
                                elif isinstance(content, dict):
                                    # Some implementations structure content differently; join text fields if present
                                    # Example: {"content": {"type":"text","text":"..."}} -> try to extract
                                    assistant_text_piece = content.get("text", "") or content.get("parts", [None])[0] or ""
                except Exception:
                    # Be robust: if anything about the chunk parsing fails, log and continue.
                    logger.debug("Failed to parse streaming chunk structure; chunk keys: %s", list(chunk.keys()))
                    assistant_text_piece = ""

                if assistant_text_piece:
                    full_raw_response_content += assistant_text_piece

                # Some implementations provide a 'done' flag on the chunk (like your Ollama example)
                if isinstance(chunk, dict) and chunk.get("done"):
                    logger.debug("Chunk reported done=True; breaking stream loop.")
                    break

            # finished reading the stream
        else:
            # Non-streaming: full JSON response in one go
            # Parse and extract the assistant content from choices[0].message.content
            try:
                parsed = response.json()
            except Exception as e:
                logger.error("Failed to parse non-streaming JSON response: %s", e, exc_info=True)
                return "Error: AI service returned malformed response."

            # Typical structure: parsed["choices"][0]["message"]["content"]
            assistant_content = ""
            try:
                choices = parsed.get("choices", [])
                if choices and isinstance(choices, list):
                    first_choice = choices[0]
                    # Handle the OpenAI-style nested message
                    message = first_choice.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if isinstance(content, str):
                            assistant_content = content
                        else:
                            # Fallback if content is structured differently
                            assistant_content = json.dumps(content)
                    else:
                        # Older or proxy formats might put text directly
                        assistant_content = first_choice.get("text", "")
            except Exception:
                logger.exception("Unexpected structure in non-streaming response.")
                assistant_content = ""

            full_raw_response_content = (assistant_content or "").strip()

        # At this point full_raw_response_content should contain the assistant text (maybe with internal thought blocks).
        extracted_text = full_raw_response_content.strip()

        # Remove common 'think'/internal tags, keeping only the text after them.
        # Using the same approach as your Ollama function: split on known thought-end tags and take the text _after_ them.
        thought_enders = ["</think>", "[/INST]", "[/THOUGHT]", "<|end_think|>", "<|end_of_thought|>"]
        for end_tag in thought_enders:
            # If tag present, take everything after the last occurrence of the tag
            if end_tag in extracted_text:
                extracted_text = extracted_text.split(end_tag, 1)[-1].strip()

        # Additional conservative cleaning: some streams include role tags like "assistant:" or "### Assistant:"
        # If these appear at the beginning, remove them to get the pure content.
        for prefix in ["assistant:", "Assistant:", "### Assistant:", "Assistant —", "AI:" ]:
            if extracted_text.startswith(prefix):
                extracted_text = extracted_text[len(prefix):].strip()

        # Final safety/length check: if empty, return a helpful error message.
        if not extracted_text:
            logger.warning("Response parsed to empty string; returning service unavailable message.")
            return "Error: AI service did not return usable content."

        return extracted_text

    except requests.exceptions.RequestException as e:
        # Network error, timeout, connection refused, etc.
        logger.error("Network error when calling OpenAI-compatible API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."
    except Exception as e:
        # Any other unexpected failure — log full traceback and return a generic error string for the caller.
        logger.exception("An unexpected error occurred in get_openai_playlist_name")
        return "Error: AI service is currently unavailable."




# --- General AI Naming Function ---
def get_ai_playlist_name(provider, ollama_url, ollama_model_name, gemini_api_key, gemini_model_name, mistral_api_key, mistral_model_name, prompt_template, feature1, feature2, feature3, song_list, other_feature_scores_dict, openai_model_name, openai_api_key, openai_base_url):
    """
    Selects and calls the appropriate AI model based on the provider.
    Constructs the full prompt including new features.
    Applies length constraints after getting the name.
    """
    MIN_LENGTH = 15
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

    # Format the full song list for the prompt
    formatted_song_list = "\n".join([f"- {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}" for song in song_list]) # Send all songs

    # Construct the full prompt using the template and all features
    # The new prompt only requires the song list sample # type: ignore
    full_prompt = prompt_template.format(song_list_sample=formatted_song_list)

    logger.info("Sending prompt to AI (%s):\n%s", provider, full_prompt)

    # --- Call the AI Model ---
    name = "AI Naming Skipped" # Default if provider is NONE or invalid

    if provider == "OLLAMA":
        name = get_ollama_playlist_name(ollama_url, ollama_model_name, full_prompt)
    elif provider == "GEMINI":
        name = get_gemini_playlist_name(gemini_api_key, gemini_model_name, full_prompt)
    elif provider == "MISTRAL":
        name = get_mistral_playlist_name(mistral_api_key, mistral_model_name, full_prompt)
    elif provider == "OPENAI":
        name = get_openai_playlist_name(openai_model_name, full_prompt, openai_api_key, openai_base_url)
    # else: provider is NONE or invalid, name remains "AI Naming Skipped"

    # Apply length check and return final name or error
    # Only apply length check if a name was actually generated (not the skip message or an API error message)
    if name not in ["AI Naming Skipped"] and not name.startswith("Error"):
        cleaned_name = clean_playlist_name(name)
        if MIN_LENGTH <= len(cleaned_name) <= MAX_LENGTH:
            return cleaned_name
        else:
            # Return an error message indicating the length issue, but include the cleaned name for debugging
            return f"Error: AI generated name '{cleaned_name}' ({len(cleaned_name)} chars) outside {MIN_LENGTH}-{MAX_LENGTH} range."
    else:
        # Return the original skip message or API error message
        return name