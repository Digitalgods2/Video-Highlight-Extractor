from google import genai
from openai import OpenAI
import anthropic
from youtube_transcript_api import YouTubeTranscriptApi
import json
import os

# Model configurations for each provider
GEMINI_MODELS = ["gemini-3-pro", "gemini-3-flash", "gemini-3-deep-think"]
OPENAI_MODELS = ["gpt-5.2", "gpt-5.1", "gpt-5-mini", "o3-pro", "o4-mini"]
ANTHROPIC_MODELS = ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805", "claude-sonnet-4-20250514"]

def call_llm(prompt, provider, model, api_key):
    """
    Unified wrapper for calling LLM APIs (Gemini, OpenAI, or Anthropic).
    Returns the text response from the model.
    """
    if provider == "Google Gemini":
        client = genai.Client(api_key=api_key)
        print(f"DEBUG: Calling Gemini with model '{model}'")
        
        # Gemini 3 uses generation_config for output control
        # Deep Think models may need special handling
        is_deep_think = "deep-think" in model.lower()
        
        config = {
            "max_output_tokens": 8192,
            "temperature": 0.7 if not is_deep_think else None,  # Deep Think uses internal reasoning
        }
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            print(f"DEBUG: Gemini Error: {e}")
            # Fallback without config if it fails
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                return response.text.strip()
            except Exception:
                raise e
    
    elif provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        model = model.strip()
        print(f"DEBUG: Calling OpenAI with model '{model}'")
        
        # Determine model type for parameter selection
        is_reasoning = model.startswith("o1") or model.startswith("o3") or model.startswith("o4")
        is_gpt5 = model.startswith("gpt-5")
        
        # Base parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # GPT-5 series and reasoning models use max_completion_tokens
        # Reasoning models don't support temperature
        if is_reasoning:
            params["max_completion_tokens"] = 10000
            # No temperature for reasoning models
        elif is_gpt5:
            # GPT-5 uses max_completion_tokens (not max_tokens)
            params["max_completion_tokens"] = 8192
            params["temperature"] = 0.3 if "mini" in model else 0.7
        else:
            # Legacy models (gpt-4o, etc.) use max_tokens
            params["temperature"] = 0.7
            params["max_tokens"] = 4096

        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            print(f"DEBUG: OpenAI Error: {error_str}")
            
            # Catch 1: Temperature unsupported (reasoning model)
            if "temperature" in error_str and "unsupported" in error_str:
                print("DEBUG: Retrying without temperature...")
                params.pop("temperature", None)
                try:
                    response = client.chat.completions.create(**params)
                    return response.choices[0].message.content.strip()
                except Exception:
                    pass

            # Catch 2: max_tokens unsupported (needs max_completion_tokens)
            if "max_tokens" in error_str and "unsupported" in error_str:
                print("DEBUG: Retrying with max_completion_tokens...")
                params.pop("max_tokens", None)
                params["max_completion_tokens"] = 8192
                try:
                    response = client.chat.completions.create(**params)
                    return response.choices[0].message.content.strip()
                except Exception:
                    pass
            
            # Re-raise if we couldn't fix it
            raise e
    
    elif provider == "Anthropic":
        print(f"DEBUG: Calling Anthropic with model '{model}'")
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            # Anthropic Claude API uses messages format
            # max_tokens is required for Claude (controls output length)
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text from response content blocks
            return response.content[0].text.strip()
            
        except anthropic.APIError as e:
            print(f"DEBUG: Anthropic API Error: {e}")
            raise e
        except Exception as e:
            print(f"DEBUG: Anthropic Error: {e}")
            raise e
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_transcript(video_id):
    """
    Fetches the transcript for a given YouTube video ID.
    Returns a list of dictionaries with 'text', 'start', and 'duration'.
    """
    try:
        # v1.2.3+ uses instance-based API
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id)
        
        # Convert FetchedTranscriptSnippet objects to dicts
        result = []
        for snippet in transcript_data:
            result.append({
                'text': snippet.text,
                'start': snippet.start,
                'duration': getattr(snippet, 'duration', 0)
            })
        return result
        
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def get_transcript_text_for_interval(transcript, start, end):
    """
    Extracts the combined text from transcript entries that overlap with [start, end].
    Returns tuple: (combined_text, first_entry_index, last_entry_index)
    """
    texts = []
    first_idx = None
    last_idx = None
    
    for i, entry in enumerate(transcript):
        entry_start = entry['start']
        entry_end = entry_start + entry.get('duration', 3)
        
        # Check if this entry overlaps with our interval
        if entry_end >= start and entry_start <= end:
            texts.append(entry['text'])
            if first_idx is None:
                first_idx = i
            last_idx = i
    
    return ' '.join(texts), first_idx, last_idx


def validate_clip_completeness(text, api_key, provider="Google Gemini", model="gemini-2.5-flash"):
    """
    Asks LLM if the text is a complete thought.
    Returns tuple: (is_complete: bool, issue: str or None)
    """
    prompt = f"""
    You are checking if a video transcript excerpt is a COMPLETE thought.
    
    TEXT: "{text}"
    
    Analyze this text and determine:
    1. Does it START mid-sentence? (missing beginning)
    2. Does it END mid-sentence? (cut off)
    3. Is the main idea/joke/point fully expressed?
    
    Respond with ONLY valid JSON (no markdown):
    - If complete: {{"complete": true}}
    - If incomplete: {{"complete": false, "issue": "starts_mid_sentence" OR "ends_mid_sentence" OR "both"}}
    """
    
    try:
        text_response = call_llm(prompt, provider, model, api_key)
        
        # Clean markdown if present
        if text_response.startswith("```"):
            text_response = text_response.split("```")[1]
            if text_response.startswith("json"):
                text_response = text_response[4:]
        text_response = text_response.strip()
        
        result = json.loads(text_response)
        return result.get('complete', True), result.get('issue', None)
        
    except Exception as e:
        print(f"Error validating clip: {e}")
        return True, None  # Assume complete on error


def validate_and_expand_clips(transcript, intervals, api_key, max_clip_seconds, provider="Google Gemini", model="gemini-2.5-flash"):
    """
    Validates each clip for completeness and expands boundaries if needed.
    Loops up to MAX_EXPANSION_PASSES times per clip to handle multi-entry thoughts.
    DISCARDS clips that cannot be completed within max_clip_seconds (quality over quantity).
    Returns a new list of (start, end) tuples with corrected timestamps.
    """
    MAX_EXPANSION_PASSES = 3
    validated_intervals = []
    discarded_count = 0
    
    for i, (start, end) in enumerate(intervals):
        current_start = start
        current_end = end
        clip_is_valid = True  # Track if clip should be kept
        
        for pass_num in range(MAX_EXPANSION_PASSES):
            # Get the text for this interval
            text, first_idx, last_idx = get_transcript_text_for_interval(transcript, current_start, current_end)
            
            if not text or first_idx is None:
                print(f"Clip {i+1}: No transcript text found, discarding")
                clip_is_valid = False
                break
            
            # Check completeness
            is_complete, issue = validate_clip_completeness(text, api_key, provider, model)
            
            if is_complete:
                if pass_num == 0:
                    print(f"Clip {i+1}: Complete ✓")
                else:
                    print(f"Clip {i+1}: Complete after {pass_num} expansion(s) ✓")
                break
            
            if pass_num == 0:
                print(f"Clip {i+1}: Incomplete ({issue}) - expanding...")
            else:
                print(f"  Pass {pass_num + 1}: Still incomplete ({issue}) - expanding more...")
            
            expanded = False
            
            # Expand backwards if starts mid-sentence
            if issue in ['starts_mid_sentence', 'both']:
                if first_idx > 0:
                    prev_entry = transcript[first_idx - 1]
                    new_start = prev_entry['start']
                    if new_start < current_start:
                        print(f"  Expanded start: {current_start:.1f}s -> {new_start:.1f}s")
                        current_start = new_start
                        expanded = True
            
            # Expand forwards if ends mid-sentence
            if issue in ['ends_mid_sentence', 'both']:
                if last_idx < len(transcript) - 1:
                    next_entry = transcript[last_idx + 1]
                    new_end = next_entry['start'] + next_entry.get('duration', 3)
                    if new_end > current_end:
                        print(f"  Expanded end: {current_end:.1f}s -> {new_end:.1f}s")
                        current_end = new_end
                        expanded = True
            
            # If we couldn't expand further, stop trying
            if not expanded:
                print(f"  Cannot expand further (at transcript boundary) - DISCARDING")
                clip_is_valid = False
                break
            
            # Check if we've hit max clip length - DISCARD instead of clamp
            if current_end - current_start > max_clip_seconds:
                print(f"  Exceeded max {max_clip_seconds}s and still incomplete - DISCARDING (quality over quantity)")
                clip_is_valid = False
                break
        
        if clip_is_valid:
            validated_intervals.append((current_start, current_end))
        else:
            discarded_count += 1
    
    if discarded_count > 0:
        print(f"\n🗑️ Discarded {discarded_count} clips that couldn't be completed within {max_clip_seconds}s")
    
    return validated_intervals

def analyze_humor(transcript, api_key, max_clip_seconds=15, max_clips=5, provider="Google Gemini", model="gemini-2.5-flash"):
    """
    Sends the transcript to LLM to identify humorous sections.
    Returns a list of (start, end) tuples.
    """
    if not api_key:
        raise ValueError("API Key is required")

    # Prepare transcript for prompt - include start AND end times with clear labels
    formatted_transcript = ""
    for i, entry in enumerate(transcript):
        start = entry['start']
        duration = entry.get('duration', 3)  # Default 3s if missing
        end = start + duration
        text = entry['text']
        formatted_transcript += f"LINE {i+1} | START:{start:.2f}s | END:{end:.2f}s | TEXT: \"{text}\"\n"

    prompt = f"""
    You are an expert video editor and comedian. Your task is to analyze the following transcript of a YouTube video and identify the FUNNIEST sections to create a "gag reel".
    
    CRITICAL INSTRUCTION: You must return valid JSON only. Do not wrap it in markdown code blocks.
    The JSON should be a list of objects with "start", "end", "humor_score", and "reasoning" fields.
    
    HUMOR SCORE (1-10):
    - 10: Absolutely hilarious, guaranteed laugh-out-loud moment
    - 9: Very funny, would make most people laugh
    - 8: Genuinely funny, solid comedic moment
    - 7: Amusing but not remarkable
    - 6 or below: Mildly interesting but not actually funny
    
    Example: 
    [
        {{"start": 10.5, "end": 20.0, "humor_score": 9, "reasoning": "Perfect timing on the joke about penguins, followed by genuine laughter."}}, 
        {{"start": 45.0, "end": 60.0, "humor_score": 8, "reasoning": "Absurd situation that escalates quickly, very unexpected."}}
    ]
    
    TIMING IS CRITICAL - Each transcript line shows [START_TIME - END_TIME]:
    - Your clip "end" time MUST be the END_TIME of the last line you want to include, NOT the START_TIME
    - The END_TIME is when the speaker FINISHES saying that line
    - If you use a START_TIME as your end, you will CUT OFF their sentence mid-word
    - Always include an extra 1-2 seconds after the last line's END_TIME for breathing room
    
    CRITERIA:
    1. Focus ONLY on genuinely HUMOROUS content: jokes, punchlines, funny reactions, laughter, comedic timing, or absurd statements.
    2. Do NOT include general "interesting" or "engaging" content unless it is actually funny.
    3. Each clip should be {max_clip_seconds} seconds or less.
    4. BE HONEST with your humor_score - do not inflate scores. Only rate 8+ if it's GENUINELY funny.
    5. REASONING IS REQUIRED: You must explain in 1 sentence WHY this specific moment is funny.
    
    CRITICAL QUANTITY RULES (READ THIS CAREFULLY):
    - {max_clips} is the ABSOLUTE MAXIMUM, NOT a target or goal
    - NEVER return exactly {max_clips} clips - if you return exactly {max_clips}, you are probably padding
    - Return ONLY clips that are genuinely funny - if only 2 moments are funny, return only 2
    - IT IS BETTER TO RETURN 3 GREAT CLIPS THAN {max_clips} MEDIOCRE ONES
    - DO NOT pad with low-quality content to fill the quota
    - If the video has no funny moments, return an EMPTY list []
    - Be RUTHLESSLY selective - when in doubt, leave it out
    - Ask yourself: "Would I actually share this clip?" If not, DON'T INCLUDE IT
    - Most videos have 5-10 genuinely funny moments at most, not {max_clips}
    
    Here is the transcript:
    {formatted_transcript}
    """

    try:
        text_response = call_llm(prompt, provider, model, api_key)
        
        # Cleanup if model adds markdown
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.startswith("```"):
            text_response = text_response[3:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
        
        text_response = text_response.strip()
        
        # Try to parse as JSON first
        try:
            clips = json.loads(text_response)
        except json.JSONDecodeError:
            # Fallback: use regex to extract timestamp pairs with optional humor_score
            import re
            print(f"JSON parse failed, trying regex fallback...")
            # Updated regex to handle reasoning (though simple regex might fail on complex reasoning strings, this is a fallback)
            pattern = r'"start"\s*:\s*([\d.]+)\s*,\s*"end"\s*:\s*([\d.]+)\s*,\s*"humor_score"\s*:\s*(\d+)\s*,\s*"reasoning"\s*:\s*(".*?")'
            matches = re.findall(pattern, text_response)
            if matches:
                # Fallback won't capture reasoning perfectly via regex, so we mock it if needed
                clips = [{"start": float(m[0]), "end": float(m[1]), "humor_score": int(m[2]), "reasoning": m[3].strip('"')} for m in matches]
            else:
                print(f"Regex fallback also failed")
                return []
        
        # Process clips: filter by humor_score and enforce max length
        MINIMUM_HUMOR_SCORE = 8
        results = []
        filtered_count = 0
        
        for clip in clips:
            start = float(clip['start'])
            end = float(clip['end'])
            humor_score = int(clip.get('humor_score', 10))  # Default to 10 if missing
            reasoning = clip.get('reasoning', 'No reasoning provided')
            
            # Filter out clips below the minimum humor score
            if humor_score < MINIMUM_HUMOR_SCORE:
                filtered_count += 1
                print(f"Filtered out clip ({start:.1f}s - {end:.1f}s) Score: {humor_score}. Reason: {reasoning}")
                continue
            
            print(f"ACCEPTED clip ({start:.1f}s - {end:.1f}s) Score: {humor_score}. Reason: {reasoning}")
            
            # Enforce max clip length
            if end - start > max_clip_seconds:
                end = start + max_clip_seconds
            
            results.append((start, end))
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} clips with humor_score below {MINIMUM_HUMOR_SCORE}")

        # Hard limit based on user setting
        if len(results) > max_clips:
            print(f"Enforcing max limit: Trimming {len(results)} clips down to {max_clips}")
            results = results[:max_clips]
            
        return results

    except Exception as e:
        print(f"Error analyzing humor: {e}")
        # Return empty list on failure so the app doesn't crash
        return []

def analyze_quotes(transcript, api_key, max_clip_seconds=15, max_clips=5, provider="Google Gemini", model="gemini-2.5-flash"):
    """
    Sends the transcript to LLM to identify memorable quotes.
    Returns a list of (start, end) tuples.
    """
    if not api_key:
        raise ValueError("API Key is required")

    # Prepare transcript for prompt - include start AND end times with clear labels
    formatted_transcript = ""
    for i, entry in enumerate(transcript):
        start = entry['start']
        duration = entry.get('duration', 3)  # Default 3s if missing
        end = start + duration
        text = entry['text']
        formatted_transcript += f"LINE {i+1} | START:{start:.2f}s | END:{end:.2f}s | TEXT: \"{text}\"\n"

    prompt = f"""
    You are a world-class video editor with impeccable taste. Your job is to find ONLY the most EXCEPTIONAL moments in this transcript - the kind of quotes that would make someone stop scrolling and share the video.
    
    CRITICAL INSTRUCTION: You must return valid JSON only. Do not wrap it in markdown code blocks.
    The JSON should be a list of objects with "start", "end", "quality_score", and "reasoning" fields.
    
    QUALITY SCORE (1-10):
    - 10: Absolutely legendary quote, instantly shareable
    - 9: Exceptional moment that would make someone stop scrolling
    - 8: Genuinely great quote worth including
    - 7: Decent but not remarkable
    - 6 or below: Not worth including
    
    Example: 
    [
        {{"start": 10.5, "end": 20.0, "quality_score": 9, "reasoning": "Profound insight about life that resonates universally."}}, 
        {{"start": 45.0, "end": 60.0, "quality_score": 8, "reasoning": "Very witty remark that perfectly sums up the situation."}}
    ]
    
    TIMING IS CRITICAL - Each transcript line shows [START_TIME - END_TIME]:
    - Your clip "end" time MUST be the END_TIME of the last line you want to include, NOT the START_TIME
    - The END_TIME is when the speaker FINISHES saying that line
    - If you use a START_TIME as your end, you will CUT OFF their sentence mid-word
    - Always add 2-3 seconds AFTER the last line's END_TIME for breathing room
    
    ABSOLUTE REQUIREMENTS FOR COMPLETENESS (READ CAREFULLY):
    - **NEVER cut off a sentence mid-thought.**
    - The selected text MUST end with a sentence-ending punctuation mark (period, question mark, or exclamation point).
    - If the line you picked doesn't end with punctuation, YOU MUST INCLUDE THE NEXT LINE.
    - check the LINE BEFORE: Does the sentence start there? If so, INCLUDE IT.
    - check the LINE AFTER: Does the sentence continue there? If so, INCLUDE IT.
    - It is better to include slightly too much context than to cut off the idea.
    - Each quote MUST be a COMPLETE, SELF-CONTAINED IDEA.
    - Verify that the start and end timestamps cover the ENTIRE sentence structure.
    
    ABSOLUTE REQUIREMENTS:
    - Include 1-2 seconds BEFORE the quote starts for context
    - BE HONEST with your quality_score - do not inflate scores. Only rate 8+ if it's truly exceptional.
    - REASONING IS REQUIRED: You must explain in 1 sentence WHY this specific quote is exceptional.
    
    WHAT MAKES A QUOTE WORTH INCLUDING (must fit AT LEAST ONE):
    1. FUNNY - genuinely hilarious, not just mildly amusing. Would make someone laugh out loud.
    2. WEIRD - bizarre, unexpected, or strange enough to be memorable
    3. PROFOUND - deep insight that makes you think differently about something important
    4. GREAT ADVICE - actionable wisdom that could genuinely help someone's life, career, or relationships
    5. QUOTABLE - a phrase so well-crafted it could become a catchphrase or be printed on a t-shirt
    
    WHAT TO REJECT (DO NOT include these):
    - Trivial observations or small talk
    - Incomplete thoughts or sentences that trail off
    - Generic statements anyone could make
    - Transitions like "so anyway" or "moving on"
    - Questions without memorable answers
    - Anything that requires context from before/after to understand
    
    CRITICAL QUANTITY RULES (READ THIS CAREFULLY):
    - {max_clips} is the ABSOLUTE MAXIMUM, NOT a target or goal
    - NEVER return exactly {max_clips} clips - if you return exactly {max_clips}, you are probably padding
    - Return ONLY clips that genuinely meet the criteria above
    - IT IS BETTER TO RETURN 3 GREAT CLIPS THAN {max_clips} MEDIOCRE ONES
    - DO NOT pad with low-quality content to fill the quota
    - If only 2 moments qualify, return only 2
    - If the video has no qualifying moments, return an EMPTY list []
    - Be RUTHLESSLY selective - when in doubt, leave it out
    - Ask yourself: "Would I actually share this clip?" If not, DON'T INCLUDE IT
    - Most videos have 5-10 genuinely quotable moments at most, not {max_clips}
    - Each clip MUST be {max_clip_seconds} seconds or less
    
    Here is the transcript:
    {formatted_transcript}
    """

    try:
        text_response = call_llm(prompt, provider, model, api_key)
        
        # Cleanup if model adds markdown
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.startswith("```"):
            text_response = text_response[3:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
        
        text_response = text_response.strip()
        
        # Try to parse as JSON first
        try:
            clips = json.loads(text_response)
        except json.JSONDecodeError:
            # Fallback: use regex to extract timestamp pairs with optional quality_score
            import re
            print(f"JSON parse failed, trying regex fallback...")
            pattern = r'"start"\s*:\s*([\d.]+)\s*,\s*"end"\s*:\s*([\d.]+)(?:\s*,\s*"quality_score"\s*:\s*(\d+))?'
            matches = re.findall(pattern, text_response)
            if matches:
                clips = [{"start": float(m[0]), "end": float(m[1]), "quality_score": int(m[2]) if m[2] else 10} for m in matches]
            else:
                print(f"Regex fallback also failed")
                return []
        
        # Process clips: filter by quality_score and enforce max length
        MINIMUM_QUALITY_SCORE = 8
        results = []
        filtered_count = 0
        
        for clip in clips:
            start = float(clip['start'])
            end = float(clip['end'])
            quality_score = int(clip.get('quality_score', 10))  # Default to 10 if missing
            reasoning = clip.get('reasoning', 'No reasoning provided')
            
            # Filter out clips below the minimum quality score
            if quality_score < MINIMUM_QUALITY_SCORE:
                filtered_count += 1
                print(f"Filtered out quote ({start:.1f}s - {end:.1f}s) Score: {quality_score}. Reason: {reasoning}")
                continue
            
            # Enforce max clip length
            if end - start > max_clip_seconds:
                end = start + max_clip_seconds
            
            results.append((start, end))
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} quotes with quality_score below {MINIMUM_QUALITY_SCORE}")
            
        # Hard limit based on user setting
        if len(results) > max_clips:
            print(f"Enforcing max limit: Trimming {len(results)} clips down to {max_clips}")
            results = results[:max_clips]
            
        return results

    except Exception as e:
        print(f"Error analyzing quotes: {e}")
        return []

def parse_manual_transcript(text):
    """
    Parses a manually pasted transcript string into the expected format:
    [{'start': 0.0, 'text': 'words', 'duration': 5.0}, ...]
    
    Supports:
    [00:12] Hello world
    0:12 Hello world
    00:12
    Hello world
    """
    import re
    
    # Regex to find timestamps: HH:MM:SS or MM:SS or M:SS
    timestamp_regex = r"\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?"
    
    lines = text.split('\n')
    entries = []
    
    current_entry = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(timestamp_regex, line)
        if match:
            # Found a line starting with a timestamp
            ts_str = match.group(1)
            parts = list(map(int, ts_str.split(':')))
            
            seconds = 0
            if len(parts) == 3: # HH:MM:SS
                seconds = parts[0]*3600 + parts[1]*60 + parts[2]
            elif len(parts) == 2: # MM:SS
                seconds = parts[0]*60 + parts[1]
                
            # Remove timestamp from text
            remaining_text = line[match.end():].strip()
            
            # If we had a previous entry, save it
            if current_entry and current_entry['text']:
                entries.append(current_entry)
            
            # Start new entry (duration will be calculated after)
            current_entry = {'start': float(seconds), 'text': remaining_text}
            
        else:
            # No timestamp, append to current entry text
            if current_entry:
                if current_entry['text']:
                    current_entry['text'] += " " + line
                else:
                    current_entry['text'] = line
                    
    # Append the last one
    if current_entry and current_entry['text']:
        entries.append(current_entry)
    
    # Calculate durations based on gaps between timestamps
    for i in range(len(entries)):
        if i < len(entries) - 1:
            # Duration = next entry's start - this entry's start
            entries[i]['duration'] = entries[i+1]['start'] - entries[i]['start']
        else:
            # Last entry - estimate 5 seconds
            entries[i]['duration'] = 5.0
        
        # Ensure minimum duration of 2 seconds
        if entries[i]['duration'] < 2.0:
            entries[i]['duration'] = 2.0
        
    return entries
