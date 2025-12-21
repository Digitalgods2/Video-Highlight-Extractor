import streamlit as st
import os
import re
from dotenv import load_dotenv
from analysis_utils import get_transcript, analyze_humor, analyze_quotes, validate_and_expand_clips
from video_utils import download_video, create_gag_reel, create_single_clip, PRE_ROLL_BUFFER, POST_ROLL_BUFFER

# Load env vars
load_dotenv()

def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def run_clip_analysis(transcript, extraction_mode, api_key, max_clip_seconds, max_clips, provider, model):
    """
    Runs the appropriate analysis (humor or quotes) based on extraction_mode.
    Returns a list of (start, end) tuples, or None if no clips found.
    """
    if extraction_mode == "😂 Funny Moments":
        return analyze_humor(transcript, api_key, max_clip_seconds, max_clips, provider, model)
    else:
        return analyze_quotes(transcript, api_key, max_clip_seconds, max_clips, provider, model)

def main():
    st.set_page_config(page_title="Video Highlight Extractor", page_icon="🎬", layout="wide")
    
    # Initialize session state
    if 'cached_url' not in st.session_state:
        st.session_state.cached_url = None
    if 'cached_video_path' not in st.session_state:
        st.session_state.cached_video_path = None
    if 'cached_transcript' not in st.session_state:
        st.session_state.cached_transcript = None
    if 'found_intervals' not in st.session_state:
        st.session_state.found_intervals = None
    if 'selected_clips' not in st.session_state:
        st.session_state.selected_clips = {}
    if 'preview_clips' not in st.session_state:
        st.session_state.preview_clips = {}
    if 'step' not in st.session_state:
        st.session_state.step = 1  # 1=Input, 2=Preview, 3=Done
    
    st.title("🎬 Video Highlight Extractor")
    
    # Model configurations
    GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
    OPENAI_MODELS = ["gpt-5.2", "gpt-4o"]
    
    # Sidebar for Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Settings Section
        with st.expander("⚙️ Settings", expanded=True):
            # Provider Selection
            provider = st.selectbox(
                "AI Provider",
                ["Google Gemini", "OpenAI"],
                index=0
            )
            
            # Model Selection (changes based on provider)
            if provider == "Google Gemini":
                model = st.selectbox("Model", GEMINI_MODELS, index=0)
            else:
                model = st.selectbox("Model", OPENAI_MODELS, index=0)
            
            st.divider()
            
            # API Keys
            default_gemini_key = os.getenv("GEMINI_API_KEY", "")
            default_openai_key = os.getenv("OPENAI_API_KEY", "")
            
            gemini_key = st.text_input("Gemini API Key", value=default_gemini_key, type="password")
            openai_key = st.text_input("OpenAI API Key", value=default_openai_key, type="password")
            
            if st.button("💾 Save API Keys"):
                with open(".env", "w") as f:
                    f.write(f"GEMINI_API_KEY={gemini_key}\n")
                    f.write(f"OPENAI_API_KEY={openai_key}\n")
                os.environ["GEMINI_API_KEY"] = gemini_key
                os.environ["OPENAI_API_KEY"] = openai_key
                st.success("API Keys saved!")
        
        # Get the active API key based on provider
        if provider == "Google Gemini":
            api_key = gemini_key
        else:
            api_key = openai_key
        
        st.divider()
        st.subheader("Extraction Mode")
        extraction_mode = st.radio("What to extract:", ["😂 Funny Moments", "💬 Memorable Quotes"], index=0)
        
        st.divider()
        st.subheader("Clip Settings")
        max_clip_seconds = st.slider("Max Clip Length (seconds)", min_value=5, max_value=60, value=15, step=5)
        max_clips = st.slider("Max Number of Clips", min_value=3, max_value=50, value=10, step=1)
        
        st.divider()
        if st.button("🔄 Start Over"):
            st.session_state.step = 1
            st.session_state.found_intervals = None
            st.session_state.selected_clips = {}
            st.session_state.preview_clips = {}
            st.rerun()
    
    # ========== STEP 1: INPUT ==========
    if st.session_state.step == 1:
        st.markdown("### Step 1: Video Config")
        
        # Check if we have a cached session
        if st.session_state.cached_url:
            st.info(f"📁 **Current Video Loaded:** {st.session_state.cached_url}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Re-Analyze with New Settings", type="primary", use_container_width=True):
                    # Logic to re-run analysis using cached data
                    input_url = st.session_state.cached_url
                    confirm_rerun = True
                else:
                    confirm_rerun = False
            
            with col2:
                if st.button("🗑️ Start Fresh / New Video", use_container_width=True):
                    # Clear session state
                    st.session_state.cached_url = None
                    st.session_state.cached_video_path = None
                    st.session_state.cached_transcript = None
                    st.session_state.found_intervals = None
                    st.session_state.selected_clips = {}
                    st.session_state.preview_clips = {}
                    st.rerun()
            
            st.divider()
            st.caption("Change settings in the sidebar (Extraction Mode, Length, Count) then click 'Re-Analyze'.")

            if confirm_rerun:
                # Reuse cached data
                transcript = st.session_state.cached_transcript
                
                # Analyze for clips using helper function
                with st.spinner("Analyzing transcript..."):
                    intervals = run_clip_analysis(transcript, extraction_mode, api_key, max_clip_seconds, max_clips, provider, model)
                
                if not intervals:
                    st.warning("No clips found with current settings. Try adjusting the slider or changing modes.")
                    return
                
                # Validate and expand clips for completeness
                with st.spinner(f"Validating {len(intervals)} clips for completeness..."):
                    intervals = validate_and_expand_clips(transcript, intervals, api_key, max_clip_seconds, provider, model)
                
                st.success(f"Found {len(intervals)} validated clips!")
                st.session_state.found_intervals = intervals
                # Reset selections and specific previews regarding new intervals
                st.session_state.selected_clips = {i: True for i in range(len(intervals))}
                st.session_state.preview_clips = {} # Clear old previews as intervals changed
                st.session_state.step = 2
                st.rerun()

        else:
            # Standard Input Flow
            st.markdown("#### Enter Video Details")
            url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
            
            st.markdown("#### Manual Transcript (Optional)")
            manual_transcript = st.text_area("Paste transcript with timestamps", height=100, 
                                             help="Format: '0:05 text...' or '[0:05] text...'")
            
            if st.button("🔍 Find Clips", type="primary"):
                if not url:
                    st.error("Please enter a YouTube URL.")
                    return
                if not api_key:
                    st.error("Please enter an API Key in the sidebar.")
                    return
                
                video_id = extract_video_id(url)
                if not video_id:
                    st.error("Invalid YouTube URL.")
                    return
                
                # Fetch/parse transcript
                transcript = None
                if manual_transcript:
                    from analysis_utils import parse_manual_transcript
                    with st.spinner("Parsing transcript..."):
                        transcript = parse_manual_transcript(manual_transcript)
                        if not transcript:
                            st.error("Could not parse timestamps. Use format: '0:05 Hello'")
                            return
                else:
                    with st.spinner("Fetching transcript..."):
                        transcript = get_transcript(video_id)
                        if not transcript:
                            st.error("No transcript found. Try pasting one manually.")
                            return
                
                st.session_state.cached_transcript = transcript
                st.session_state.cached_url = url
                
                # Analyze for clips FIRST (before downloading)
                with st.spinner("Analyzing transcript..."):
                    intervals = run_clip_analysis(transcript, extraction_mode, api_key, max_clip_seconds, max_clips, provider, model)
                
                if not intervals:
                    st.warning("No clips found in this video. Try a different video or adjust settings.")
                    return
                
                # Validate and expand clips for completeness
                with st.spinner(f"Validating {len(intervals)} clips for completeness..."):
                    intervals = validate_and_expand_clips(transcript, intervals, api_key, max_clip_seconds, provider, model)
                
                st.success(f"Found {len(intervals)} validated clips! Downloading video...")
                
                # Download video ONLY if clips were found
                with st.spinner("Downloading video..."):
                    video_path = download_video(url)
                    if not video_path:
                        st.error("Failed to download video.")
                        return
                    st.session_state.cached_video_path = video_path

                st.session_state.found_intervals = intervals
                st.session_state.selected_clips = {i: True for i in range(len(intervals))}
                st.session_state.step = 2
                st.rerun()
    
    # ========== STEP 2: PREVIEW & SELECT ==========
    elif st.session_state.step == 2:
        st.markdown("### Step 2: Preview & Select Clips")
        st.info("Watch each clip preview and uncheck any you don't want to include.")
        
        intervals = st.session_state.found_intervals
        video_path = st.session_state.cached_video_path
        
        if not intervals or not video_path:
            st.error("No clips found. Please start over.")
            st.session_state.step = 1
            return
        
        # Create preview clips if not already done
        if not st.session_state.preview_clips:
            st.info("Generating preview clips... This may take a moment.")
            progress_bar = st.progress(0)
            
            for i, (start, end) in enumerate(intervals):
                progress_bar.progress((i + 1) / len(intervals), text=f"Creating preview {i+1} of {len(intervals)}...")
                try:
                    preview_path = create_single_clip(video_path, start, end, i)
                    if preview_path and os.path.exists(preview_path):
                        st.session_state.preview_clips[i] = preview_path
                    else:
                        print(f"Preview clip {i} creation returned: {preview_path}")
                except Exception as e:
                    print(f"Error creating preview {i}: {e}")
            
            progress_bar.empty()
            st.rerun()  # Rerun to show the previews
        
        # Display clips in a grid
        cols = st.columns(2)
        
        for i, (start, end) in enumerate(intervals):
            col = cols[i % 2]
            with col:
                # Calculate actual duration including the buffers added in video_utils
                actual_duration = (end + POST_ROLL_BUFFER) - max(0, start - PRE_ROLL_BUFFER)
                st.markdown(f"**Clip {i+1}** ({actual_duration:.1f}s)")
                
                # Show video preview
                preview_path = st.session_state.preview_clips.get(i)
                if preview_path and os.path.exists(preview_path):
                    st.video(preview_path)
                else:
                    st.warning(f"Preview unavailable ({start:.1f}s - {end:.1f}s)")
                
                # Checkbox to include/exclude
                st.session_state.selected_clips[i] = st.checkbox(
                    "Include this clip", 
                    value=st.session_state.selected_clips.get(i, True),
                    key=f"clip_{i}"
                )
                st.divider()
        
        # Count selected
        num_selected = sum(1 for v in st.session_state.selected_clips.values() if v)
        st.markdown(f"**Selected: {num_selected} of {len(intervals)} clips**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Input"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("🎬 Stitch Selected Clips", type="primary", disabled=(num_selected == 0)):
                # Get selected intervals
                selected_intervals = [intervals[i] for i, selected in st.session_state.selected_clips.items() if selected]
                
                with st.spinner("Creating final reel..."):
                    output_file = create_gag_reel(video_path, selected_intervals)
                    if output_file:
                        st.session_state.final_reel = output_file
                        st.session_state.step = 3
                        st.rerun()
                    else:
                        st.error("Failed to create reel.")
    
    # ========== STEP 3: DONE ==========
    elif st.session_state.step == 3:
        st.markdown("### Step 3: Your Reel is Ready! 🎉")
        
        output_file = getattr(st.session_state, 'final_reel', None)
        if output_file and os.path.exists(output_file):
            st.video(output_file)
            
            with open(output_file, "rb") as f:
                st.download_button(
                    "⬇️ Download Reel",
                    data=f,
                    file_name="highlight_reel.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Reel not found. Please start over.")
        
        if st.button("🔄 Create Another Reel"):
            st.session_state.step = 1
            st.session_state.found_intervals = None
            st.session_state.selected_clips = {}
            st.session_state.preview_clips = {}
            st.rerun()

if __name__ == "__main__":
    main()
