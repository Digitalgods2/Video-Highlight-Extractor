# 🎬 Video Highlight Extractor

An AI-powered application that takes a YouTube video URL, analyzes its transcript, and automatically extracts the best moments into a highlight reel.

## Features

- **Dual AI Providers:**
  - 🤖 **Google Gemini** - Fast, reliable (Gemini 2.5 Flash/Pro)
  - 🧠 **OpenAI** - Powerful reasoning (GPT-5.2, GPT-4o)
- **Settings Dashboard**: Configure providers, models, and API keys in one place
- **Two Extraction Modes:**
  - 😂 **Funny Moments** - Creates a gag reel of the funniest sections
  - 💬 **Memorable Quotes** - Extracts profound, clever, weird, or quotable moments
- **Smart Validation**: AI verifies and expands clips to ensure complete thoughts (no cut-off sentences)
- **Quality Filtering**: Scored 1-10; strictly enforces "quality over quantity" (bad clips are discarded)
- **Clip Preview & Selection**: Preview all detected clips before stitching; uncheck any you don't want
- **Re-Analyze Without Re-Downloading**: Switch models or tweak settings instantly using cached video
- **Configurable Settings**:
  - Max clip length (5-60 seconds)
  - Max number of clips (3-50)
- **Context Buffers**: Clips include 0.5s pre-roll and 2.0s post-roll for complete sentences
- **API Key Persistence**: Save both keys to `.env` for convenience

## Prerequisites

- **Python 3.8+**
- **FFmpeg**: Must be installed and in your PATH ([Download](https://ffmpeg.org/download.html))
- **API Keys**:
  - [Google Gemini API Key](https://aistudio.google.com/) (Free tier available)
  - [OpenAI API Key](https://platform.openai.com/) (Optional, for GPT models)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Digitalgods2/Video-Highlight-Extractor.git
   cd Video-Highlight-Extractor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the **⚙️ Settings** expander in the sidebar:
   - Select **AI Provider** (Google Gemini or OpenAI)
   - Choose your **Model** (e.g., `gpt-4o`, `gemini-2.5-flash`)
   - Enter and Save your **API Keys**

3. Choose **Extraction Mode** (Funny Moments or Memorable Quotes) & adjust settings.

4. Paste a **YouTube URL** and click **🔍 Find Clips**.

5. **Preview & Select**: Watch each clip, uncheck any duds.

6. Click **🎬 Stitch Selected Clips** to generate your reel.

6. Download the final video!

## Project Structure

```
├── app.py              # Main Streamlit application
├── analysis_utils.py   # Transcript fetching & Gemini AI analysis
├── video_utils.py      # Video download (yt-dlp) & editing (FFmpeg/MoviePy)
├── requirements.txt    # Python dependencies
└── .env                # API key storage (created on first save)
```

## How It Works

1. **Transcript Extraction**: Fetches YouTube captions or parses manual input
2. **AI Analysis**: Sends transcript to Gemini with strict quality criteria
3. **Quality Filtering**: Each clip is scored 1-10; only 8+ are returned
4. **Video Download**: Uses yt-dlp to fetch the video file
5. **Preview Generation**: FFmpeg extracts each clip for preview
6. **Final Stitching**: MoviePy concatenates selected clips with black slugs

## Disclaimer

This tool relies on YouTube transcripts. Videos without captions cannot be processed automatically. Downloading copyrighted material may violate YouTube's Terms of Service; use responsibly.
