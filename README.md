# Slide Explainer

An interactive tool for explaining presentation slides using AI and text-to-speech technology.

## Features

- Upload and process PowerPoint presentations or image files
- Interactive grid display of slides
- AI-powered slide explanation using Gemini or OpenRouter API
- Text-to-speech conversion with multiple provider options:
  - Google Text-to-Speech (gTTS)
  - ElevenLabs
  - Kokoro TTS (both local and web-hosted API)
- Voice input for questions
- Real-time Q&A chat interface
- Customizable explanation prompts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AntonioSabbatellaUni/Slides-Explainer
cd slide_explainer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
KOKORO_API_URL=your_kokoro_api_url  # Optional: for web-hosted Kokoro
KOKORO_MODEL_PATH=/path/to/kokoro/model  # Optional: for local Kokoro
```

## Usage

1. Start the application:
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your presentation or image files

4. Use the interface to:
   - Navigate through slides
   - Get AI explanations
   - Ask questions
   - Control audio playback
   - Choose TTS provider and voice options

### Using Kokoro TTS

The application supports both local and web-hosted versions of Kokoro TTS:

1. **Web-hosted API**:
   - Set `KOKORO_API_URL` in your `.env` file
   - Select "Kokoro" as TTS provider
   - Choose "API" as Kokoro TTS Type
   - Select your preferred voice

2. **Local Model**:
   - Download the Kokoro model
   - Set `KOKORO_MODEL_PATH` in your `.env` file
   - Select "Kokoro" as TTS provider
   - Choose "Local" as Kokoro TTS Type
   - Select your preferred voice

Available Kokoro voices: af, ar, cs, de, el, en, es, fi, fr, hi, hu, it, ja, ko, nl, pl, pt, ru, sv, tr, uk, vi, zh

## Project Structure

```
slide_explainer/
├── src/
│   ├── web/
│   │   ├── templates/
│   │   └── components/
│   ├── utils/
│   │   ├── slide_processor.py
│   │   ├── audio_handler.py
│   │   └── llm_interface.py
│   ├── models/
│   │   └── cache.py
│   └── app.py
├── data/
├── tests/
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 