"""
Audio Handler Module
-----------------
Handles text-to-speech conversion using various providers.
"""

import os
from typing import Union, Optional, Dict
from pathlib import Path
import tempfile
import requests
import json
from gtts import gTTS
# Commenting out ElevenLabs imports due to compatibility issues
# from elevenlabs import generate, Voice, set_api_key
import torchaudio

class KokoroTTS:
    """Handles Kokoro TTS functionality."""
    
    def __init__(self, local_model_path: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize Kokoro TTS.
        
        Parameters
        ----------
        local_model_path : Optional[str]
            Path to local Kokoro model directory.
        api_url : Optional[str]
            URL for web-hosted Kokoro TTS service.
        """
        self.local_model_path = local_model_path
        self.api_url = api_url or os.getenv("KOKORO_API_URL")
        self.is_local = local_model_path is not None
        
        if self.is_local:
            try:
                import torch
                import torchaudio
                # Configure torchaudio backend
                torchaudio.set_audio_backend("soundfile")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Load model here (simplified for example)
                self.model = None  # TODO: Implement actual model loading
            except ImportError as e:
                print(f"Failed to load local Kokoro dependencies: {e}")
                self.is_local = False
    
    def generate_speech(self, text: str, voice_id: str = "af") -> bytes:
        """
        Generate speech using Kokoro TTS.
        
        Parameters
        ----------
        text : str
            Text to convert to speech.
        voice_id : str
            Voice ID to use (default: "af").
            
        Returns
        -------
        bytes
            Audio data in WAV format.
        """
        if self.is_local:
            return self._generate_local(text, voice_id)
        else:
            return self._generate_api(text, voice_id)
    
    def _generate_local(self, text: str, voice_id: str) -> bytes:
        """Generate speech using local Kokoro model."""
        # TODO: Implement local generation
        # This is a placeholder for the actual implementation
        raise NotImplementedError("Local Kokoro TTS not implemented")
    
    def _generate_api(self, text: str, voice_id: str) -> bytes:
        """Generate speech using Kokoro API."""
        if not self.api_url:
            raise ValueError("Kokoro API URL not configured")
            
        try:
            response = requests.post(
                f"{self.api_url}/tts",
                json={
                    "text": text,
                    "voice_id": voice_id
                }
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Kokoro API request failed: {e}")

class AudioHandler:
    """Handles text-to-speech conversion."""
    
    def __init__(self):
        """Initialize the audio handler with API keys and TTS providers."""
        # Initialize API keys
        # Commenting out ElevenLabs initialization
        # self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        # if self.elevenlabs_api_key:
        #     set_api_key(self.elevenlabs_api_key)
        
        # Initialize Kokoro TTS
        self.kokoro = KokoroTTS(
            local_model_path=os.getenv("KOKORO_MODEL_PATH"),
            api_url=os.getenv("KOKORO_API_URL")
        )
            
        # Create temporary directory for audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "slide_explainer_audio"
        self.temp_dir.mkdir(exist_ok=True)
    
    def text_to_speech(
        self,
        text: str,
        provider: str = "gTTS",
        language: str = "en",
        voice_id: Optional[str] = None,
        kokoro_type: str = "api"  # 'api' or 'local'
    ) -> Path:
        """
        Convert text to speech.
        
        Parameters
        ----------
        text : str
            The text to convert to speech.
        provider : str
            The TTS provider to use ("gTTS" or "Kokoro").
        language : str
            The language code (for gTTS).
        voice_id : Optional[str]
            The voice ID (for Kokoro).
        kokoro_type : str
            Kokoro TTS type ('api' or 'local').
            
        Returns
        -------
        Path
            Path to the generated audio file.
            
        Raises
        ------
        ValueError
            If the provider is not supported or API keys are missing.
        """
        # Generate unique filename
        output_file = self.temp_dir / f"audio_{hash(text)}.mp3"
        
        if provider == "gTTS":
            tts = gTTS(text=text, lang=language)
            tts.save(str(output_file))
            
        # Commenting out ElevenLabs support
        # elif provider == "ElevenLabs":
        #     if not self.elevenlabs_api_key:
        #         raise ValueError("ElevenLabs API key not found")
        #         
        #     voice = Voice(
        #         voice_id=voice_id or "EXAVITQu4vr4xnSDxMaL"  # Default voice
        #     )
        #     
        #     audio = generate(
        #         text=text,
        #         voice=voice,
        #         model="eleven_monolingual_v1"
        #     )
        #     
        #     with open(output_file, "wb") as f:
        #         f.write(audio)
                
        elif provider == "Kokoro":
            try:
                audio_data = self.kokoro.generate_speech(
                    text,
                    voice_id=voice_id or "af"
                )
                with open(output_file, "wb") as f:
                    f.write(audio_data)
            except Exception as e:
                raise RuntimeError(f"Kokoro TTS failed: {e}")
                
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        return output_file
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old audio files.
        
        Parameters
        ----------
        max_age_hours : int
            Maximum age of files to keep (in hours).
        """
        import time
        current_time = time.time()
        
        for file in self.temp_dir.glob("*.mp3"):
            file_age = current_time - file.stat().st_mtime
            if file_age > max_age_hours * 3600:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
    
    def __del__(self):
        """Cleanup temporary files on object destruction."""
        self.cleanup_old_files() 