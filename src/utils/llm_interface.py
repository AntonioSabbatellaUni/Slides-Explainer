"""
LLM Interface Module
------------------
Handles interactions with AI models (Gemini and OpenRouter).
"""

import os
from typing import Dict, Optional, Union
import google.generativeai as genai
import aiohttp
import json
from PIL import Image
import base64
import io

class LLMInterface:
    """Interface for interacting with Language Models."""
    
    # Class-level attribute for Gemini models
    GEMINI_MODELS = {
        "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite-preview-02-05",
        "Gemini 2.0 Thinking": "gemini-2.0-flash-thinking-exp-01-21"
    }
    
    def __init__(self):
        """Initialize the LLM interface with API keys."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert a PIL Image to base64 string.
        
        Parameters
        ----------
        image : Image.Image
            The image to convert.
            
        Returns
        -------
        str
            Base64 encoded image string.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    async def generate_explanation(
        self,
        image: Image.Image,
        prompt: str,
        provider: str = "Gemini",
        model_name: str = "Gemini 2.0 Flash Lite",
        temperature: float = 0.7
    ) -> str:
        """
        Generate an explanation for the given image.
        
        Parameters
        ----------
        image : Image.Image
            The image to explain.
        prompt : str
            The prompt for the explanation.
        provider : str
            The AI provider to use ("Gemini" or "OpenRouter").
        model_name : str
            The model name to use (for Gemini, one of the GEMINI_MODELS keys).
        temperature : float
            The temperature parameter for generation.
            
        Returns
        -------
        str
            The generated explanation.
            
        Raises
        ------
        ValueError
            If the provider is not supported or API keys are missing.
        """
        if provider == "Gemini":
            if not self.gemini_api_key:
                raise ValueError("Gemini API key not found")
            
            model_id = self.GEMINI_MODELS.get(model_name, self.GEMINI_MODELS["Gemini 2.0 Flash Lite"])
            model = genai.GenerativeModel(model_id)
            response = await model.generate_content_async([prompt, image])
            return response.text
            
        elif provider == "OpenRouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key not found")
                
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "google/gemini-pro-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "image": self._image_to_base64(image)
                            }
                        ]
                    }
                ],
                "temperature": temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def chat_response(
        self,
        messages: list,
        provider: str = "Gemini",
        model_name: str = "Gemini 2.0 Flash Lite",
        temperature: float = 0.7
    ) -> str:
        """
        Generate a chat response.
        
        Parameters
        ----------
        messages : list
            List of chat messages.
        provider : str
            The AI provider to use.
        model_name : str
            The model name to use (for Gemini, one of the GEMINI_MODELS keys).
        temperature : float
            The temperature parameter.
            
        Returns
        -------
        str
            The generated response.
        """
        if provider == "Gemini":
            if not self.gemini_api_key:
                raise ValueError("Gemini API key not found")
            
            model_id = self.GEMINI_MODELS.get(model_name, self.GEMINI_MODELS["Gemini 2.0 Flash Lite"])
            model = genai.GenerativeModel(model_id)
            response = await model.generate_content_async(messages[-1]["content"])
            return response.text
            
        elif provider == "OpenRouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key not found")
                
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "google/gemini-pro",
                "messages": messages,
                "temperature": temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        
        else:
            raise ValueError(f"Unsupported provider: {provider}") 