"""
Slide Processor Module
--------------------
Handles processing of presentation files and images.
"""

from typing import List, Union, Tuple
from pathlib import Path
import io
from PIL import Image
from pptx import Presentation
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
from PyPDF2 import PdfReader

class SlideProcessor:
    """Handles the processing of presentation slides and images."""
    
    @staticmethod
    def process_file(file: Union[str, Path, io.BytesIO]) -> List[Image.Image]:
        """
        Process an uploaded file (PPTX, PDF, or image) and return a list of PIL Images.
        
        Parameters
        ----------
        file : Union[str, Path, io.BytesIO]
            The file to process. Can be a path or file-like object.
            
        Returns
        -------
        List[Image.Image]
            List of processed images.
            
        Raises
        ------
        ValueError
            If the file format is not supported.
        """
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            suffix = file_path.suffix.lower()
            
            if suffix == '.pptx':
                return SlideProcessor._process_pptx(file_path)
            elif suffix == '.pdf':
                return SlideProcessor._process_pdf_file(file_path)
            else:
                return [Image.open(file_path)]
        else:
            # Handle uploaded file
            if hasattr(file, 'type'):
                # Streamlit UploadedFile
                if file.type == 'application/pdf':
                    # Save PDF content to a temporary file
                    pdf_bytes = io.BytesIO(file.getvalue())
                    return SlideProcessor._process_pdf_bytes(pdf_bytes)
                elif file.type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
                    # Save PPTX content to a temporary file
                    pptx_bytes = io.BytesIO(file.getvalue())
                    return SlideProcessor._process_pptx(pptx_bytes)
                elif file.type.startswith('image/'):
                    return [Image.open(io.BytesIO(file.getvalue()))]
                else:
                    raise ValueError(f"Unsupported file type: {file.type}")
            else:
                # Try to determine file type from content
                content = file.read() if hasattr(file, 'read') else file
                try:
                    return [Image.open(io.BytesIO(content))]
                except:
                    try:
                        return SlideProcessor._process_pdf_bytes(io.BytesIO(content))
                    except:
                        try:
                            return SlideProcessor._process_pptx(io.BytesIO(content))
                        except:
                            raise ValueError("Unable to determine file type")
    
    @staticmethod
    def _process_pptx(pptx_file: Union[str, Path, io.BytesIO]) -> List[Image.Image]:
        """
        Extract images from a PowerPoint presentation.
        
        Parameters
        ----------
        pptx_file : Union[str, Path, io.BytesIO]
            The PowerPoint file to process.
            
        Returns
        -------
        List[Image.Image]
            List of slide images.
        """
        presentation = Presentation(pptx_file)
        slides = []
        
        for slide in presentation.slides:
            # Convert slide to image
            # Note: This is a simplified version. In practice, you'd need to
            # implement proper slide-to-image conversion
            slide_image = Image.new('RGB', (1920, 1080), 'white')
            slides.append(slide_image)
            
        return slides
    
    @staticmethod
    def _process_pdf_file(pdf_path: Union[str, Path]) -> List[Image.Image]:
        """
        Convert PDF file to list of images.
        
        Parameters
        ----------
        pdf_path : Union[str, Path]
            Path to the PDF file.
            
        Returns
        -------
        List[Image.Image]
            List of page images.
        """
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                dpi=200,  # Adjust DPI as needed
                fmt='png',
                poppler_path=None  # Will use system-installed poppler
            )
            return images
        except Exception as e:
            raise ValueError(f"Failed to process PDF file: {e}")
    
    @staticmethod
    def _process_pdf_bytes(pdf_bytes: io.BytesIO) -> List[Image.Image]:
        """
        Convert PDF bytes to list of images.
        
        Parameters
        ----------
        pdf_bytes : io.BytesIO
            PDF file as bytes.
            
        Returns
        -------
        List[Image.Image]
            List of page images.
        """
        try:
            # Convert PDF pages to images
            images = convert_from_bytes(
                pdf_bytes.getvalue(),
                dpi=200,  # Adjust DPI as needed
                fmt='png',
                poppler_path=None  # Will use system-installed poppler
            )
            return images
        except Exception as e:
            raise ValueError(f"Failed to process PDF content: {e}")
    
    @staticmethod
    def create_grid(
        images: List[Image.Image],
        cols: int,
        target_size: Tuple[int, int] = (800, 600)
    ) -> Image.Image:
        """
        Create a grid of images.
        
        Parameters
        ----------
        images : List[Image.Image]
            List of images to arrange in a grid.
        cols : int
            Number of columns in the grid.
        target_size : Tuple[int, int]
            Target size for each image in the grid.
            
        Returns
        -------
        Image.Image
            Combined grid image.
        """
        if not images:
            raise ValueError("No images provided")
            
        # Resize images to target size
        resized_images = [
            img.resize(target_size, Image.Resampling.LANCZOS)
            for img in images
        ]
        
        # Calculate grid dimensions
        n_images = len(resized_images)
        rows = (n_images + cols - 1) // cols
        
        # Create blank canvas
        grid_width = cols * target_size[0]
        grid_height = rows * target_size[1]
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Place images in grid
        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            x = col * target_size[0]
            y = row * target_size[1]
            grid_image.paste(img, (x, y))
            
        return grid_image
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
        """
        Convert a PIL Image to bytes.
        
        Parameters
        ----------
        image : Image.Image
            The image to convert.
        format : str
            The output format (default: 'PNG').
            
        Returns
        -------
        bytes
            The image data as bytes.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue() 