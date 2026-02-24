"""
Color Calibration Card Removal Agent
Uses AI vision (GPT-4V or Claude) to detect and remove color cards from medical images
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
import base64
import os
import glob
from pathlib import Path
from typing import List, Optional, Literal
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import json

load_dotenv()

class CardDetectionResult(BaseModel):
    image_name: str = Field(description="Name of the image file (just filename)")
    has_card: bool = Field(description="Whether a color calibration card is present")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of detection"
    )
    x0: Optional[int] = Field(default=None, description="Left x coordinate of card")
    y0: Optional[int] = Field(default=None, description="Top y coordinate of card")
    x1: Optional[int] = Field(default=None, description="Right x coordinate of card")
    y1: Optional[int] = Field(default=None, description="Bottom y coordinate of card")
    notes: Optional[str] = Field(default="", description="Any additional observations")

class ColorCardRemovalAgent:
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the agent
        
        Args:
            model: Model to use - "gpt-4o" (OpenAI) or "claude-sonnet-4" (Anthropic)
        """
        self.model_name = model
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=CardDetectionResult)
        self.results = []

    def get_image_files(self, folder_path: str) -> List[str]:
        """Get all image files from folder"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        return sorted(image_files)

    def encode_image(self, image_path: str) -> str:
        """Convert image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def detect_card_in_image(self, image_path: str) -> CardDetectionResult:
        """
        Use AI to detect color calibration card in image
        """
        print(f"Analyzing: {os.path.basename(image_path)}")
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Get image dimensions for context
        img = Image.open(image_path)
        width, height = img.size
        
        # Create prompt
        prompt = f"""Analyze this medical image for a COLOR CALIBRATION CARD.

A color calibration card is a rectangular card with:
- A grid of colored squares (like a checkerboard)
- Multiple different colors (red, blue, green, yellow, brown, etc.)
- Often has a logo or text in the center
-has a longer green rectangular stem
- Used for standardizing colors in medical photography

IMPORTANT: 
- DO NOT confuse wounds/lesions with color cards
- Wounds are organic, irregular shapes with natural tissue
- Color cards are artificial, rectangular objects with bright colored squares

Image dimensions: {width}x{height} pixels

Respond ONLY with valid JSON matching this exact format:
{{
  "image_name": "{os.path.basename(image_path)}",
  "has_card": true or false,
  "confidence": "high" or "medium" or "low",
  "x0": left x coordinate (or null if no card),
  "y0": top y coordinate (or null if no card),
  "x1": right x coordinate (or null if no card),
  "y1": bottom y coordinate (or null if no card),
  "notes": "brief description of what you see"
}}

If there is NO color calibration card, set has_card to false and all coordinates to null.
If there IS a card, provide the bounding box coordinates.

{self.parser.get_format_instructions()}
"""

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        )

        # Get response
        try:
            response = self.llm.invoke([message])
            
            # Try to parse as JSON directly first
            try:
                # Clean response
                content = response.content.strip()
                # Remove markdown code blocks if present
                content = content.replace("```json", "").replace("```", "").strip()
                
                result_dict = json.loads(content)
                result = CardDetectionResult(**result_dict)
                
            except json.JSONDecodeError:
                # Fallback to Pydantic parser
                result = self.parser.parse(response.content)
            
            result.image_name = os.path.basename(image_path)
            
            print(f"  → Card detected: {result.has_card} ({result.confidence} confidence)")
            if result.has_card:
                print(f"    Bbox: ({result.x0}, {result.y0}) to ({result.x1}, {result.y1})")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Return safe default
            return CardDetectionResult(
                image_name=os.path.basename(image_path),
                has_card=False,
                confidence="low",
                notes=f"Error during detection: {str(e)}"
            )

    def remove_card_from_image(self, image_path: str, result: CardDetectionResult, 
                               output_path: str) -> bool:
        """
        Remove the detected card from image by filling with surrounding color
        """
        if not result.has_card or result.x0 is None:
            # No card, just copy original
            img = Image.open(image_path)
            img.save(output_path)
            return False
        
        img = Image.open(image_path)
        img_np = np.array(img)
        
        x0, y0, x1, y1 = result.x0, result.y0, result.x1, result.y1
        
        # Sample surrounding colors
        surrounding = []
        if y0 > 30:
            surrounding.append(img_np[y0-30:y0-10, x0:x1].mean(axis=(0,1)))
        if y1 < img.height - 30:
            surrounding.append(img_np[y1+10:y1+30, x0:x1].mean(axis=(0,1)))
        if x0 > 30:
            surrounding.append(img_np[y0:y1, x0-30:x0-10].mean(axis=(0,1)))
        if x1 < img.width - 30:
            surrounding.append(img_np[y0:y1, x1+10:x1+30].mean(axis=(0,1)))
        
        if surrounding:
            avg_color = tuple(np.median(surrounding, axis=0).astype(int))
            draw = ImageDraw.Draw(img)
            draw.rectangle((x0, y0, x1, y1), fill=avg_color)
        
        img.save(output_path)
        return True

    def process_folder(self, input_folder: str, output_folder: str, 
                      save_results: bool = True) -> List[CardDetectionResult]:
        """
        Process all images in folder
        
        Args:
            input_folder: Folder with original images
            output_folder: Folder to save cleaned images
            save_results: Whether to save detection results to CSV
        """
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all images
        image_files = self.get_image_files(input_folder)
        
        if not image_files:
            print(f"No images found in {input_folder}")
            return []
        
        print(f"\n{'='*70}")
        print(f"Processing {len(image_files)} images")
        print(f"Input: {input_folder}")
        print(f"Output: {output_folder}")
        print(f"{'='*70}\n")
        
        results = []
        cards_removed = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] ", end="")
            
            # Detect card
            result = self.detect_card_in_image(image_path)
            results.append(result)
            
            # Remove card and save
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, filename)
            
            removed = self.remove_card_from_image(image_path, result, output_path)
            if removed:
                cards_removed += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Processing complete!")
        print(f"   Total images: {len(image_files)}")
        print(f"   Cards detected: {sum(r.has_card for r in results)}")
        print(f"   Cards removed: {cards_removed}")
        print(f"   Output folder: {output_folder}")
        print(f"{'='*70}\n")
        
        # Save results
        if save_results:
            self.save_results(results, output_folder)
        
        self.results = results
        return results

    def save_results(self, results: List[CardDetectionResult], output_folder: str):
        """Save detection results to CSV and JSON"""
        output_path = Path(output_folder)
        
        # Save as CSV
        csv_path = output_path / "card_detection_results.csv"
        df = pd.DataFrame([r.dict() for r in results])
        df.to_csv(csv_path, index=False)
        print(f"📊 Results saved to: {csv_path}")
        
        # Save as JSON (with full details)
        json_path = output_path / "card_detection_results.json"
        with open(json_path, 'w') as f:
            json.dump([r.dict() for r in results], f, indent=2)
        print(f"📄 JSON saved to: {json_path}")

# Example usage
if __name__ == "__main__":
    # Initialize agent with GPT-4V
    agent = ColorCardRemovalAgent(model="gpt-5.2")
    
    # Process folder
    input_folder = "/Users/yusufabdulkadir/Leprosy AI Project/final_dataset/resized_leprosy/images"
    output_folder = "/Users/yusufabdulkadir/Leprosy AI Project/final_dataset/resized_leprosy/cleaned_images"
    
    results = agent.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        save_results=True
    )
    
    # Print summary
    print("\n📈 Summary by confidence:")
    df = pd.DataFrame([r.dict() for r in results])
    if len(df) > 0 and 'has_card' in df.columns:
        detected = df[df['has_card'] == True]
        if len(detected) > 0:
            print(detected.groupby('confidence').size())