import os
import re
import json
import http.client
from pathlib import Path
from typing import Dict, Any, List, Tuple
from io import BytesIO

import requests
from PIL import Image


class ImageGenerator:
    API_HOST = "xxxxx"
    API_KEY = "xxxxxx"

    @classmethod
    def generate_image(cls, *, prompt: str, save_path: str, max_width: int = 800, max_height: int = 600, **kwargs) -> Tuple[str, Tuple[int, int]]:
        try:
            conn = http.client.HTTPSConnection(cls.API_HOST)
            
            payload = json.dumps({
                "model": "gpt-image-1",
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "size": f"{max_width}x{max_height}"
            })
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {cls.API_KEY}',
                'Content-Type': 'application/json'
            }
            
            print(f"Sending request to generate image with prompt: {prompt}")
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            response = json.loads(data.decode("utf-8"))
            
            content = response['choices'][0]['message']['content']
            image_url_match = re.search(r'!\[.*?\]\((.*?)\)', content)
            if not image_url_match:
                raise ValueError("Cannot extract image URL from response")
            
            image_url = image_url_match.group(1)
            print(f"Image URL extracted: {image_url}")
            
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image = Image.open(BytesIO(image_response.content))
            image.save(save_path, "PNG")
            print(f"Image saved to: {save_path}")
            
            return save_path, image.size
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            raise
            
        finally:
            if 'conn' in locals():
                conn.close()


class VisualizationHelper:

    def __init__(self, save_dir: Path, avg_width: int = None, avg_height: int = None):
        self._figure_counter = 0
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.avg_width = avg_width
        self.avg_height = avg_height
        self.fast_mode = str(os.environ.get('P2P_FAST_MODE', '0')).lower() in {'1', 'true', 'yes', 'y'}
        
        self._default_width = 1024
        self._default_height = 768
        self._fast_width = 800
        self._fast_height = 600

    def set_figure_start_index(self, start_index: int):
        self._figure_counter = start_index

    def process_slide_content(self, slide_data: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        content = slide_data.get("content") or {}
        visual_elements: List[Dict[str, Any]] = content.get("visual_elements") or []
        core_points: List[str] = content.get("core_points") or []
        title: str = slide_data.get("slide_title") or ""
        slide_id = slide_data.get("slide_id")
        if slide_id is None:
            import hashlib
            seed = (title + "|" + (content.get("scholar_request", {}).get("reason") or "")).encode("utf-8")
            slide_id = hashlib.sha1(seed).hexdigest()[:8]

        print(f"\n=== Processing slide content for image generation ===")
        print(f"Title: {title}")
        
        scholar_req = content.get("scholar_request") or {}
        reason = (scholar_req.get("reason") or "").strip()
        rationale = "generated due to explicit image request"
        
        print(f"Scholar request reason: '{reason}'")

        if not reason:
            print("No reason provided in scholar_request, skipping image generation")
            return slide_data
            
        prompt = reason
        print(f"Using scholar request reason as prompt: {prompt}")

        new_id = self._next_figure_id()
        file_stem = f"{new_id.replace(' ', '_')}_S{slide_id}"
        save_path = self.save_dir / f"{file_stem}.png"

        if save_path.exists():
            print(f"\nImage already exists at {save_path}, reusing existing image")
            try:
                with Image.open(save_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading existing image dimensions: {e}")
                return slide_data
        else:
            try:
                print(f"\n============================Generating image============================")
                print(f"Prompt: {prompt}")
                print(f"Save path: {save_path}")
                
                target_width = self.avg_width or (self._default_width if not self.fast_mode else self._fast_width)
                target_height = self.avg_height or (self._default_height if not self.fast_mode else self._fast_height)
                
                target_width = max(target_width, 800) 
                target_height = max(target_height, 600) 
                print(f"Generating image with dimensions: {target_width}x{target_height}")
                image_path, (width, height) = ImageGenerator.generate_image(
                    prompt=prompt,
                    save_path=str(save_path),
                    max_width=target_width,
                    max_height=target_height
                )
                print(f"Successfully generated and saved image! Dimensions: {width}x{height}")
            except Exception as e:
                print(f"Failed to generate image: {str(e)}")
                return slide_data

        new_visual = {
            "id": new_id,
            "source": new_id,
            "type": "figure",
            "caption": title or "Illustrative diagram",
            "description": f"AI-generated visual to illustrate: {title}",
            "local_path": str(save_path),
            "width": width,
            "height": height,
            "is_generated": True,
            "generation_prompt": prompt, 
        }

        content.setdefault("visual_elements", visual_elements)
        content["visual_elements"].append(new_visual)
        content.setdefault("enhancements", {})["generated_visual_rationale"] = rationale
        slide_data["content"] = content
        
        print(f"Added new visual element with ID: {new_id}")
        return slide_data

    def _next_figure_id(self) -> str:
        self._figure_counter += 1
        return f"Generated Figure {self._figure_counter}"

def maybe_generate_visual_for_slide(save_dir: Path, slide: Dict[str, Any], avg_width=None, avg_height=None) -> Dict[str, Any]:
    content = slide.get("content") or {}
    scholar_req = content.get("scholar_request") or {}
    rtype = (scholar_req.get("type") or "none").lower()
    reason = (scholar_req.get("reason") or "").strip()
    
    slide_title = slide.get("slide_title", "Unknown Slide")
    print(f"\n=== maybe_generate_visual_for_slide ===")
    print(f"Slide: {slide_title}")
    print(f"Scholar request type: {rtype}")
    print(f"Reason: '{reason}'")
    
    if rtype != "image":
        print("Not an image request, skipping visual generation")
        return slide

    helper = VisualizationHelper(save_dir=save_dir, avg_width=avg_width, avg_height=avg_height)
    
    max_fig = 0
    ve: List[Dict[str, Any]] = content.get("visual_elements") or []
    for v in ve:
        if (v.get("type") == "figure") and isinstance(v.get("id"), str):
            import re
            m = re.search(r"(\d+)", v["id"])
            if m:
                max_fig = max(max_fig, int(m.group(1)))
    if max_fig:
        helper.set_figure_start_index(max_fig)

    print(f"Generating visual with reason: '{reason}'")
    updated = helper.process_slide_content(slide)
    slide.update(updated)

    content = slide.get("content") or {}
    visuals: List[Dict[str, Any]] = content.get("visual_elements") or []
    if len(visuals) > 3:
        preferred: List[Dict[str, Any]] = []
        others: List[Dict[str, Any]] = []
        for v in visuals:
            if v.get("is_generated") or v.get("type") == "table":
                preferred.append(v)
            else:
                others.append(v)
        new_visuals = preferred + others
        content["visual_elements"] = new_visuals[:3]
        slide["content"] = content
        print(f"Trimmed visuals to maximum of 3, kept {len(new_visuals[:3])} visuals")

    print(f"Visual generation completed for slide: {slide_title}")
    return slide
