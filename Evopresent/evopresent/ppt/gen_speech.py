import json
from camel.agents import ChatAgent
from camel.models import ModelFactory
from jinja2 import Environment, StrictUndefined
from openai import OpenAI

from typing import Dict, List, Optional

class SpeechContext:
    """
    Manage context for sequential speech-script generation across slides.
    Stores generated scripts and provides a brief context summary (previous slides and next slide).
    """
    def __init__(self):
        """Initialize context container."""
        self.generated_scripts: List[Dict] = []
        self.slides: List[Dict] = []
        
    def set_all_slides(self, slides: List[Dict]):
        """Set all slide metadata parsed from presentation.json."""
        self.slides = slides
        
    def add_script(self, script: Dict):
        """Append a generated speech script (with metadata) to history."""
        self.generated_scripts.append(script)
    
    def get_context_summary(self, current_index: int) -> Dict:
        """Return brief context for the current slide (previous two + next).

        Args:
            current_index (int): Zero-based index of the current slide.

        Returns:
            Dict: {'previous_slides': [...], 'next_slide': {...} or None}
        """
        context = {
            'previous_slides': [],
            'next_slide': None
        }
        
        # Look back at up to two already-generated scripts
        if current_index > 0:
            start_idx = max(0, current_index - 2)
            for i in range(start_idx, current_index):
                script = self.generated_scripts[i]
                context['previous_slides'].append({
                    'title': script['slide_title'],
                    'speech_script': script.get('speech_script', ''),
                })
        
        # Peek at the next slide if available
        if current_index + 1 < len(self.slides):
            next_slide = self.slides[current_index + 1]
            context['next_slide'] = {
                'title': next_slide.get('slide_title', ''),
                'main_points': next_slide.get('content', {}).get('main_points', [])
            }
            
        return context


# DeepSeek Configuration
deepseek_client = OpenAI(
    api_key="sk-a735b8725f33478cbb45465e5473c65f",
    base_url="https://api.deepseek.com"
)


def generate_speech_script(args, actor_config, content_result, text_content):
    """
    Generates a speech script for each slide sequentially, considering the context of previous slides.

    Args:
        args: Command line arguments.
        actor_config: Configuration for the language model actor.
        content_result: The parsed presentation content from presentation.json.
        text_content: The raw text content of the paper.

    Returns:
        A dictionary containing the speech scripts for each slide.
    """
    print("\n=================Generating Speech Script (Sequential)===================")

    

    with open("evopresent/ppt/prompts/speech_script_agent_single.txt", "r") as f:
        template_text = f.read()
        parts = template_text.split('\n\n', 1)
        system_msg = parts[0]
        template_content = parts[1]

    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(template_content)

    

    slides = content_result.get('slides', [])
    speech_context = SpeechContext()
    speech_context.set_all_slides(slides)
    all_speech_scripts = {'slides': []}

    for i, slide in enumerate(slides):
        print(f"Generating speech for slide {i + 1}/{len(slides)}: {slide.get('slide_title', 'Untitled')}")

        context_summary = speech_context.get_context_summary(i)

        slide_title_lower = slide.get('slide_title', '').lower()
        if 'introduction' in slide_title_lower or 'conclusion' in slide_title_lower or \
           'title' in slide_title_lower or 'thank' in slide_title_lower or \
           'reference' in slide_title_lower:
            slide_type = 'functional'
        else:
            slide_type = 'content'

        max_retries = 3
        retry_count = 0
        last_error = None
        speech_script = None

        while retry_count < max_retries:
            try:
                jinja_args = {
                    'paper_content': text_content,
                    'current_slide': slide,
                    'context_summary': context_summary,
                    'slide_type': slide_type
                }

                prompt = template.render(**jinja_args)

                

                # DeepSeek implementation
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                )
                response_content = response.choices[0].message.content
                if response_content.strip().startswith("```json"):
                    response_content = response_content.strip()[7:-4]
                if response_content.strip().startswith("```"):
                    response_content = response_content.strip()[3:-4]
                speech_script = json.loads(response_content)

                if 'speech_script' in speech_script:
                    print(f"Successfully generated speech script for slide {i + 1}.")
                    break
                else:
                    raise ValueError("Invalid JSON format from model: missing speech_script key.")
            except Exception as e:
                print(f"Error generating speech script for slide {i + 1} (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                last_error = e

        if speech_script is None:
            print(f"Failed to generate speech script for slide {i + 1} after {max_retries} attempts.")
            speech_script = {
                'speech_script': f"Error: Could not generate speech script. Last error: {last_error}"
            }
        
        # Add slide info to the script object
        speech_script['slide_index'] = i + 1
        speech_script['slide_title'] = slide.get('slide_title', 'Untitled')

        # Add to context for next slide and collect
        speech_context.add_script(speech_script)
        all_speech_scripts['slides'].append(speech_script)

    return all_speech_scripts
