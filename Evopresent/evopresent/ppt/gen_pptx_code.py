import json
import os
import re
import tempfile
from PIL import Image
import base64
import traceback
from concurrent.futures import ThreadPoolExecutor
from evopresent.ppt.combine_slides import combine_html_slides
import mimetypes
import matplotlib
import matplotlib.pyplot as plt
from openai import OpenAI
from bs4 import BeautifulSoup
from evopresent.ppt.layout_extractor import extract_layout_sync
from evopresent.ppt.layout_checker import score_layout, advise_layout
from evopresent.ppt.mathjax_utils import inject_mathjax_and_group_equations
from evopresent.ppt.stable_mathjax import inject_mathjax_stable, validate_mathjax_injection
import html as html_lib
 
matplotlib.use('Agg') 
plt.rcParams['text.usetex'] = True  
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts,mathrsfs,bm,mathtools}'  
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

def get_project_root():
 
    root_dir = os.getenv('EVOPRESENT_ROOT')
    if root_dir and os.path.exists(root_dir):
        return root_dir
        
    current_dir = os.getcwd()
    
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'evopresent')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    return os.getcwd()


HTML_MODEL = os.getenv('EVOP_HTML_MODEL', None)
CHECKER_MODEL = os.getenv('EVOP_CHECKER_MODEL', None)

PROVIDER_CONFIG = {
    'deepseek': {
        'default_model': 'deepseek-chat',
        'base_url_env': 'EVOP_DEEPSEEK_BASE_URL',
        'fallback_base_url': 'https://api.deepseek.com',
        'api_key_envs': ['EVOP_DEEPSEEK_API_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY'],
    },
    'gpt-4o': {
        'default_model': 'gpt-4o',
        'base_url_env': 'EVOP_OPENAI_BASE_URL',
        'fallback_base_url': 'https://api.openai.com/v1',
        'api_key_envs': ['EVOP_OPENAI_API_KEY', 'OPENAI_API_KEY'],
    },
    'gpt-5': {
        'default_model': 'gpt-5',
        'base_url_env': 'EVOP_OPENAI_BASE_URL',
        'fallback_base_url': 'https://api.openai.com/v1',
        'api_key_envs': ['EVOP_OPENAI_API_KEY', 'OPENAI_API_KEY'],
    },

    'gemini': {
        'default_model': 'gemini-2.5-pro',
        'base_url_env': 'EVOP_GEMINI_BASE_URL',
        'fallback_base_url': "https://www.chataiapi.com/v1",
        'api_key_envs': ['EVOP_GEMINI_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY'],
    },
    'claude': {
        'default_model': 'claude-3-5-sonnet-20241022',
        'base_url_env': 'EVOP_CLAUDE_BASE_URL',
        'fallback_base_url': None,
        'api_key_envs': ['EVOP_CLAUDE_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY'],
    },
}

def _env_first(keys):
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return None

def _detect_provider_from_selector(selector: str) -> str:
    if not selector:
        return ''
    s = selector.lower()
    for key in PROVIDER_CONFIG.keys():
        if s == key or key in s:
            return key
    return ''

def _create_client_for_provider(provider_key: str):
    cfg = PROVIDER_CONFIG.get(provider_key)
    if not cfg:
        raise ValueError(f"Unsupported provider '{provider_key}'. Supported: {list(PROVIDER_CONFIG.keys())}")
    base_url = os.getenv(cfg['base_url_env']) or cfg['fallback_base_url']
    api_key = _env_first(cfg['api_key_envs']) or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(f"No API key found for provider '{provider_key}'. Checked {cfg['api_key_envs']} and OPENAI_API_KEY")
    if not base_url and provider_key in ('gemini', 'claude'):
        raise ValueError(f"Provider '{provider_key}' requires an OpenAI-compatible proxy. Set {cfg['base_url_env']}.")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def resolve_provider_and_create_client(selector: str, default_provider: str):
    provider = _detect_provider_from_selector(selector) or default_provider
    client = _create_client_for_provider(provider)
    if selector and selector.lower() in PROVIDER_CONFIG:
        model = PROVIDER_CONFIG[provider]['default_model']
    elif not selector:
        model = PROVIDER_CONFIG[provider]['default_model']
    else:
        model = selector
    return client, model


html_client = None
check_client = None
def configure_clients(html_selector: str, checker_selector: str):
    global html_client, check_client, HTML_MODEL, CHECKER_MODEL
    if html_client is None or not HTML_MODEL:
        html_client, HTML_MODEL = resolve_provider_and_create_client(html_selector, default_provider='deepseek')
    if check_client is None or not CHECKER_MODEL:
        check_client, CHECKER_MODEL = resolve_provider_and_create_client(checker_selector, default_provider='gpt')


def get_image_path(image_id, images_mapping, base_dir=None):
    """Get the path of an image"""
    try:
        if images_mapping and isinstance(images_mapping, dict):
            if image_id in images_mapping:
                if isinstance(images_mapping[image_id], dict) and 'path' in images_mapping[image_id]:
                    path = images_mapping[image_id]['path']
                else:
                    path = images_mapping[image_id]
                
                if os.path.exists(path):
                    rel_path = os.path.relpath(path, get_project_root())
                    return rel_path
                else:
                    print(f"Image file not found at: {path}")
        
        if not hasattr(get_image_path, 'model_name_t'):
            print("Warning: model_name_t not set for get_image_path")
            return None
        image_dir = os.path.join(f'{get_image_path.model_name_t}_images_and_tables')
        
        path = os.path.join(image_dir, f"{image_id}.png")
        if os.path.exists(os.path.join(get_project_root(), path)):
            return path
        
        if not image_id.startswith('Figure'):
            path = os.path.join(image_dir, f"Figure_{image_id}.png")
            if os.path.exists(os.path.join(get_project_root(), path)):
                return path
        
        print(f"Warning: Could not find image {image_id} in any of the expected locations")
        return None
        
    except Exception as e:
        print(f"Error finding image path: {e}")
        return None

def load_resource_mappings(base_dir, paper_name, model_name_t=None):
    images_mapping = {}
    
    if not model_name_t:
        raise ValueError("model_name_t is required")
        
    image_dir = os.path.join(get_project_root(), f"{model_name_t}_images_and_tables", paper_name)
    
    print(f"Searching for images in: {image_dir}")
    
    if not os.path.isdir(image_dir):
        print(f"Directory not found or is not a directory: {image_dir}")
        return {}

    try:
        for file_name in os.listdir(image_dir):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            file_path = os.path.join(image_dir, file_name)
            
            image_id = None
            
            match_table = re.search(r'-table-(\d+)', file_name, re.IGNORECASE)
            if match_table:
                num_id = match_table.group(1)
                image_id = f"Table {num_id}"
            else:
                
                match_picture = re.search(r'-picture-(\d+)', file_name, re.IGNORECASE)
                if match_picture:
                    num_id = match_picture.group(1)
                    image_id = f"Figure {num_id}"
                else:
                    
                    match_generated = re.search(r'Generated_Figure_(\d+)', file_name, re.IGNORECASE)
                    if match_generated:
                        num_id = match_generated.group(1)
                        image_id = f"Generated Figure {num_id}"

            if image_id:
                if image_id not in images_mapping:
                    images_mapping[image_id] = file_path
                else:
                    print(f"Warning: Duplicate image ID '{image_id}'. Keeping first found: {images_mapping[image_id]}. Ignoring: {file_path}")

    except Exception as e:
        print(f"Warning: Error while loading images: {e}")
        traceback.print_exc()
        
    return images_mapping
 

def get_slide_style(markdown_text=None, style_name=None):
    try:
        styles_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'styles')
        styles = {}
        for filename in os.listdir(styles_dir):
            if not filename.endswith('.css'):
                continue
            name = os.path.splitext(filename)[0]
            path = os.path.join(styles_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    styles[name] = f.read()
            except Exception:
                pass
        if style_name:
            key = style_name.strip().lower()
            if key in styles:
                return styles[key]
            print(f"Warning: Specified style '{style_name}' not found, falling back to default")
    except Exception as e:
        print(f"Warning: failed to load styles: {e}")
        return ''


async def render_html_to_image(html_content, slide_title, try_index=None, wait_for_animations=True, output_dir_name="critique_screenshots"):
    import asyncio
    from playwright.async_api import async_playwright
    import pathlib
    import os

    try:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, "debug_layouts", output_dir_name)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            output_dir = tempfile.gettempdir()

        safe_title = re.sub(r'[^\w\s-]', '', slide_title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        if not safe_title:
            safe_title = "untitled_slide"
        
        if try_index is not None:
            filename = f"{safe_title}_try_{try_index}.png"
        else:
            filename = f"{safe_title}.png"
        screenshot_path = os.path.join(output_dir, filename)

        
        temp_root_dir = os.path.join(project_root, "debug_layouts", "temp_html")
        try:
            os.makedirs(temp_root_dir, exist_ok=True)
        except Exception:
            temp_root_dir = output_dir
        temp_html_path = os.path.join(temp_root_dir, f"temp_{safe_title}.html")
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        file_uri = pathlib.Path(temp_html_path).resolve().as_uri()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-software-rasterizer',
                    '--disable-setuid-sandbox'
                ]
            )
            page = await browser.new_page()
            
            await page.set_viewport_size({'width': 1440, 'height': 810})
            
            await page.goto(file_uri, wait_until="load")
            
            
            if wait_for_animations:
                await page.wait_for_timeout(3000)  
                
                await page.wait_for_load_state("networkidle")

            try:
                await page.wait_for_function("window.MathJax || window._mathjaxReady === true", timeout=8000)
                
                await page.evaluate("""
                    () => {
                        if (window.MathJax && window.MathJax.typesetPromise) {
                            return window.MathJax.typesetPromise();
                        }
                        return null;
                    }
                """)
                await page.wait_for_function("window._mathjaxReady === true", timeout=4000)
            except Exception:
                pass
            

            await page.screenshot(
                path=screenshot_path,
                full_page=False 
            )
            
            await browser.close()
        try:
            os.remove(temp_html_path)
        except Exception:
            pass

        return screenshot_path
        
    except Exception as e:
        print(f"Error rendering HTML to image with Playwright: {e}")
        traceback.print_exc()
        try:
            if 'temp_html_path' in locals() and os.path.exists(temp_html_path):
                os.remove(temp_html_path)
            if 'screenshot_path' in locals() and os.path.exists(screenshot_path):
                os.remove(screenshot_path)
        except Exception:
            pass
        return None


def render_html_to_image_sync(html_content, slide_title, try_index=None, wait_for_animations=True, output_dir_name="critique_screenshots"):
    import asyncio
    
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(render_html_to_image(html_content, slide_title, try_index, wait_for_animations, output_dir_name))
            )
            return future.result()
    except RuntimeError:
        return asyncio.run(render_html_to_image(html_content, slide_title, try_index, wait_for_animations, output_dir_name))

def generate_layout_html(slide_data, image_info, client, slide_style, extra_guidance=None):
    try:
        title = slide_data.get('slide_title', '')
        core_points = slide_data.get('content', {}).get('core_points')
        main_points = core_points if core_points is not None else slide_data.get('content', {}).get('main_points', [])
        script_text = slide_data.get('content', {}).get('script', '')
        equations_obj = slide_data.get('content', {}).get('equations', {})

        prompt_path = os.path.join(get_project_root(), 'evopresent/ppt/prompts/slide_design_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        system_prompt_path = os.path.join(get_project_root(), 'evopresent/ppt/prompts/slide_design_system_prompt.txt')
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        enhanced_visual_elements = []
        num_images = len(image_info)
        CONTENT_WIDTH = 1440
        CONTENT_HEIGHT = 810

        total_text_length = sum(len(p) for p in (main_points or [])) + len(script_text or '')

        total_pixel_area = 0
        image_areas = []
        for img in image_info:
            width = img.get('width', 'unknown')
            height = img.get('height', 'unknown')
            area = 0
            if width != 'unknown' and height != 'unknown' and width > 0 and height > 0:
                area = width * height
            image_areas.append({'area': area, 'img': img})
            total_pixel_area += area

        base_image_fraction = 0.5
        text_reduction_factor = min(total_text_length / 2000, 0.4)  
        final_image_fraction = base_image_fraction * (1 - text_reduction_factor)

        if num_images == 1:
            final_image_fraction = min(0.5, final_image_fraction)  
        
        total_target_area = (CONTENT_WIDTH * CONTENT_HEIGHT) * final_image_fraction if num_images > 0 else 0

        for item in image_areas:
            img = item['img']
            width = img.get('width', 'unknown')
            height = img.get('height', 'unknown')
            original_area = item['area']
            
            aspect_ratio = 'unknown'
            orientation = 'unknown'
            suggested_width = 'auto'
            suggested_height = 'auto'
            
            if width != 'unknown' and height != 'unknown' and width > 0 and height > 0:
                aspect_ratio = round(width / height, 2)
                orientation = 'square'
                if width > height * 1.3:
                    orientation = 'landscape'
                elif height > width * 1.3:
                    orientation = 'portrait'

                
                if total_pixel_area > 0:
                    proportion = original_area / total_pixel_area
                    target_area = total_target_area * proportion
                else:
                    target_area = total_target_area / num_images if num_images > 0 else 0

                
                scale_factor = (target_area / original_area) ** 0.5 if original_area > 0 else 1.0
                
                
                scale_factor = max(0.2, min(scale_factor, 2.0))
                
                suggested_width = int(width * scale_factor)
                suggested_height = int(height * scale_factor)

                
                max_img_width = CONTENT_WIDTH * 0.9
                max_img_height = CONTENT_HEIGHT * 0.9
                if suggested_width > max_img_width:
                    scale_w = max_img_width / suggested_width
                    suggested_width = int(max_img_width)
                    suggested_height = int(suggested_height * scale_w)
                
                if suggested_height > max_img_height:
                    scale_h = max_img_height / suggested_height
                    suggested_height = int(max_img_height)
                    suggested_width = int(suggested_width * scale_h)

            enhanced_visual_elements.append({
                'id': img['id'],
                'type': img['type'],
                
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'orientation': orientation,
                'size_category': 'large' if (width != 'unknown' and width > 600) or (height != 'unknown' and height > 400) else 'medium',
                'suggested_width': suggested_width,
                'suggested_height': suggested_height
            })

        
        def _load_guidance_map(path: str):
            guidance = {}
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    current_key = None
                    current_lines = []
                    for raw_line in f:
                        line = raw_line.rstrip('\n')
                        if line.startswith('# '):
                            if current_key is not None:
                                guidance[current_key] = '\n'.join(current_lines).strip()
                            current_key = line[2:].strip()
                            current_lines = []
                        else:
                            current_lines.append(line)
                    if current_key is not None:
                        guidance[current_key] = '\n'.join(current_lines).strip()
            except Exception:
                guidance = {}
            return guidance

        guidance_path = os.path.join(get_project_root(), 'evopresent/ppt/prompts/image_placement_guidance.txt')
        guidance = _load_guidance_map(guidance_path)

        num_main_points = len(main_points)
   
        if num_images == 1:
            if enhanced_visual_elements:
                orientation = enhanced_visual_elements[0].get('orientation', 'unknown')
                prefix = guidance.get('single_prefix', '')
                if orientation == 'square':
                    suffix = guidance.get('single_square', '')
                elif orientation == 'landscape':
                    suffix = guidance.get('single_landscape', '')
                elif orientation == 'portrait':
                    suffix = guidance.get('single_portrait', '')
                else:
                    suffix = ''
                suggestion = (prefix + ' ' + suffix).strip()
                enhanced_visual_elements[0]['suggested_placement'] = suggestion
        elif num_images == 2:
            suggestion_base = guidance.get('two_base', '')
            if len(enhanced_visual_elements) > 0:
                enhanced_visual_elements[0]['suggested_placement'] = suggestion_base
            if len(enhanced_visual_elements) > 1:
                enhanced_visual_elements[1]['suggested_placement'] = suggestion_base
        elif num_images > 2:
            
            suggestion_base = guidance.get('many_base', '')
            suggestion_base = suggestion_base.format(num_images=num_images)
            for i in range(num_images):
                if len(enhanced_visual_elements) > i:
                    enhanced_visual_elements[i]['suggested_placement'] = suggestion_base

        main_points_text = '\n'.join([f"{point}" for point in main_points]) if main_points else ""
        
        equations_for_prompt = equations_obj if equations_obj else []
        prompt = prompt_template.format(
            title=title,
            core_points=main_points_text,
            script_text=(script_text or ""),
            visual_elements=json.dumps(enhanced_visual_elements, indent=2),
            equations=json.dumps(equations_for_prompt, indent=2) if equations_for_prompt else "",
            css_styles=slide_style
        )

        
        if extra_guidance:
            prompt += f"\n\nCHECKER FEEDBACK (MUST FIX IN THIS PASS):\n{extra_guidance}\n\nPlease revise layout and styling to address all issues while respecting the 1440x810 boundary rules.\n"

        
        aspect_ratio_styles = ""
        for img in enhanced_visual_elements:
            if img.get('width') != 'unknown' and img.get('height') != 'unknown' and img['height'] > 0:
                aspect_ratio_styles += f"div[data-image-id='{img['id']}'] {{ aspect-ratio: {img['width']} / {img['height']}; }}\n"
        
        
        try:
            extra_path = os.path.join(get_project_root(), 'evopresent/ppt/prompts/slide_design_extra_block.txt')
            with open(extra_path, 'r', encoding='utf-8') as f:
                extra_block = f.read()
            extra_block = extra_block.replace('{{ASPECT_RATIO_STYLES}}', aspect_ratio_styles)
            prompt += "\n\n" + extra_block + "\n"
        except Exception as e:
            prompt += "\n\nUse simple grey placeholders for data-image-id elements and keep all content within 1440x810.\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
 
        response = client.chat.completions.create(
             model=HTML_MODEL,
            messages=messages,
            # temperature=0.7,
        )

        layout_html = response.choices[0].message.content
        
        
        code_start = layout_html.find("```html")
        if code_start != -1:
            code_end = layout_html.rfind("```")
            if code_end > code_start:
                layout_html = layout_html[code_start + len("```html"):code_end].strip()

        search_content = layout_html.lower()
        start_index = search_content.find('<!doctype html>')
        if start_index == -1:
            start_index = search_content.find('<html')

        if start_index != -1:
            layout_html = layout_html[start_index:]
        
        layout_html = layout_html.strip()  

        return layout_html

    except Exception as e:
        print(f"Error generating layout HTML: {e}")
        traceback.print_exc()
        return None

def replace_image_placeholders(html_content, image_details_map):
    if not image_details_map:
        return html_content

    soup = BeautifulSoup(html_content, 'html.parser')
    placeholders = soup.find_all(attrs={"data-image-id": True})
    if not placeholders:
        return html_content

    for placeholder in placeholders:
        placeholder_id = placeholder.get('data-image-id')
        visual = image_details_map.get(placeholder_id)
        if not visual:
            print(f"Warning: No visual data found for placeholder_id: {placeholder_id}")
            continue

        full_img_path = visual.get('path')
        
        with open(full_img_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(full_img_path)
            if not mime_type:
                mime_type = 'image/png'
            data_url = f"data:{mime_type};base64,{img_data}"

        caption_text = visual.get("caption", "")
        original_placeholder_style = placeholder.get('style', '')
        
        style_properties = [prop.strip() for prop in original_placeholder_style.split(';') if prop.strip()]
        filtered_properties = [
            prop for prop in style_properties 
            if not prop.lower().startswith('border') 
            and not prop.lower().startswith('background-color')
            and not prop.lower().startswith('outline')
            and not prop.lower().startswith('box-shadow')
        ]
        cleaned_style = '; '.join(filtered_properties)

        
        figure_tag = soup.new_tag('figure', attrs={
            'style': cleaned_style + '; margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;'
        })

        
        img_tag = soup.new_tag('img', attrs={
            'src': data_url,
            'alt': caption_text,
            'loading': 'lazy',
            'style': 'max-width: 100%; max-height: 90%; object-fit: contain; display: block; width: auto; height: auto; border: none; outline: none;'
        })
        figure_tag.append(img_tag)

        
        if caption_text and not caption_text.strip().startswith(('Figure', 'figure')):
            figcaption_tag = soup.new_tag('figcaption', attrs={
                'style': 'text-align: center; margin-top: 8px; font-size: 14px; max-width: 100%; padding: 0 10px;'
            })
            
            cleaned_caption = re.sub(r'(?i)figure\s*[-.]?\s*\d+\s*[:.]\s*', '', caption_text)
            cleaned_caption = re.sub(r'(?i)figure\s*[-.]?\s*\d+\s*$', '', cleaned_caption)
            cleaned_caption = re.sub(r'(?i)^figure\s*[-.]?\s*\d+\s*', '', cleaned_caption)
            cleaned_caption = cleaned_caption.strip()
            if cleaned_caption:
                figcaption_tag.string = cleaned_caption
                figure_tag.append(figcaption_tag)

        
        placeholder.replace_with(figure_tag)
            
    return str(soup)


def generate_html_slide(slide_data, images_mapping=None, slide_style=None, use_checker=True, checker_threshold=8.7, checker_max_attempts=3):
    try:
        
        if slide_style is None:
            raise ValueError("slide_style must be provided")

        
        global html_client, check_client, HTML_MODEL, CHECKER_MODEL
        if html_client is None or not HTML_MODEL or check_client is None or not CHECKER_MODEL:
            configure_clients(
                os.getenv('EVOP_HTML_MODEL', 'deepseek'),
                os.getenv('EVOP_CHECKER_MODEL', 'gpt')
            )

        
        image_info_for_llm = []
        image_details_for_replacement = {}
 
        for visual in slide_data.get('content', {}).get('visual_elements', []):
            if visual.get('type') in ['figure', 'table']:
                img_path = None
                full_img_path = None
                width, height = None, None
                
                
                if visual.get('local_path'):
                    full_img_path = visual.get('local_path')
                    if os.path.isabs(full_img_path):
                        img_path = os.path.relpath(full_img_path, get_project_root())
                    else:
                        img_path = full_img_path
                        full_img_path = os.path.join(get_project_root(), full_img_path)
                    
                else:
                    
                    img_path = get_image_path(visual.get('id'), images_mapping)
                    if img_path:
                        full_img_path = os.path.join(get_project_root(), img_path) if not os.path.isabs(img_path) else img_path
                
                if full_img_path and os.path.exists(full_img_path):
                    try:
                        with Image.open(full_img_path) as img:
                            width, height = img.size
                    except Exception as e:
                        print(f"Warning: Could not get dimensions for image {img_path}. Error: {e}")
                elif full_img_path:
                    print(f"Warning: Image file not found at {full_img_path}")
                
                
                image_info_for_llm.append({
                    'id': visual.get('id'),
                    'type': visual.get('type'),
                    
                    'width': width,
                    'height': height,
                    'caption': visual.get('caption', ''),
                })
                
                
                if full_img_path and os.path.exists(full_img_path):
                    image_details_for_replacement[visual.get('id')] = {
                        'path': full_img_path,
                        'caption': visual.get('caption', '')
                    }
        if use_checker:
            max_attempts = int(checker_max_attempts)
            html_with_placeholders = None
            last_feedback_text = None
            
            
            attempts_data = []
            best_attempt = None
            
            for attempt in range(1, max_attempts + 1):
                html_with_placeholders = generate_layout_html(
                    slide_data,
                    image_info_for_llm,
                    html_client,
                    slide_style,
                    extra_guidance=last_feedback_text
                )
                if not html_with_placeholders:
                    print("Failed to generate unified HTML for slide.")
                    return None

                
                project_root = get_project_root()
                tmp_dir = os.path.join(project_root, "debug_layouts", "temp_html")
                os.makedirs(tmp_dir, exist_ok=True)
                safe_title = re.sub(r'[^\w\s-]', '', slide_data.get('slide_title', 'untitled_slide')).strip()
                safe_title = re.sub(r'[-\s]+', '_', safe_title)
                tmp_html_path = os.path.join(tmp_dir, f"checker_{safe_title}_attempt{attempt}.html")
                try:
                    with open(tmp_html_path, 'w', encoding='utf-8') as f:
                        f.write(html_with_placeholders)
                except Exception:
                    pass

                
                try:
                    
                    layout_payload = extract_layout_sync(tmp_html_path, 1440, 810, wait_seconds=2.5)
                except Exception as e:
                    print(f"Warning: layout extraction failed: {e}")
                    layout_payload = {"canvas": {"width": 1440, "height": 810}, "items": []}

                
                try:
                    slide_title = slide_data.get('slide_title', 'untitled_slide')
                    screenshot_path = render_html_to_image_sync(
                        html_with_placeholders,
                        slide_title,
                        try_index=attempt,
                        wait_for_animations=True,
                        output_dir_name="layout_checker_screenshots",
                    )
                except Exception:
                    screenshot_path = None

                
                try:
                    
                    slide_title = slide_data.get('slide_title', 'untitled_slide')
                    scoring = score_layout(
                        html_with_placeholders,
                        layout_payload,
                        client=check_client,
                        model=CHECKER_MODEL,
                        slide_title=slide_title,
                        screenshot_path=screenshot_path,
                    )
                    score = scoring.get('AestheticScore')
                    acceptable = (score is not None and float(score) > float(checker_threshold))
                    try_score_str = f"{float(score):.2f}" if score is not None else "None"
                    print(f"[LayoutChecker] slide='{slide_title}' attempt={attempt} score={try_score_str} acceptable={acceptable} threshold={checker_threshold}")
                except Exception as e:
                    print(f"Error in score_layout: {e}")
                    traceback.print_exc()
                    score = None
                    acceptable = False

                
                attempt_data = {
                    'html': html_with_placeholders,
                    'score': score if score is not None else 0.0,
                    'attempt_num': attempt,
                    'acceptable': acceptable
                }
                attempts_data.append(attempt_data)
                
                
                if best_attempt is None or (score is not None and score > best_attempt['score']):
                    best_attempt = attempt_data

                if acceptable:
                    break

                
                if attempt == max_attempts:
                    html_with_placeholders = best_attempt['html']
                    
                    
                    best_html_path = os.path.join(tmp_dir, f"checker_{safe_title}_BEST_attempt{best_attempt['attempt_num']}.html")
                    try:
                        with open(best_html_path, 'w', encoding='utf-8') as f:
                            f.write(html_with_placeholders)
                    except Exception:
                        pass
                    break

                
                try:
                    advice = advise_layout(
                        html_with_placeholders,
                        layout_payload,
                        client=check_client,
                        model=CHECKER_MODEL,
                        slide_title=slide_title,
                        screenshot_path=screenshot_path,
                    )
                    fixes = advice.get('Fixes', []) or []
                    if fixes:
                        last_feedback_text = "Fixes:\n- " + "\n- ".join(fixes)
                    else:
                        last_feedback_text = "Please refine alignment, spacing, hierarchy, and keep all elements within 1440x810 without overlap."
                except Exception:
                    last_feedback_text = "Please improve visual hierarchy, alignment, spacing and ensure all elements stay within 1440x810 bounds without overlap."
        else:
            
            html_with_placeholders = generate_layout_html(
                slide_data,
                image_info_for_llm,
                html_client,
                slide_style,
                extra_guidance=None
            )
            if not html_with_placeholders:
                print("Failed to generate HTML for slide (checker disabled).")
                return None

        
        if not image_info_for_llm and html_with_placeholders:
            soup = BeautifulSoup(html_with_placeholders, 'html.parser')
            for placeholder in soup.find_all(attrs={"data-image-id": True}):
                placeholder.decompose()
            html_with_placeholders = str(soup)

        
        try:
            final_rendered_html = replace_image_placeholders(html_with_placeholders, image_details_for_replacement)
        except Exception as e:
            print(f"Warning: Image embedding failed: {e}")
            final_rendered_html = html_with_placeholders

        
        try:
            def _escape_in_math_delims(text: str) -> str:
                if not text:
                    return text
                import re
                patterns = [
                    re.compile(r"\$\$([\s\S]*?)\$\$"),
                    re.compile(r"\$([^$]+?)\$"),
                    re.compile(r"\\\(([\s\S]*?)\\\)"),
                    re.compile(r"\\\[([\s\S]*?)\\\]")
                ]
                def repl(m):
                    inner = m.group(1)
                    
                    normalized = html_lib.unescape(inner)
                    normalized = normalized.replace(' | ', ' \\mid ')
                    escaped = (normalized
                               .replace('&', '&amp;')
                               .replace('<', '&lt;')
                               .replace('>', '&gt;'))
                    return m.group(0).replace(inner, escaped)
                for p in patterns:
                    text = p.sub(repl, text)
                return text
            final_rendered_html = _escape_in_math_delims(final_rendered_html)
        except Exception as e:
            print(f"Warning: math sanitization failed: {e}")

        
        try:
            if not validate_mathjax_injection(final_rendered_html):
                final_rendered_html = inject_mathjax_stable(final_rendered_html, slide_data)
                
                if validate_mathjax_injection(final_rendered_html):
                    pass
                else:
                    print("MathJax injection validation failed; using fallback")
                    final_rendered_html = inject_mathjax_and_group_equations(final_rendered_html, slide_data)
            else:
                pass
        except Exception as e:
            print(f"Stable MathJax injection failed, using fallback: {e}")
            try:
                final_rendered_html = inject_mathjax_and_group_equations(final_rendered_html, slide_data)
            except Exception as e2:
                print(f"Fallback injection also failed: {e2}")

        

        return final_rendered_html.strip()
        
    except Exception as e:
        print(f"Error generating HTML slide: {e}")
        traceback.print_exc()
        return None



def _generate_and_save_slide_worker(args):
    slide_index, slide_data, images_mapping, slide_style, output_dir, use_checker, checker_threshold, checker_max_attempts = args
    try:
        html_content = generate_html_slide(
            slide_data,
            images_mapping,
            slide_style,
            use_checker,
            checker_threshold=checker_threshold,
            checker_max_attempts=checker_max_attempts,
        )
        if html_content:
            output_path = os.path.join(output_dir, f'slide_{slide_index}.html')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            # Log per-slide completion with index and title (if available)
            slide_title = (slide_data.get('slide_title', '') or '').strip()
            if slide_title:
                print(f"Generated slide {slide_index+1}: {slide_title}")
            else:
                print(f"Generated slide {slide_index+1}: {output_path}")
        else:
            print(f"Failed to generate HTML for slide {slide_index}")
    except Exception as e:
        print(f"Error processing slide {slide_index}: {e}")
        traceback.print_exc()


def generate_html_presentation(outline_path, output_dir, base_dir=None, paper_name=None, model_name_t=None, style=None, use_checker=True, checker_scope='all', checker_threshold=8.7, checker_max_attempts=3):
    try:
        
        with open(outline_path, 'r') as f:
            outline_data = json.load(f)
        
        
        if not paper_name:
            raise ValueError("paper_name is required")
        if not model_name_t:
            raise ValueError("model_name_t is required")
        
        
        get_image_path.model_name_t = model_name_t
        
        images_mapping = load_resource_mappings(base_dir, paper_name, model_name_t)
        
        
        all_content = ""
        for slide in outline_data.get('slides', []):
            if 'content' in slide:
                
                all_content += slide.get('slide_title', '') + "\n"
                
                for point in slide['content'].get('main_points', []):
                    all_content += point + "\n"
                
                for visual in slide['content'].get('visual_elements', []):
                    all_content += visual.get('caption', '') + "\n"
                    all_content += visual.get('description', '') + "\n"
        
        
        slide_style = get_slide_style(all_content, style)
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        slides = outline_data.get('slides', [])
        
        original_count = len(slides)
        slides = [s for s in slides if not re.search(r"\bappendix\b", (s.get('slide_title', '') or ''), flags=re.IGNORECASE)]
        removed_count = original_count - len(slides)
        if removed_count > 0:
            pass
        
        normalized_scope = (checker_scope or 'all').lower()
        if normalized_scope in ('with-images', 'images'):
            normalized_scope = 'images'
        elif normalized_scope in ('without-images', 'no-images', 'text', 'text-only'):
            normalized_scope = 'text'
        elif normalized_scope in ('none',):
            normalized_scope = 'none'
        else:
            normalized_scope = 'all'

        task_args = []
        for slide_index, slide_data in enumerate(slides):
            
            visuals = (slide_data.get('content', {}) or {}).get('visual_elements', []) or []
            has_images = any(v.get('type') in ['figure', 'table'] for v in visuals)

            
            per_slide_checker = bool(use_checker)
            if per_slide_checker:
                if normalized_scope == 'images':
                    per_slide_checker = has_images
                elif normalized_scope == 'text':
                    per_slide_checker = not has_images
                elif normalized_scope == 'none':
                    per_slide_checker = False
                else:
                    per_slide_checker = True

            task_args.append((
                slide_index,
                slide_data,
                images_mapping,
                slide_style,
                output_dir,
                per_slide_checker,
                checker_threshold,
                checker_max_attempts,
            ))

        
        with ThreadPoolExecutor(max_workers=10) as executor:
            
            list(executor.map(_generate_and_save_slide_worker, task_args))

        return True
        
    except Exception as e:
        print(f"Error generating HTML presentation: {e}")
        traceback.print_exc()
        return False

def combine_html_slides(presentation_dir, output_path):
    
    from evopresent.ppt.combine_slides import combine_html_slides as _combine
    return _combine(presentation_dir, output_path)

def generate_presentation(outline_path, output_path, base_dir=None, paper_name=None, model_name_t=None, style=None, use_checker=True, checker_scope='all', checker_threshold=8.7, checker_max_attempts=3):
    try:
        presentation_dir = os.path.join(get_project_root(), 'results', paper_name, 'generated_presentations', 'slides')
        os.makedirs(presentation_dir, exist_ok=True)

        success = generate_html_presentation(
            outline_path=outline_path,
            output_dir=presentation_dir,
            base_dir=base_dir,
            paper_name=paper_name,
            model_name_t=model_name_t,
            style=style,
            use_checker=use_checker,
            checker_scope=checker_scope,
            checker_threshold=checker_threshold,
            checker_max_attempts=checker_max_attempts,
        )
        if not success:
            print("Failed to generate HTML slides")
            return False
        print("Combining HTML slides into a single presentation...")
        combine_html_slides(presentation_dir, output_path)

        return True
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate presentation from outline')
    parser.add_argument('--outline_path', required=True, help='Path to the outline JSON file')
    parser.add_argument('--output', required=True, help='Path to save the generated presentation (e.g., slides.html)')
    parser.add_argument('--base_dir', default=None,
                      help='Base directory of the project (optional, will try to detect automatically)')
    parser.add_argument('--paper_name', default=None,
                      help='Name of the paper (e.g., Nips, ICLR, etc.)')
    parser.add_argument('--model_name_t', required=True,
                      help='Name of the model (e.g., 4o, gpt4v, etc.)')
    parser.add_argument('--model_name_v', required=True,
                      help='Name of the model (e.g., 4o, gpt4v, etc.)')
    parser.add_argument('--style', default=None, help='Style name for the presentation')
    parser.add_argument('--checker', choices=['on', 'off'], default='on', help='Enable or disable layout checker loop')
    parser.add_argument('--checker-scope', choices=['all', 'images', 'text', 'none'], default='all',
                      help='Which slides to run checker on: all, only images, only text-only, or none')
    parser.add_argument('--html-model', default=os.getenv('EVOP_HTML_MODEL', 'deepseek'),
                      help='Generator: provider name (deepseek/gpt/gemini/claude) or explicit model id')
    parser.add_argument('--checker-model', default=os.getenv('EVOP_CHECKER_MODEL', 'gpt'),
                      help='Checker: provider name (deepseek/gpt/gemini/claude) or explicit model id')
    parser.add_argument('--checker-threshold', type=float, default=8.7,
                      help='Aesthetic score threshold to accept a layout (default: 8.7)')
    parser.add_argument('--checker-max-attempts', type=int, default=3,
                      help='Max attempts for the checker refinement loop (default: 3)')
    
    args = parser.parse_args()
    
    try:
        if not args.base_dir:
            args.base_dir = get_project_root()
        
        
        configure_clients(args.html_model, args.checker_model)

        success = generate_presentation(
            args.outline_path, 
            args.output, 
            args.base_dir,
            paper_name=args.paper_name,
            model_name_t=args.model_name_t,
            style=args.style,
            use_checker=(args.checker == 'on'),
            checker_scope=args.checker_scope,
            checker_threshold=args.checker_threshold,
            checker_max_attempts=args.checker_max_attempts,
            )
            
        if success:
            print(f"\nSuccessfully generated HTML presentation")
        else:
            print(f"\nFailed to generate HTML presentation")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

