
from dotenv import load_dotenv
from utils.src.utils import get_json_from_response
from evopresent.ppt.visualization_helper import VisualizationHelper
from utils.src.model_utils import parse_pdf
import json
import random
import os

from camel.models import ModelFactory
from camel.agents import ChatAgent
from tenacity import retry, stop_after_attempt
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat, BoundingBox
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from pathlib import Path
import PIL
from PIL import ImageOps
from collections import defaultdict
from marker.models import create_model_dict
from utils.wei_utils import *
from utils.pptx_utils import *
import torch
from jinja2 import Template, Environment, StrictUndefined
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import html

load_dotenv()
IMAGE_RESOLUTION_SCALE = 5.0

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.do_formula_enrichment = True  # Enable formula enrichment

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def _expand_and_clamp_bbox(bbox: BoundingBox, page_w: float, page_h: float, pad_x_ratio: float = 0.06, pad_y_ratio: float = 0.08) -> BoundingBox:
    """Expand bbox by ratios and clamp to page bounds."""
    l, r = bbox.l, bbox.r
    b, t = bbox.b, bbox.t
    width = max(0.0, r - l)
    height = max(0.0, t - b)
    pad_x = width * pad_x_ratio
    pad_y = height * pad_y_ratio

    l_exp = max(0.0, l - pad_x)
    r_exp = min(page_w, r + pad_x)
    b_exp = max(0.0, b - pad_y)
    t_exp = min(page_h, t + pad_y)

    # Ensure valid box after clamping
    if r_exp <= l_exp:
        mid_x = (l + r) * 0.5
        l_exp = max(0.0, mid_x - 1.0)
        r_exp = min(page_w, mid_x + 1.0)
    if t_exp <= b_exp:
        mid_y = (b + t) * 0.5
        b_exp = max(0.0, mid_y - 1.0)
        t_exp = min(page_h, mid_y + 1.0)

    return BoundingBox(l=l_exp, t=t_exp, r=r_exp, b=b_exp, coord_origin=bbox.coord_origin)

def _sanitize_latex_equation(eq_text: str) -> str:
    """Sanitize LaTeX for safe MathJax rendering inside HTML.

    - Normalize whitespace
    - Undo any prior HTML entities, then escape <, >, &
    - Convert spaced conditional bar ' | ' to '\\mid'
    """
    try:
        # Normalize whitespace
        eq = re.sub(r"\s+", " ", eq_text or "").strip()

        # Undo common entities to avoid double-escaping, then escape
        eq = eq.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        eq = html.escape(eq, quote=False)

        # Replace spaced conditional bar with \mid (keep spaces around)
        # Safe for probability conditionals like P( A | B ); won't affect |x| or ||x||
        eq = eq.replace(" | ", r" \mid ")

        return eq
    except Exception:
        return eq_text


def extract_equations(text, raw_result=None):
    """Extract equations from text content and images.
    Returns a dictionary of equations with their context and images."""
    # Match both inline math ($...$) and display math ($$...$$)
    inline_pattern = r'\$([^$]+)\$'
    display_pattern = r'\$\$([^$]+)\$\$'
    
    equations = {
        'inline': [],
        'display': [],
        'images': [],  # Will be populated with equation images later
        'equation_map': {}  # Map to link LaTeX equations with their corresponding images
    }
    
    # Extract display equations with context
    display_matches = re.finditer(display_pattern, text)
    for match in display_matches:
        # Get some context before and after the equation
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        
        # Parse the equation components
        eq_text_raw = match.group(1)
        eq_text = _sanitize_latex_equation(eq_text_raw)
        eq_parts = eq_text.split('=')
        
        # Create a more readable explanation
        if len(eq_parts) > 1:
            left_side = eq_parts[0].strip()
            right_side = '='.join(eq_parts[1:]).strip()
            
            # Extract variable names
            var_pattern = r'\\mathcal{([^}]+)}'
            variables = re.findall(var_pattern, eq_text)
            
            # Create explanation dictionary
            explanation = {
                'equation': eq_text,
                'left_side': left_side,
                'right_side': right_side,
                'variables': {},
                'natural_language': f"This equation defines {left_side} as {right_side}"
            }
            
            # Add variable explanations if available
            for var in variables:
                var_full = f'\\mathcal{{{var}}}'
                explanation['variables'][var_full] = f"Set or space {var}"
        else:
            explanation = {
                'equation': eq_text,
                'natural_language': "Mathematical expression or constraint"
            }
        
        equations['display'].append({
            'equation': eq_text,
            'equation_raw': eq_text_raw,
            'context': context,
            'original_text': match.group(0),
            'original_text_sanitized': f"$${eq_text}$$",
            'equation_id': f'eq_{len(equations["display"]) + 1}',  # Add unique ID
            'explanation': explanation
        })
    
    # Extract inline equations with context
    inline_matches = re.finditer(inline_pattern, text)
    for match in inline_matches:
        if not any(match.group(0) in eq['original_text'] for eq in equations['display']):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            eq_text_raw = match.group(1)
            eq_text = _sanitize_latex_equation(eq_text_raw)

            equations['inline'].append({
                'equation': eq_text,
                'equation_raw': eq_text_raw,
                'context': context,
                'original_text': match.group(0),
                'original_text_sanitized': f"${eq_text}$",
                'equation_id': f'inline_eq_{len(equations["inline"]) + 1}',  # Add unique ID
                'explanation': {
                    'equation': eq_text,
                    'natural_language': "Inline mathematical expression"
                }
            })
    
    return equations

def extract_text_from_pdf(paper_path, raw_result=None):
    """Extracts and preprocesses text content from a PDF.
    If raw_result is provided, reuse it to avoid re-conversion."""
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")

    # Convert PDF using docling if not provided
    if raw_result is None:
        raw_result = doc_converter.convert(paper_path)

    # Generate markdown content
    raw_markdown = raw_result.document.export_to_markdown()
    text_content = markdown_clean_pattern.sub("", raw_markdown)

    # Fallback to marker if docling fails
    if len(text_content) < 500:
        print('\nParsing with docling failed, using marker instead\n')
        parser_model = create_model_dict(device='cuda', dtype=torch.float16)
        text_content, _ = parse_pdf(paper_path, model_lst=parser_model, save_file=False)

    # Pre-process text content
    text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
    text_content = re.sub(r' +', ' ', text_content)
    text_content = re.sub(r'\n([A-Z][A-Za-z ]+)(?:\n|$)', r'\n## \1\n', text_content)
    text_content = re.sub(r'(Figure \d+)', r'**\1**', text_content)
    text_content = re.sub(r'(Table \d+)', r'**\1**', text_content)
    text_content = re.sub(r'([^\n])\n- ', r'\1\n\n- ', text_content)
    text_content = re.sub(r'(Equation \d+)', r'**\1**', text_content)
    
    return text_content, raw_result

@retry(stop=stop_after_attempt(5))
def parse_raw(args, actor_config, text_content, raw_result, version=1, avg_width=None, avg_height=None):
    """Parse raw content from PDF for presentation generation"""
    use_cache = getattr(args, 'use_cache', True)

    # Create output directories
    content_dir = f'contents'
    output_dir = Path(f'{args.model_name_t}_ppt_content/{args.paper_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = args.paper_name

    # Prepare equations cache path and load if available
    equations_path = f'results/{args.paper_name}/{content_dir}/{args.model_name_t}_{args.paper_name}_equations.json'
    Path(equations_path).parent.mkdir(parents=True, exist_ok=True)
    if use_cache and os.path.exists(equations_path):
        try:
            with open(equations_path, 'r') as f:
                equations = json.load(f)
        except Exception:
            equations = extract_equations(text_content)
    else:
        # Extract equations before preprocessing text
        equations = extract_equations(text_content)

    # Save markdown and HTML versions
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    raw_result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    raw_result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"

    
    # Load detailed content generation prompt
    with open("evopresent/ppt/prompts/talk_content_agent.txt", "r") as f:
        content_template_text = f.read()
        parts = content_template_text.split('\n\n')
        content_sys_msg = parts[0]
        content_template_content = '\n\n'.join(parts[1:])

    # Create template environment for detailed content
    content_jinja_env = Environment(undefined=StrictUndefined)
    content_template = content_jinja_env.from_string(content_template_content)

    # Create model and agent for detailed content generation
    if args.model_name_t.startswith('vllm_qwen'):
        content_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    else:
        content_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )

    content_agent = ChatAgent(
        system_message=content_sys_msg,
        model=content_model,
        message_window_size=10,
        token_limit=actor_config.get('token_limit', None)
    )

    content_max_retries = 3
    content_retry_count = 0
    content_last_error = None
    content_result = None
    feedback_prompt = ""
    
    while content_retry_count < content_max_retries:
        try:
            content_jinja_args = {
                'markdown_document': text_content,
                'total_slides': args.target_slides,
                'feedback': feedback_prompt
            }
            content_prompt = content_template.render(**content_jinja_args)
            
            content_agent.reset()
            content_response = content_agent.step(content_prompt)
            content_input_token, content_output_token = account_token(content_response)

            try:
                content_result = get_json_from_response(content_response.msgs[0].content)
            except Exception as e:
                print(f'Failed to parse content JSON response: {str(e)}')
                feedback_prompt = f"Your previous attempt failed with a JSON parsing error: {e}. Please ensure your output is valid JSON."
                content_retry_count += 1
                content_last_error = e
                continue

            if not content_result:
                print('Empty content response received')
                feedback_prompt = "Your previous attempt returned an empty response. Please generate the full JSON content as requested."
                content_retry_count += 1
                content_last_error = ValueError("Empty content response")
                continue

            if isinstance(content_result, dict) and 'slides' in content_result:
                # Verify slide count matches target
                actual_slides = len(content_result['slides'])
                if actual_slides != args.target_slides:
                    print(f'Invalid slide count: got {actual_slides}, expected {args.target_slides}')
                    feedback_prompt = f"Your previous attempt was unsuccessful. You generated {actual_slides} slides, but the required number is {args.target_slides}. Please adhere strictly to the 'TOTAL_SLIDES' instruction and generate exactly {args.target_slides} slides."
                    content_retry_count += 1
                    content_last_error = ValueError(f"Slide count mismatch: {actual_slides} != {args.target_slides}")
                    continue
                # Enforce conclusion placement: no core content after conclusion/future work
                try:
                    titles = [ (s.get('slide_type'), (s.get('slide_title') or '').lower()) for s in content_result['slides'] ]
                    last_conclusion_idx = max([i for i, (_, t) in enumerate(titles) if any(k in t for k in ['conclusion', 'future work', 'future directions'])], default=-1)
                    if last_conclusion_idx != -1 and last_conclusion_idx < len(titles) - 1:
                        # Check for core content after conclusion (content slides with non-closing titles)
                        tail_core = False
                        for s in content_result['slides'][last_conclusion_idx+1:]:
                            stype = s.get('slide_type')
                            title_l = (s.get('slide_title') or '').lower()
                            is_functional_tail = stype == 'functional' or any(k in title_l for k in ['acknowledg', 'thanks', 'thank you', 'q&a', 'contact', 'resource', 'arxiv', 'code', 'dataset'])
                            if stype == 'content' and not is_functional_tail:
                                tail_core = True
                                break
                        if tail_core:
                            print('Core content detected after Conclusion/Future Work. Requesting regeneration with strict ordering.')
                            feedback_prompt = (
                                "Your previous attempt placed core content after the Conclusion/Future Work section. "
                                "Please ensure the storyline maintains forward progression and that Conclusion/Future Work appear only at the end. "
                                "If you are under the target slide count, expand earlier Tier 1 content (methods/results) instead of appending core content after the conclusion."
                            )
                            content_retry_count += 1
                            content_last_error = ValueError('Core content after conclusion detected')
                            continue
                except Exception as _:
                    pass
                break

            print(f'Retry {content_retry_count + 1}: Invalid content response structure')
            feedback_prompt = "Your previous response had an invalid structure. Please ensure it contains a 'slides' key with an array of slide objects."
            content_retry_count += 1
            content_last_error = ValueError("Invalid content response structure")

        except Exception as e:
            print(f'Error on content attempt {content_retry_count + 1}: {str(e)}')
            feedback_prompt = f"Your previous attempt failed with an exception: {e}. Please try again."
            content_retry_count += 1
            content_last_error = e
            continue

    if content_retry_count == content_max_retries:
        error_msg = f"Failed to generate valid content after {content_max_retries} attempts. Last error: {str(content_last_error)}"
        print(error_msg)
        raise ValueError(error_msg)

    # Clean captions at planning stage: remove leading Figure/Table prefixes
    try:
        slides = (content_result or {}).get('slides') or []
        for slide in slides:
            content = slide.get('content') or {}
            visuals = content.get('visual_elements') or []
            for v in visuals:
                cap = v.get('caption', '')
                if not isinstance(cap, str):
                    continue
                # Remove prefixes like "Figure 5:", "Fig. 5:", "Table 3:" at start
                new_cap = re.sub(r"^(?i)(figure|fig\.)\s*[-.]?\s*\d+\s*[:.\-\s]+", "", cap.strip())
                new_cap = re.sub(r"^(?i)(table)\s*[-.]?\s*\d+\s*[:.\-\s]+", "", new_cap)
                v['caption'] = new_cap.strip()
    except Exception:
        pass

    # Save results
    os.makedirs(f'results/{content_dir}', exist_ok=True)
    
    # Save equations
    with open(equations_path, 'w') as f:
        json.dump(equations, f, indent=2)
    
    # Add equations to content result
    content_result['equations'] = equations
    
    # Save HTML with updated content including generated images
    raw_result.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
    
    print(f"Equations saved to: {equations_path}")
    print(f"Presentation content will be saved by ppt_gen_pipeline.py")

    return content_input_token, content_output_token, content_result


def gen_image_and_table(args, vision_config, raw_result=None):
    """Generate and process images, tables and equations from the paper using vision model.
    Adds disk caching and parallelization for faster processing."""
    use_cache = getattr(args, 'use_cache', True)
    force_refresh = getattr(args, 'force_refresh', False)
    max_workers = int(getattr(args, 'max_workers', max(2, min(8, (os.cpu_count() or 4)))))

    # Output and cache paths
    output_dir = Path(f'{args.model_name_t}_images_and_tables/{args.paper_name}')
    images_json_path = Path(f'{args.model_name_t}_images_and_tables/{args.paper_name}_images.json')
    tables_json_path = Path(f'{args.model_name_t}_images_and_tables/{args.paper_name}_tables.json')
    equations_json_path = Path(f'{args.model_name_t}_images_and_tables/{args.paper_name}_equations.json')

    output_dir.mkdir(parents=True, exist_ok=True)
    images_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Early return from cache
    if use_cache and (images_json_path.exists() and tables_json_path.exists() and equations_json_path.exists()) and not force_refresh:
        try:
            with open(images_json_path, 'r') as f:
                images = json.load(f)
            with open(tables_json_path, 'r') as f:
                tables = json.load(f)
            with open(equations_json_path, 'r') as f:
                equations = json.load(f)
            return 0, 0, images, tables, equations, raw_result
        except Exception:
            pass

    # Ensure we have raw_result for fresh extraction
    if raw_result is None:
        raw_source = args.paper_path
        raw_result = doc_converter.convert(raw_source)

    input_token, output_token = 0, 0

    # Containers
    tables = {}
    images = {}
    equations = {'images': [], 'equation_map': {}}

    # Crop mode configuration (controls how broadly we capture context around figures)
    crop_mode = getattr(args, 'image_crop_mode', 'expanded')  # 'expanded' (default) or 'full_asset'
    pad_x_ratio_default = float(getattr(args, 'crop_pad_x_ratio', 0.06))
    pad_y_ratio_default = float(getattr(args, 'crop_pad_y_ratio', 0.08))

    # Helper to normalize caption text for grouping
    def _normalize_caption(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()

    # TABLES: prepare tasks with stable indices
    tables_with_caption = []
    for t in raw_result.document.tables:
        try:
            c = t.caption_text(raw_result.document)
        except Exception:
            c = ""
        if len(c) > 0:
            tables_with_caption.append((t, c))
    table_tasks = [(idx + 1, t, c) for idx, (t, c) in enumerate(tables_with_caption)]

    # If user wants to preserve the entire PNG asset as exported in markdown, build images from assets
    pictures_built_by_assets = False
    if crop_mode == 'full_asset':
        try:
            asset_md_stem = f"{args.paper_name}-assets"
            asset_md = output_dir / f"{asset_md_stem}.md"
            raw_result.document.save_as_markdown(asset_md, image_mode=ImageRefMode.REFERENCED)
            assets_dir = output_dir / f"{asset_md_stem}_artifacts"
            asset_files = []
            if os.path.isdir(assets_dir):
                asset_files = sorted([f for f in os.listdir(assets_dir) if f.lower().endswith('.png')])

            pic_idx = 1
            for pic in raw_result.document.pictures:
                try:
                    caption = pic.caption_text(raw_result.document)
                except Exception:
                    caption = ""
                if len(caption) == 0:
                    continue

                # Pick the corresponding asset by order; fallback to pic image if missing
                img_path = None
                if pic_idx - 1 < len(asset_files):
                    img_path = assets_dir / asset_files[pic_idx - 1]

                try:
                    if img_path is not None and os.path.isfile(img_path):
                        src_img = PIL.Image.open(img_path)
                    else:
                        src_img = pic.get_image(raw_result.document)
                except Exception:
                    # Skip if we cannot obtain image
                    pic_idx += 1
                    continue

                max_width = int(args.slide_width_inches * 96 * 0.85)
                max_height = int(args.slide_height_inches * 96 * 0.6)
                width, height = src_img.size
                scale = min(max_width / width, max_height / height)
                new_size = (int(width * scale), int(height * scale))
                out_img = src_img.resize(new_size, PIL.Image.Resampling.LANCZOS)
                # Add small white border to avoid edge clipping
                border_px = max(2, int(min(new_size) * 0.01))
                out_img = ImageOps.expand(out_img, border=border_px, fill=(255, 255, 255))
                out_path = output_dir / f"{args.paper_name}-picture-{pic_idx}.png"
                out_img.save(out_path, format="PNG")

                images[f"Figure {pic_idx}"] = {
                    'caption': caption,
                    'image_path': str(out_path),
                    'width': new_size[0],
                    'height': new_size[1],
                    'original_size': (width, height),
                    'scale_factor': scale
                }
                pic_idx += 1

            pictures_built_by_assets = True
        except Exception:
            pictures_built_by_assets = False

    def _process_table(idx, table, caption):
        try:
            table_img_path = output_dir / f"{args.paper_name}-table-{idx}.png"
            table_img = table.get_image(raw_result.document)
            max_width = int(args.slide_width_inches * 96)
            max_height = int(args.slide_height_inches * 96 * 0.8)
            width, height = table_img.size
            scale = min(max_width / width, max_height / height)
            new_size = (int(width * scale), int(height * scale))
            table_img = table_img.resize(new_size, PIL.Image.LANCZOS)
            # Add small white border to avoid edge clipping
            border_px = max(2, int(min(new_size) * 0.01))
            table_img = ImageOps.expand(table_img, border=border_px, fill=(255, 255, 255))
            table_img.save(table_img_path, format="PNG")
            return idx, {
                'caption': caption,
                'table_path': str(table_img_path),
                'width': new_size[0],
                'height': new_size[1],
                'original_size': (width, height),
                'scale_factor': scale
            }
        except Exception:
            return idx, None

    # IMAGES: group by caption, assign uncaptioned, then build union and single tasks
    grouped_by_caption = defaultdict(list)
    uncaptioned_pictures = []
    for pic in raw_result.document.pictures:
        try:
            caption_text = pic.caption_text(raw_result.document)
        except Exception:
            caption_text = ""
        if len(caption_text) == 0:
            uncaptioned_pictures.append(pic)
            continue
        norm_caption = _normalize_caption(caption_text)
        fig_match = re.search(r'(?:figure|fig\.?)\s*(\d+)', norm_caption)
        if fig_match:
            fig_key = f"fig_{fig_match.group(1)}"
        else:
            fig_key = f"cap_{hashlib.md5(norm_caption.encode('utf-8')).hexdigest()[:10]}"
        grouped_by_caption[fig_key].append(pic)

    def _rect_distance(a, b):
        dx = 0.0
        if a.r < b.l:
            dx = b.l - a.r
        elif b.r < a.l:
            dx = a.l - b.r
        dy = 0.0
        if a.t < b.b:
            dy = b.b - a.t
        elif b.t < a.b:
            dy = a.b - b.t
        return (dx**2 + dy**2) ** 0.5

    if uncaptioned_pictures:
        group_meta = {}
        for fig_key, items in grouped_by_caption.items():
            try:
                bboxes = [it.prov[0].bbox for it in items if len(it.prov) > 0]
                if not bboxes:
                    continue
                l = min(bb.l for bb in bboxes)
                r = max(bb.r for bb in bboxes)
                bb = min(bb.b for bb in bboxes)
                tt = max(bb.t for bb in bboxes)
                union_bbox = BoundingBox(l=l, t=tt, r=r, b=bb, coord_origin=bboxes[0].coord_origin)
                try:
                    page_ix = items[0].prov[0].page_no - 1
                except Exception:
                    try:
                        page_ix = next((p.page_no - 1 for p in items[0].prov if getattr(p, 'page_no', None) is not None), None)
                    except Exception:
                        page_ix = None
                group_meta[fig_key] = {"bbox": union_bbox, "page_ix": page_ix}
            except Exception:
                continue

        for pic in uncaptioned_pictures:
            try:
                if len(pic.prov) == 0:
                    continue
                pic_bbox = pic.prov[0].bbox
                try:
                    pic_page_ix = pic.prov[0].page_no - 1
                except Exception:
                    try:
                        pic_page_ix = next((p.page_no - 1 for p in pic.prov if getattr(p, 'page_no', None) is not None), None)
                    except Exception:
                        pic_page_ix = None

                best_key = None
                best_dist = None
                for gk, meta in group_meta.items():
                    if meta.get("page_ix") is not None and pic_page_ix is not None and meta["page_ix"] != pic_page_ix:
                        continue
                    dist = _rect_distance(meta["bbox"], pic_bbox)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_key = gk
                if best_key is not None and (best_dist is None or best_dist < 50):
                    grouped_by_caption[best_key].append(pic)
            except Exception:
                continue

    processed_ids = set()
    union_tasks = []
    for fig_key, items in grouped_by_caption.items():
        if len(items) <= 1:
            continue
        try:
            bboxes = [it.prov[0].bbox for it in items if len(it.prov) > 0]
            l = min(bb.l for bb in bboxes)
            r = max(bb.r for bb in bboxes)
            b = min(bb.b for bb in bboxes)
            t = max(bb.t for bb in bboxes)
            width = r - l
            height = t - b
            pad_x = width * 0.06
            pad_y = height * 0.08
            raw_union_bbox = BoundingBox(l=l - pad_x, t=t + pad_y, r=r + pad_x, b=b - pad_y, coord_origin=bboxes[0].coord_origin)
            try:
                page_ix = items[0].prov[0].page_no - 1
            except Exception:
                try:
                    page_ix = next((p.page_no - 1 for p in items[0].prov if getattr(p, 'page_no', None) is not None), None)
                except Exception:
                    page_ix = None
            # Clamp bbox to page bounds
            if page_ix is not None and 0 <= page_ix < len(raw_result.pages):
                page_obj = raw_result.pages[page_ix]
                page_w = (page_obj.size or {}).width if hasattr(page_obj.size, 'width') else None
                page_h = (page_obj.size or {}).height if hasattr(page_obj.size, 'height') else None
                if page_w is not None and page_h is not None:
                    union_bbox = _expand_and_clamp_bbox(raw_union_bbox, page_w, page_h)
                else:
                    union_bbox = raw_union_bbox
            else:
                union_bbox = raw_union_bbox

            union_tasks.append({
                'items': items,
                'union_bbox': union_bbox,
                'page_ix': page_ix
            })
            for it in items:
                processed_ids.add(id(it))
        except Exception:
            continue

    # Singles: pictures not processed in unions
    single_pics = []
    for pic in raw_result.document.pictures:
        if id(pic) in processed_ids:
            continue
        try:
            caption = pic.caption_text(raw_result.document)
        except Exception:
            caption = ""
        if len(caption) > 0:
            is_equation = any(keyword in caption.lower() for keyword in ['equation', 'formula', 'eq.', 'eqn'])
            single_pics.append({'pic': pic, 'caption': caption, 'is_equation': is_equation})

    # Assign stable indices
    union_count = len(union_tasks)
    non_equation_tasks = [t for t in single_pics if not t['is_equation']]
    equation_tasks = [t for t in single_pics if t['is_equation']]
    for idx, task in enumerate(union_tasks, start=1):
        task['img_idx'] = idx
    for offset, task in enumerate(non_equation_tasks, start=1):
        task['img_idx'] = union_count + offset
    for eq_idx, task in enumerate(equation_tasks, start=1):
        task['eq_idx'] = eq_idx

    def _process_union_task(task):
        try:
            page = raw_result.pages[task['page_ix']] if task['page_ix'] is not None else None
            if page is None:
                return None
            merged_img = page.get_image(scale=IMAGE_RESOLUTION_SCALE, cropbox=task['union_bbox'])
            max_width = int(args.slide_width_inches * 96 * 0.85)
            max_height = int(args.slide_height_inches * 96 * 0.6)
            w, h = merged_img.size
            scale = min(max_width / w, max_height / h)
            new_size = (int(w * scale), int(h * scale))
            merged_img = merged_img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            # Add small white border to avoid edge clipping
            border_px = max(2, int(min(new_size) * 0.01))
            merged_img = ImageOps.expand(merged_img, border=border_px, fill=(255, 255, 255))
            out_name = f"{args.paper_name}-picture-{task['img_idx']}.png"
            merged_path = output_dir / out_name
            merged_img.save(merged_path, format="PNG")
            caption_text = task['items'][0].caption_text(raw_result.document)
            return task['img_idx'], {
                'caption': caption_text,
                'image_path': str(merged_path),
                'width': new_size[0],
                'height': new_size[1],
                'original_size': (w, h),
                'scale_factor': scale
            }
        except Exception:
            return None

    def _process_single_image_task(task):
        try:
            pic = task['pic']
            caption = task['caption']
            # Prefer page-based crop with expanded bbox (to include legends/labels)
            image_img = None
            if len(getattr(pic, 'prov', [])) > 0:
                pic_bbox = pic.prov[0].bbox
                try:
                    page_ix = pic.prov[0].page_no - 1
                except Exception:
                    page_ix = None
                if page_ix is not None and 0 <= page_ix < len(raw_result.pages):
                    page_obj = raw_result.pages[page_ix]
                    page_w = (page_obj.size or {}).width if hasattr(page_obj.size, 'width') else None
                    page_h = (page_obj.size or {}).height if hasattr(page_obj.size, 'height') else None
                    if page_w is not None and page_h is not None:
                        crop_bbox = _expand_and_clamp_bbox(pic_bbox, page_w, page_h, pad_x_ratio=0.06, pad_y_ratio=0.08)
                        image_img = page_obj.get_image(scale=IMAGE_RESOLUTION_SCALE, cropbox=crop_bbox)
            if image_img is None:
                image_img = pic.get_image(raw_result.document)

            max_width = int(args.slide_width_inches * 96 * 0.85)
            max_height = int(args.slide_height_inches * 96 * 0.6)
            width, height = image_img.size
            scale = min(max_width / width, max_height / height)
            min_dimension = 200
            if width * scale < min_dimension or height * scale < min_dimension:
                scale = max(min_dimension / width, min_dimension / height)
            new_size = (int(width * scale), int(height * scale))
            image_img = image_img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            # Add small white border to avoid edge clipping
            border_px = max(2, int(min(new_size) * 0.01))
            image_img = ImageOps.expand(image_img, border=border_px, fill=(255, 255, 255))
            image_img_path = output_dir / f"{args.paper_name}-picture-{task['img_idx']}.png"
            image_img.save(image_img_path, format="PNG")
            return task['img_idx'], {
                'caption': caption,
                'image_path': str(image_img_path),
                'width': new_size[0],
                'height': new_size[1],
                'original_size': (width, height),
                'scale_factor': scale
            }
        except Exception:
            return None

    def _process_equation_task(task):
        try:
            pic = task['pic']
            caption = task['caption']
            eq_img = pic.get_image(raw_result.document)
            max_width = int(args.slide_width_inches * 96 * 0.6)
            max_height = int(args.slide_height_inches * 96 * 0.4)
            width, height = eq_img.size
            scale = min(max_width / width, max_height / height)
            new_size = (int(width * scale), int(height * scale))
            eq_img = eq_img.resize(new_size, PIL.Image.LANCZOS)
            # Add small white border
            border_px = max(2, int(min(new_size) * 0.01))
            eq_img = ImageOps.expand(eq_img, border=border_px, fill=(255, 255, 255))
            eq_img_path = output_dir / f"{args.paper_name}-equation-{task['eq_idx']}.png"
            eq_img.save(eq_img_path, format="PNG")
            eq_num_match = re.search(r'equation\s*(\d+)', caption.lower())
            eq_id = f"eq_{eq_num_match.group(1)}" if eq_num_match else f"eq_img_{task['eq_idx']}"
            eq_data = {
                'caption': caption,
                'equation_path': str(eq_img_path),
                'width': new_size[0],
                'height': new_size[1],
                'original_size': (width, height),
                'scale_factor': scale,
                'equation_id': eq_id
            }
            return eq_id, eq_data
        except Exception:
            return None

    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Tables
        future_tables = {executor.submit(_process_table, idx, t, c): idx for idx, t, c in table_tasks}
        for future in as_completed(future_tables):
            idx, data = future.result()
            if data is not None:
                tables[str(idx)] = data

        # Only run union/single extraction if we did not already build pictures from full assets
        if not pictures_built_by_assets:
            # Union images
            future_unions = {executor.submit(_process_union_task, task): task['img_idx'] for task in union_tasks}
            for future in as_completed(future_unions):
                res = future.result()
                if res is None:
                    continue
                img_idx, data = res
                images[f"Figure {img_idx}"] = data

            # Single images (non-equations)
            future_images = {executor.submit(_process_single_image_task, task): task['img_idx'] for task in non_equation_tasks}
            for future in as_completed(future_images):
                res = future.result()
                if res is None:
                    continue
                img_idx, data = res
                images[f"Figure {img_idx}"] = data

        # Equation images
        future_equations = {executor.submit(_process_equation_task, task): task.get('eq_idx') for task in equation_tasks}
        for future in as_completed(future_equations):
            res = future.result()
            if res is None:
                continue
            eq_id, eq_data = res
            equations['images'].append(eq_data)
            equations['equation_map'][eq_id] = eq_data

    # Persist JSON caches
    with open(images_json_path, 'w') as f:
        json.dump(images, f, indent=4)
    with open(tables_json_path, 'w') as f:
        json.dump(tables, f, indent=4)
    with open(equations_json_path, 'w') as f:
        json.dump(equations, f, indent=4)

    return input_token, output_token, images, tables, equations, raw_result
