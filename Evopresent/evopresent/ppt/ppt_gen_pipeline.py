import argparse
import time
import os
import json
import re
from pathlib import Path
from utils.wei_utils import get_agent_config
from evopresent.ppt.parse_raw import parse_raw, gen_image_and_table, extract_text_from_pdf
from evopresent.ppt.gen_speech import generate_speech_script
from evopresent.ppt.gen_pptx_code import generate_presentation, configure_clients
from camel.models import ModelFactory
from evopresent.ppt.scholar_agent import ScholarAgent
from evopresent.ppt.visualization_helper import maybe_generate_visual_for_slide

def get_project_root():
    """Get the project root directory"""
    # Try to get from environment variable first
    root_dir = os.getenv('EVOPRESENT_ROOT')
    if root_dir and os.path.exists(root_dir):
        return root_dir

    # Otherwise, use the current working directory
    current_dir = os.getcwd()

    # Look for common project markers
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'evopresent')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    # If no markers found, use current directory
    return os.getcwd()


def create_directories(paper_name):
    """Create necessary directories if they don't exist"""
    # Get project root directory
    base_dir = os.path.join(get_project_root(), 'results', paper_name)

    directories = [
        'contents',
        'generated_ppts',
    ]

    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
    return base_dir


def _strip_leading_figure_table_prefix(text: str) -> str:
    try:
        if not text:
            return ""
        t = text.strip()
        # Remove leading "Figure 5:", "Fig. 5:", "Table 3:" (various punctuation)
        t = re.sub(r"^(?i)(figure|fig\.)\s*[-.]?\s*\d+\s*[:.\-\s]+", "", t)
        t = re.sub(r"^(?i)(table)\s*[-.]?\s*\d+\s*[:.\-\s]+", "", t)
        return t.strip()
    except Exception:
        return text or ""


def _norm_caption(text: str) -> str:
    try:
        if text is None:
            return ""
        t = _strip_leading_figure_table_prefix(text)
        t = t.lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^a-z0-9\s]", "", t)
        return t.strip()
    except Exception:
        return ""


def _build_extracted_indices(images: dict, tables: dict):
    """Build indices from extracted resources for fast lookup by caption/id."""
    id_to_item = {}
    cap_to_items = {}

    # Figures
    for fig_id, data in (images or {}).items():
        cap = data.get('caption', '')
        path = data.get('image_path') or data.get('path')
        if not path:
            continue
        item = {'id': fig_id, 'path': path, 'caption': cap, 'type': 'figure'}
        id_to_item[fig_id] = item
        norm = _norm_caption(cap)
        if norm:
            cap_to_items.setdefault(norm, []).append(item)

    # Tables
    for idx_str, data in (tables or {}).items():
        cap = data.get('caption', '')
        path = data.get('table_path') or data.get('path')
        if not path:
            continue
        table_id = f"Table {idx_str}"
        item = {'id': table_id, 'path': path, 'caption': cap, 'type': 'table'}
        id_to_item[table_id] = item
        norm = _norm_caption(cap)
        if norm:
            cap_to_items.setdefault(norm, []).append(item)

    return id_to_item, cap_to_items


def reconcile_visual_ids_with_extracted(presentation_content: dict, images: dict, tables: dict):
    """Rewrite each visual element to use the extracted ID, not the caption's Figure X.

    Logic per visual element:
      - Clean caption (remove leading Figure/Table X) and normalize.
      - If this normalized caption uniquely matches an extracted item (same type if provided),
        set v['id'] to that item's id, and set v['local_path'] to its path.
      - Always clean v['caption'] to remove leading Figure/Table X to avoid confusion.
      - If unique caption match not found, and v['id'] like Figure/Table N appears inside
        some extracted caption, map to that extracted ID.
    """
    try:
        if not presentation_content:
            return presentation_content

        id_to_item, cap_to_items = _build_extracted_indices(images, tables)
        slides = presentation_content.get('slides') or []

        for slide in slides:
            content = slide.get('content') or {}
            visuals = content.get('visual_elements') or []
            for v in visuals:
                # Clean up caption text
                original_caption = v.get('caption', '')
                cleaned_caption = _strip_leading_figure_table_prefix(original_caption)
                v['caption'] = cleaned_caption

                # Skip generated visuals that already have a path
                if v.get('local_path'):
                    continue

                desired_type = (v.get('type') or '').lower() or None
                norm = _norm_caption(original_caption)
                chosen = None

                # 1) Try exact normalized caption match
                candidates = cap_to_items.get(norm, []) if norm else []
                if desired_type:
                    candidates = [c for c in candidates if c.get('type') == desired_type] or candidates
                if len(candidates) == 1:
                    chosen = candidates[0]

                # 2) Fallback: use numeric from v['id'] to find extracted caption containing it
                if chosen is None:
                    vid = v.get('id') or ''
                    m = re.search(r"(?i)\b(figure|fig\.|table)\s*(\d+)\b", vid)
                    if m:
                        label = m.group(1).lower()
                        num = m.group(2)
                        target_phrase = f"{('figure' if 'fig' in label or 'figure' in label else 'table')} {num}"
                        # search across relevant pool
                        pool = images.items() if 'figure' in target_phrase else tables.items()
                        for ext_id, data in pool:
                            cap = (data or {}).get('caption', '')
                            if re.search(rf"(?i)\b{re.escape(target_phrase)}\b", cap):
                                path = data.get('image_path') or data.get('table_path') or data.get('path')
                                if path:
                                    chosen = {'id': ext_id if 'figure' not in target_phrase else ext_id,
                                              'path': path,
                                              'caption': cap,
                                              'type': 'table' if 'table' in target_phrase else 'figure'}
                                    break

                if chosen is not None:
                    v['id'] = chosen.get('id')
                    v['local_path'] = chosen.get('path')

        return presentation_content
    except Exception as e:
        print(f"Warning: reconcile_visual_ids_with_extracted failed: {e}")
        return presentation_content


def main(args):
    start_time = time.time()

    # Create necessary directories with paper name
    if args.paper_name is None:
        args.paper_name = args.paper_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')

    base_dir = create_directories(args.paper_name)

    # Get model configurations
    actor_config_t = get_agent_config(args.model_name_t)
    actor_config_v = get_agent_config(args.model_name_v)

    # Create model instances
    model_t = ModelFactory.create(
        model_platform=actor_config_t['model_platform'],
        model_type=actor_config_t['model_type'],
        model_config_dict=actor_config_t['model_config']
    )
    model_v = ModelFactory.create(
        model_platform=actor_config_v['model_platform'],
        model_type=actor_config_v['model_type'],
        model_config_dict=actor_config_v['model_config']
    )

    # Add models to configs
    actor_config_t['model'] = model_t
    actor_config_v['model'] = model_v

    # Initialize content holders
    text_content, raw_result = None, None

    # Generate talk outline
    outline_file = os.path.join(base_dir, 'contents', f'{args.model_name_t}_{args.paper_name}_outline.json')
    content_file = os.path.join(base_dir, 'contents', f'{args.model_name_t}_{args.paper_name}_presentation.json')

    # Generate images and tables
    images_dir = os.path.join(get_project_root(), f'{args.model_name_t}_images_and_tables')
    os.makedirs(images_dir, exist_ok=True)
    images_file = os.path.join(images_dir, f'{args.model_name_t}_{args.paper_name}_images.json')
    tables_file = os.path.join(images_dir, f'{args.model_name_t}_{args.paper_name}_tables.json')

    if os.path.exists(images_file) and os.path.exists(tables_file) and not args.force_refresh:
        print('Using existing images and tables...')
        with open(images_file, 'r') as f:
            images = json.load(f)
        with open(tables_file, 'r') as f:
            tables = json.load(f)
    else:
        print('=============Extracting images and tables================')
        _, _, images, tables, equations, raw_result = gen_image_and_table(args, actor_config_v)

    if os.path.exists(content_file) and not args.force_refresh:
        print('Using existing presentation content...')
        with open(content_file, 'r') as f:
            presentation_content = json.load(f)
        input_token, output_token = 0, 0
    else:
        print('=============Generating presentation content================')
        if raw_result is None:
            print("Parsing PDF to generate content...")
            text_content, raw_result = extract_text_from_pdf(args.paper_path)
        else:
            print("Extracting text from parsed PDF...")
            text_content, _ = extract_text_from_pdf(args.paper_path)  # Simplified to always extract text if content needs generating
        input_token, output_token, presentation_content = parse_raw(args, actor_config_t, text_content, raw_result)

    # Align slide visuals to extracted IDs and clean captions before enrichment
    try:
        presentation_content = reconcile_visual_ids_with_extracted(presentation_content, images, tables)
    except Exception:
        pass

    # Save original presentation content before scholar enrichment (unified location)
    original_outline_file = os.path.join(base_dir, 'contents', f'{args.model_name_t}_{args.paper_name}_original_presentation.json')
    with open(original_outline_file, 'w') as f:
        json.dump(presentation_content, f, indent=2)
    print(f"Original presentation content saved to: {original_outline_file}")

    # Scholar enrichment (toggleable)
    if (getattr(args, 'scholar', 'on') or 'on').lower() == 'on':
        try:
            print('\n=================Step 4: Scholar Enrichment===================')
            scholar = ScholarAgent(actor_config_t)

            # Enrichment based on scholar_request
            # Prepare image save dir
            image_save_dir = Path(f'{args.model_name_t}_images_and_tables/{args.paper_name}')
            image_save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {len(presentation_content.get('slides', []))} slides for enrichment...")

            for i, slide in enumerate(presentation_content.get('slides', [])):
                # Ensure each slide has a stable numeric slide_id for downstream unique filenames
                try:
                    if slide.get('slide_id') is None:
                        slide['slide_id'] = i + 1
                except Exception:
                    slide['slide_id'] = i + 1
                slide_title = slide.get('slide_title', f'Slide {i+1}')
                content = slide.get('content') or {}
                req = content.get('scholar_request') or {}
                rtype = (req.get('type') or 'none').lower()
                reason = (req.get('reason') or '').strip()

                print(f"\n--- Processing slide {i+1}: {slide_title} ---")
                print(f"Scholar request type: {rtype}")
                if reason:
                    print(f"Reason: {reason}")

                if rtype == 'knowledge':
                    print("Applying knowledge enrichment...")
                    updated, source = scholar.enrich_slide_knowledge(paper_markdown=text_content, slide=slide)
                    slide.update(updated)
                    print(f"Knowledge enrichment completed from {source}")
                elif rtype == 'image':
                    print("Applying image enrichment...")
                    # Use the enhanced decide_and_apply_enrichment method
                    updated = scholar.decide_and_apply_enrichment(
                        paper_markdown=text_content,
                        slide=slide,
                        save_dir=image_save_dir
                    )
                    slide.update(updated)
                    print("Image enrichment completed")
                else:
                    print("No enrichment requested")

        except Exception as e:
            print(f"Scholar enrichment step failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print('\n=================Step 4: Scholar Enrichment (disabled)===================')
        print('Skipping ScholarAgent enrichment because --scholar is off.')

    total_input_token = input_token
    total_output_token = output_token

    # Save enhanced presentation content after scholar enrichment
    temp_outline_file = os.path.join(base_dir, 'contents', f'{args.model_name_t}_{args.paper_name}_presentation.json')
    with open(temp_outline_file, 'w') as f:
        json.dump(presentation_content, f, indent=2)
    print(f"Enhanced presentation content saved to: {temp_outline_file}")

    # Save a separate JSON mapping slide_id -> script (after enrichment)
    try:
        scripts_map = []
        for idx, slide in enumerate(presentation_content.get('slides', []), start=1):
            slide_id = slide.get('slide_id')
            if slide_id is None:
                slide_id = idx  # stable fallback to ordinal index
            content = slide.get('content') or {}
            script = (content.get('script') or '').strip()
            scripts_map.append({
                'id': slide_id,
                'script': script,
                'title': slide.get('slide_title', '')
            })

        script_json_path = os.path.join(base_dir, 'contents', f'{args.model_name_t}_{args.paper_name}_scripts.json')
        with open(script_json_path, 'w') as f:
            json.dump({'slides': scripts_map}, f, indent=2, ensure_ascii=False)
        print(f"Per-slide scripts saved to: {script_json_path}")
    except Exception as e:
        print(f"Failed to save per-slide scripts JSON: {e}")

    print('=============Generating final presentation===============')
    output_dir = os.path.join(base_dir, 'generated_presentations')
    os.makedirs(output_dir, exist_ok=True)

    # Configure generator/checker model clients from CLI
    try:
        configure_clients(getattr(args, 'html_model', os.getenv('EVOP_HTML_MODEL', 'deepseek')),
                          getattr(args, 'checker_model', os.getenv('EVOP_CHECKER_MODEL', 'gpt')))
    except Exception as e:
        print(f"Warning: configure_clients failed, will rely on in-function fallback: {e}")

    # Generate HTML presentation
    output_path = os.path.join(output_dir, f'{args.paper_name}_presentation.html')

    success = generate_presentation(
        outline_path=temp_outline_file,
        output_path=output_path,
        base_dir=base_dir,
        paper_name=args.paper_name,
        model_name_t=args.model_name_t,
        style=args.style,
        use_checker=(args.checker == 'on'),
        checker_scope=getattr(args, 'checker_scope', 'all'),
        checker_threshold=getattr(args, 'checker_threshold', 8.7),
        checker_max_attempts=getattr(args, 'checker_max_attempts', 3),
    )
    if not success:
        print("Failed to generate presentation.")
        return

    # Calculate execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Save log
    log_path = os.path.join(base_dir, 'log.json')
    with open(log_path, 'w') as f:
        log_data = {
            'total_input_token': total_input_token,
            'total_output_token': total_output_token,
            'execution_time': f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}",
            'used_cache': not args.force_refresh and os.path.exists(content_file) and os.path.exists(images_file),
            'model_t': args.model_name_t,
            'model_v': args.model_name_v
        }
        json.dump(log_data, f, indent=4)

    print(f'\nTotal token consumption: {total_input_token} -> {total_output_token}')
    print(f'Execution Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}')
    print(f'Final presentation saved to: {output_path}')


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('yes', 'true', 't', '1', 'on')

    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_path', type=str, required=True, help='Path to the paper PDF file')
    parser.add_argument('--paper_name', type=str, required=True, help='Name of the paper')
    parser.add_argument('--target_slides', type=int, required=True, help='Target number of slides to generate')
    parser.add_argument('--force_refresh', action='store_true', help='Force refresh all steps without using cache')
    parser.add_argument('--model_name_t', type=str, default='4o', help='Text model name')
    parser.add_argument('--model_name_v', type=str, default='4o', help='Vision model name')
    parser.add_argument('--template_path', type=str, default=None)
    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--slide_width_inches', type=float, default=20.0)
    parser.add_argument('--slide_height_inches', type=float, default=11.25)
    parser.add_argument('--style', type=str, default=None, help='Style to use (modern/modern_blue/minimalist/tech_dark/academic/vibrant)')
    parser.add_argument('--checker', choices=['on', 'off'], default='on', help='Enable or disable layout checker loop')
    parser.add_argument('--scholar', choices=['on', 'off'], default='on', help='Enable or disable ScholarAgent enrichment')
    parser.add_argument('--checker-scope', choices=['all', 'images', 'text', 'none'], default='all',
                        help='Which slides to run checker on: all, only images, only text-only, or none')
    parser.add_argument('--use_cache', type=str2bool, nargs='?', const=True, default=True,
                        help='Use disk cache for extraction (true/false). Default: true')
    parser.add_argument('--max_workers', type=int, default=min(6, max(2, (os.cpu_count() or 4))),
                        help='Max threads for parallel extraction. Default: min(8, max(2, CPU count))')
    parser.add_argument('--html-model', default=os.getenv('EVOP_HTML_MODEL', 'deepseek'),
                        help='Generator: provider name (deepseek/gpt/gemini/claude) or explicit model id')
    parser.add_argument('--checker-model', default=os.getenv('EVOP_CHECKER_MODEL', 'gpt'),
                        help='Checker: provider name (deepseek/gpt/gemini/claude) or explicit model id')
    parser.add_argument('--checker-threshold', type=float, default=8.7,
                        help='Aesthetic score threshold to accept a layout (default: 8.7)')
    parser.add_argument('--checker-max-attempts', type=int, default=3,
                        help='Max attempts for checker refinement loop (default: 3)')
    args = parser.parse_args()

    main(args)
