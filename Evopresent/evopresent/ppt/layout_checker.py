#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, Optional
import base64
import tempfile
import shutil
import time

from openai import OpenAI


def _load_prompts() -> tuple[str, str]:
    """Load both system and task prompts."""
    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "prompts"
    )
    
    # Load system prompt
    system_prompt_path = os.path.join(base_path, "layout_checker_system_prompt.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
        
    # Load task prompt
    task_prompt_path = os.path.join(base_path, "layout_checker_prompt.txt")
    with open(task_prompt_path, "r", encoding="utf-8") as f:
        task_prompt = f.read()
        
    return system_prompt, task_prompt


def _load_scoring_prompt() -> tuple[str, str]:
    """Load scoring-only system and task prompts."""
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    system_prompt_path = os.path.join(base_path, "layout_scoring_system_prompt.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    scoring_prompt_path = os.path.join(base_path, "layout_scoring_prompt.txt")
    with open(scoring_prompt_path, "r", encoding="utf-8") as f:
        task_prompt = f.read()
    return system_prompt, task_prompt


def _load_advice_prompt() -> tuple[str, str]:
    """Load system and advice task prompts (use the checker prompts)."""
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    system_prompt_path = os.path.join(base_path, "layout_checker_system_prompt.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    advice_prompt_path = os.path.join(base_path, "layout_checker_prompt.txt")
    with open(advice_prompt_path, "r", encoding="utf-8") as f:
        task_prompt = f.read()
    return system_prompt, task_prompt


def _rects_overlap(a: Dict[str, int], b: Dict[str, int]) -> bool:
    ax1, ay1 = a.get("x", 0), a.get("y", 0)
    ax2, ay2 = ax1 + a.get("width", 0), ay1 + a.get("height", 0)
    bx1, by1 = b.get("x", 0), b.get("y", 0)
    bx2, by2 = bx1 + b.get("width", 0), by1 + b.get("height", 0)
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _get_layout_info(layout_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Get basic layout information for the model's reference."""
    items = layout_payload.get("items", []) or []
    canvas = layout_payload.get("canvas", {}) or {}
    cw = int(canvas.get("width", 1440) or 1440)
    ch = int(canvas.get("height", 810) or 810)

    # Add basic layout statistics for model context
    item_types = {}
    for it in items:
        type_ = it.get("type", "unknown")
        item_types[type_] = item_types.get(type_, 0) + 1

    return {
        "canvas": {"width": cw, "height": ch},
        "total_items": len(items),
        "item_types": item_types,
    }


def _parse_response(raw_text: str) -> Dict[str, Any]:
    """Parse the GPT response and ensure it's in the correct format."""
    # Remove code fences if present
    text = raw_text.strip()
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if end > start >= 0:
            text = text[start + 3 : end].strip()
            if "\n" in text:
                text = text.split("\n", 1)[1].strip()

    try:
        data = json.loads(text)
        # Normalize fields
        score = data.get("AestheticScore")
        if score is None:
            score = data.get("Score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None

        acceptable = data.get("Acceptable")
        if acceptable is None and score is not None:
            acceptable = bool(score >= 7.0)

        return {
            "Acceptable": bool(acceptable) if acceptable is not None else False,
            "AestheticScore": float(score) if score is not None else None,
            "Problem": data.get("Problem", []) or [],
            "Fixes": data.get("Fixes", []) or []
        }
    except Exception:
        # Fallback response if JSON parsing fails
        return {
            "Acceptable": False,
            "AestheticScore": None,
            "Problem": ["Layout analysis failed"],
            "Fixes": ["Please review the layout manually"]
        }


def _simplify_html_for_checker(html: str) -> str:
    """Remove embedded data URLs and other huge payloads to keep token size small."""
    try:
        # Replace any data: URLs in src/href with placeholders
        simplified = re.sub(r'(src|href)="data:[^"]+"', r'\1="__DATA__"', html)
        return simplified
    except Exception:
        return html


def _get_project_root():
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


def _render_html_to_image(html_content: str, filename_prefix: str = "checker", size: tuple[int, int] = (1440, 810)) -> Optional[str]:
    """Render HTML to a PNG screenshot and return the file path, or None on failure.
    Uses Playwright for better performance and reliability.
    """
    import asyncio
    
    # Use the sync wrapper to handle the async screenshot function
    return _render_html_to_image_sync(html_content, filename_prefix, size)


async def _render_html_to_image_async(html_content: str, filename_prefix: str = "checker", size: tuple[int, int] = (1440, 810)) -> Optional[str]:
    """Async version using Playwright for HTML to image rendering."""
    try:
        from playwright.async_api import async_playwright
        import pathlib
    except Exception:
        return None

    try:
        # Output directories in project root
        project_root = _get_project_root()
        output_dir = os.path.join(project_root, "debug_layouts", "layout_checker_screenshots")
        temp_dir = os.path.join(project_root, "debug_layouts", "layout_checker_temp_html")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Clean filename based on prefix
        safe_prefix = re.sub(r'[^\w\s-]', '', filename_prefix).strip()
        safe_prefix = re.sub(r'[-\s]+', '_', safe_prefix) 
        if not safe_prefix:
            safe_prefix = "checker"
        ts = int(time.time() * 1000)
        filename = f"{safe_prefix}_{ts}.png"
        screenshot_path = os.path.join(output_dir, filename)

        # Create temporary HTML file
        temp_html_path = os.path.join(temp_dir, f"temp_{safe_prefix}_{ts}.html")
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
            
            # Set viewport to match specified size
            await page.set_viewport_size({'width': size[0], 'height': size[1]})
            
            # Navigate to the HTML file
            await page.goto(file_uri, wait_until="load")
            
            # Wait for any animations or late-loading content
            await page.wait_for_timeout(2000)  # 2 seconds should be sufficient for layout checker
            await page.wait_for_load_state("networkidle")

            # Ensure MathJax (if present) has finished typesetting before screenshot
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
            
            # Take screenshot with current viewport size
            await page.screenshot(
                path=screenshot_path,
                full_page=False
            )
            
            await browser.close()

        # Clean up temporary HTML file
        try:
            os.remove(temp_html_path)
        except Exception:
            pass

        # Verify screenshot was created successfully
        if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
            return screenshot_path
        else:
            return None

    except Exception:
        # Clean up on error
        try:
            if 'temp_html_path' in locals() and os.path.exists(temp_html_path):
                os.remove(temp_html_path)
            if 'screenshot_path' in locals() and os.path.exists(screenshot_path):
                os.remove(screenshot_path)
        except Exception:
            pass
        return None


def _render_html_to_image_sync(html_content: str, filename_prefix: str = "checker", size: tuple[int, int] = (1440, 810)) -> Optional[str]:
    """Synchronous wrapper for the async render function."""
    import asyncio
    
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use thread executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(_render_html_to_image_async(html_content, filename_prefix, size))
            )
            return future.result()
    except RuntimeError:
        # No event loop running, we can use asyncio.run directly
        return asyncio.run(_render_html_to_image_async(html_content, filename_prefix, size))


def check_layout(
    html_content: str,
    layout_payload: Dict[str, Any],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4.1-2025-04-14",
    screenshot_path: Optional[str] = None,
    slide_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a GPT-4 based critique on the provided HTML and extracted layout JSON.
    Returns a dict with Acceptable (bool) and Fixes (list[str]).
    """
    system_prompt, task_prompt = _load_prompts()

    if client is None:
        client = OpenAI()

    try:
        # Get basic layout info for context
        layout_info = _get_layout_info(layout_payload)
        
        # Prepare the input for the model with enhanced context
        input_data = {
            "html": _simplify_html_for_checker(html_content),
            "layout": layout_payload,
            "layout_info": layout_info,
            "task_prompt": task_prompt,
        }

        # If no screenshot provided, try to render one internally
        if not screenshot_path:
            try:
                filename_prefix = slide_title if slide_title else "checker"
                screenshot_path = _render_html_to_image(html_content, filename_prefix)
            except Exception:
                screenshot_path = None

        # Build user content, optionally embedding screenshot as an image URL
        user_content: Any
        if screenshot_path and os.path.isfile(screenshot_path):
            try:
                with open(screenshot_path, "rb") as f:
                    b64img = base64.b64encode(f.read()).decode("ascii")
                user_content = [
                    {"type": "text", "text": json.dumps(input_data, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}},
                ]
            except Exception:
                user_content = json.dumps(input_data, ensure_ascii=False)
        else:
            user_content = json.dumps(input_data, ensure_ascii=False)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # Use chat.completions API
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
        )
        content = resp.choices[0].message.content
        return _parse_response(content)
    except Exception as e:
        # Minimal fallback response
        return {
            "Acceptable": False,
            "AestheticScore": None,
            "Fixes": [f"Layout analysis error: {str(e)}. Please review manually."]
        }


def score_layout(
    html_content: str,
    layout_payload: Dict[str, Any],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o",
    screenshot_path: Optional[str] = None,
    slide_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Scoring-only API: returns {AestheticScore, Acceptable}.
    Only uses screenshot and layout JSON for scoring; ignores HTML content.
    """
    if client is None:
        client = OpenAI()
    system_prompt, scoring_prompt = _load_scoring_prompt()
    try:
        input_data = {
            "layout": layout_payload,
            "task_prompt": scoring_prompt,
        }
        if not screenshot_path:
            try:
                filename_prefix = slide_title if slide_title else "score_checker"
                screenshot_path = _render_html_to_image(html_content, filename_prefix)
            except Exception:
                screenshot_path = None
        user_content: Any
        if screenshot_path and os.path.isfile(screenshot_path):
            try:
                with open(screenshot_path, "rb") as f:
                    b64img = base64.b64encode(f.read()).decode("ascii")
                user_content = [
                    {"type": "text", "text": json.dumps(input_data, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}},
                ]
            except Exception:
                user_content = json.dumps(input_data, ensure_ascii=False)
        else:
            user_content = json.dumps(input_data, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = client.chat.completions.create(model=model, messages=messages)
        raw = resp.choices[0].message.content
        # Parse only score+acceptable
        data = _parse_response(raw)
        # Strip problems/fixes from scoring-only output
        return {"AestheticScore": data.get("AestheticScore"), "Acceptable": data.get("Acceptable", False)}
    except Exception as e:
        return {"AestheticScore": None, "Acceptable": False}


def advise_layout(
    html_content: str,
    layout_payload: Dict[str, Any],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o",
    screenshot_path: Optional[str] = None,
    slide_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Advice-only API: returns {Problem, Fixes}."""
    if client is None:
        client = OpenAI()
    system_prompt, advice_prompt = _load_advice_prompt()
    try:
        input_data = {
            "html": _simplify_html_for_checker(html_content),
            "layout": layout_payload,
            "task_prompt": advice_prompt,
        }
        if not screenshot_path:
            try:
                filename_prefix = slide_title if slide_title else "advice_checker"
                screenshot_path = _render_html_to_image(html_content, filename_prefix)
            except Exception:
                screenshot_path = None
        user_content: Any
        if screenshot_path and os.path.isfile(screenshot_path):
            try:
                with open(screenshot_path, "rb") as f:
                    b64img = base64.b64encode(f.read()).decode("ascii")
                user_content = [
                    {"type": "text", "text": json.dumps(input_data, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}},
                ]
            except Exception:
                user_content = json.dumps(input_data, ensure_ascii=False)
        else:
            user_content = json.dumps(input_data, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.4)
        raw = resp.choices[0].message.content
        data = _parse_response(raw)
        return {"Problem": data.get("Problem", []), "Fixes": data.get("Fixes", [])}
    except Exception as e:
        return {"Problem": ["Advice generation error"], "Fixes": ["Please review manually"]}


__all__ = ["check_layout"]
