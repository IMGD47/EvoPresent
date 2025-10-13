#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layout extractor utilities for slides HTML using Playwright.

Exposes:
- extract_layout_async(input_path, width, height, wait_seconds)
- extract_layout_sync(input_path, width, height, wait_seconds)

This module is derived from the user's provided script and adapted as a reusable module.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict


# JavaScript extractor injected into the page to collect element positions and sizes
S_EXTRACTOR = r"""
(opts) => {
  const TARGET_W = opts.width;
  const TARGET_H = opts.height;
  const root = document.querySelector('.slide') || document.body;
  const rootRect = root.getBoundingClientRect();
  const scaleX = TARGET_W / rootRect.width;
  const scaleY = TARGET_H / rootRect.height;

  const isVisible = (el, cs) => {
    if (!cs) cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };

  const getElementInfo = (el, cs) => {
    const tag = el.tagName.toLowerCase();
    let type = 'other';
    let label = '';
    
    if (tag === 'img') {
      type = 'image';
      label = el.alt || 'Untitled Image';
      // Check if image is inside a figure
      const parentFigure = el.closest('figure');
      if (parentFigure) {
        const caption = parentFigure.querySelector('figcaption');
        if (caption) {
          label = caption.textContent.trim();
        }
      }
    } else {
      const isTextLike = /^h[1-6]$/.test(tag) || ['p','span','strong','em','li','blockquote','figcaption', 'a', 'b', 'i', 'u'].includes(tag);
      const hasOnlyText = el.children.length === 0 && el.textContent.trim().length > 0;
      
      if (isTextLike || (tag === 'div' && hasOnlyText)) {
        type = 'text';
        // Get text preview (first 50 chars)
        const text = el.textContent.trim();
        label = text.length > 50 ? text.substring(0, 47) + '...' : text;
        
        // Add heading level if applicable
        if (/^h[1-6]$/.test(tag)) {
          label = `H${tag[1]}: ${label}`;
        }
        
        // Mark if it's a caption
        if (tag === 'figcaption' || el.closest('figcaption')) {
          label = `Caption: ${label}`;
        }
      }
    }
    
    return { type, label };
  };

  const toCanvasRect = (rect) => {
    const x = (rect.left - rootRect.left) * scaleX;
    const y = (rect.top - rootRect.top) * scaleY;
    const w = rect.width * scaleX;
    const h = rect.height * scaleY;
    return { x: Math.round(x), y: Math.round(y), width: Math.round(w), height: Math.round(h) };
  };

  const ALLOWED = new Set(['h1','h2','h3','h4','h5','h6','p','span','strong','em','li','blockquote','figcaption','a','b','i','u','div','section','article','aside','header','footer','main','img','figure']);
  const items = [];
  let idCounter = 0;

  const treeWalker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, {
    acceptNode: (el) => {
      if (!ALLOWED.has(el.tagName.toLowerCase()) || el === root) return NodeFilter.FILTER_SKIP;
      const cs = getComputedStyle(el);
      return (cs.position !== 'fixed' && isVisible(el, cs)) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    }
  });

  while (treeWalker.nextNode()) {
    const el = treeWalker.currentNode;
    const cs = getComputedStyle(el);
    const info = getElementInfo(el, cs);
    if (info.type === 'text' || info.type === 'image') {
      const rect = el.getBoundingClientRect();
      const geom = toCanvasRect(rect);
      if (geom.width < 2 || geom.height < 2 || (!el.textContent.trim() && info.type === 'text')) continue;

      // Generate ID based on content
      let contentId = '';
      if (info.type === 'image') {
        // For images, use caption or alt text, fallback to 'img' if neither exists
        contentId = info.label.replace(/^Caption:\s*/, '').substring(0, 20).trim();
        contentId = contentId || 'img';
      } else {
        // For text, use the actual text content
        contentId = info.label.substring(0, 20).trim();
      }
      
      // Clean the contentId to make it URL-safe and valid as an identifier
      contentId = contentId.toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')  // Replace non-alphanumeric chars with underscore
        .replace(/^_+|_+$/g, '')      // Remove leading/trailing underscores
        .substring(0, 30);            // Limit length
      
      items.push({
        id: `${info.type}-${contentId}`,
        type: info.type,
        label: info.label,
        geometry: geom,
      });
    }
  }

  return {
    canvas: { width: TARGET_W, height: TARGET_H },
    meta: { extractedAt: new Date().toISOString(), sourceTitle: document.title, sourceUrl: document.location.href },
    items,
  };
}
"""


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https", "file")
    except Exception:
        return False


def analyze_spacing(items: list) -> dict:
    """
    Compute nearest neighbors and gaps for each item in four directions.
    Mirrors the user's helper to enrich the extractor payload.
    """
    relationships = {}
    for i, ref_item in enumerate(items):
        ref_geom = ref_item['geometry']
        ref_id = ref_item['id']

        neighbors = {
            'above': {'id': None, 'gap': float('inf')},
            'below': {'id': None, 'gap': float('inf')},
            'left': {'id': None, 'gap': float('inf')},
            'right': {'id': None, 'gap': float('inf')},
        }

        for j, other_item in enumerate(items):
            if i == j:
                continue

            other_geom = other_item['geometry']
            other_id = other_item['id']

            h_overlap = max(0, min(ref_geom['x'] + ref_geom['width'], other_geom['x'] + other_geom['width']) - max(ref_geom['x'], other_geom['x']))
            v_overlap = max(0, min(ref_geom['y'] + ref_geom['height'], other_geom['y'] + other_geom['height']) - max(ref_geom['y'], other_geom['y']))

            if h_overlap > 0 and other_geom['y'] < ref_geom['y']:
                gap = ref_geom['y'] - (other_geom['y'] + other_geom['height'])
                if gap >= 0 and gap < neighbors['above']['gap']:
                    neighbors['above'] = {'id': other_id, 'gap': round(gap)}

            if h_overlap > 0 and other_geom['y'] > ref_geom['y']:
                gap = other_geom['y'] - (ref_geom['y'] + ref_geom['height'])
                if gap >= 0 and gap < neighbors['below']['gap']:
                    neighbors['below'] = {'id': other_id, 'gap': round(gap)}

            if v_overlap > 0 and other_geom['x'] < ref_geom['x']:
                gap = ref_geom['x'] - (other_geom['x'] + other_geom['width'])
                if gap >= 0 and gap < neighbors['left']['gap']:
                    neighbors['left'] = {'id': other_id, 'gap': round(gap)}

            if v_overlap > 0 and other_geom['x'] > ref_geom['x']:
                gap = other_geom['x'] - (ref_geom['x'] + ref_geom['width'])
                if gap >= 0 and gap < neighbors['right']['gap']:
                    neighbors['right'] = {'id': other_id, 'gap': round(gap)}

        final_neighbors = {k: v for k, v in neighbors.items() if v['id'] is not None}
        if final_neighbors:
            relationships[ref_id] = final_neighbors

    return relationships


async def extract_layout_async(input_path: str, width: int, height: int, wait_seconds: float = 0.0) -> Dict[str, Any]:
    """
    Extract positions/sizes of text and image elements from an HTML file/URL.
    Returns a dict containing canvas size, meta, items, and relationships.
    """
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        raise RuntimeError("Playwright is required for layout extraction. Please install with: pip install playwright && playwright install") from e

    src = input_path
    p = Path(input_path)
    if p.exists() and not _is_url(input_path):
        src = p.resolve().as_uri()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1600, "height": 1000, "deviceScaleFactor": 1})
        await page.goto(src, wait_until="networkidle")
        if wait_seconds and wait_seconds > 0:
            await page.wait_for_timeout(int(wait_seconds * 1000))

        # Wait for MathJax readiness flag if present to stabilize equation rendering
        try:
            extra_timeout_ms = 4000 + int(max(0, (wait_seconds or 0)) * 1000)
            await page.wait_for_function(
                "() => (typeof window !== 'undefined' && (window._mathjaxReady === true || typeof window.MathJax === 'undefined'))",
                timeout=extra_timeout_ms
            )
        except Exception:
            # Proceed even if the readiness signal wasn't observed within time budget
            pass

        payload = await page.evaluate(S_EXTRACTOR, {"width": width, "height": height})

        if payload.get("items"):
            payload["relationships"] = analyze_spacing(payload["items"])  # enrich

        await browser.close()
        return payload


def extract_layout_sync(input_path: str, width: int, height: int, wait_seconds: float = 0.0) -> Dict[str, Any]:
    """Synchronous wrapper for extract_layout_async."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Nested loop context: create a new loop in a separate policy
        return asyncio.run(extract_layout_async(input_path, width, height, wait_seconds))
    else:
        return asyncio.run(extract_layout_async(input_path, width, height, wait_seconds))


def save_layout_to_file(payload: Dict[str, Any], out_path: str) -> None:
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "extract_layout_async",
    "extract_layout_sync",
    "save_layout_to_file",
    "analyze_spacing",
]


