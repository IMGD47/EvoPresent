import json
import re
from typing import Dict, Any, List, Tuple
from pathlib import Path

from camel.models import ModelFactory
from camel.agents import ChatAgent
from evopresent.ppt.visualization_helper import maybe_generate_visual_for_slide

class ScholarAgent:
    """
    Enriches slides according to per-slide scholar_request decisions (knowledge | image | none).
    - knowledge: Enhances script with relevant information from paper
    - image: Sets up slide for image generation
    """

    def __init__(self, actor_config: Dict[str, Any]):
        self._actor_config = actor_config
        # Create a text model for scholar review
        if actor_config.get('url') and str(actor_config.get('model_platform', '')).startswith('vllm'):
            self._model = ModelFactory.create(
                model_platform=actor_config['model_platform'],
                model_type=actor_config['model_type'],
                model_config_dict=actor_config['model_config'],
                url=actor_config['url'],
            )
        else:
            self._model = ModelFactory.create(
                model_platform=actor_config['model_platform'],
                model_type=actor_config['model_type'],
                model_config_dict=actor_config['model_config'],
            )
        self._writer_agent = None
        self._ensure_writer_agent()

    def _ensure_writer_agent(self) -> None:
        if self._writer_agent is not None:
            return
        # Writer agent for script enrichment and expansion
        self._writer_agent = ChatAgent(
            system_message=(
                "You are an expert academic presenter specializing in concise, on-point slide scripts. "
                "Enhance ONLY the current slide's script using the explicit enrichment requirement provided. "
            ),
            model=self._model,
            message_window_size=6,
            token_limit=self._actor_config.get('token_limit', None),
        )

    def _rewrite_script_direct(self, *, title: str, core_points: List[str], current_script: str, reason: str) -> str:
        try:
            self._writer_agent.reset()
            prompt = (
                "You are enhancing a single presentation slide's script.\n"
                "Constraints:\n"
                "- Use ONLY the current slide's script and the enrichment requirement below.\n"
                "- Do NOT introduce unrelated topics or external facts.\n"
                "- Stay concise and focused; aim for about 120–200 words.\n"
                "- Output ONLY the enhanced script as plain text (no headings/JSON).\n\n"
                ("Core Points:\n- " + "\n- ".join(core_points) + "\n\n" if core_points else "") +
                "Current Script:\n" + current_script + "\n\n" +
                "Enrichment Requirement:\n" + reason + "\n\n" +
                "Task: Rewrite the current script without deviating from the original content, fully addressing the enrichment requirements, while moderately expanding the script to enhance its clarity, completeness, educational value, and academic depth."
            )
            resp = self._writer_agent.step(prompt)
            new_script = resp.msgs[0].content.strip()
            return new_script
        except Exception:
            pass
        # Fallback: minimally address the reason without adding new sources
        bridge = (" " if current_script and not current_script.endswith(('.', '!', '?')) else "")
        guidance = (" (Focus: " + reason + ")" if reason else "")
        return f"{current_script}{bridge}{guidance}".strip()

    @staticmethod
    def _extract_appendix_text(paper_markdown: str) -> str:
        # Heuristic: capture sections starting at headings containing 'appendix' or 'supplementary'
        pattern = re.compile(r"(^|\n)##\s*(appendix|supplementary|supplemental)([\s\S]*)$", re.IGNORECASE)
        m = pattern.search(paper_markdown)
        if not m:
            return ""
        return m.group(3).strip()

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 0]
        return parts

    @staticmethod
    def _keywordize(texts: List[str]) -> List[str]:
        joined = " ".join(texts).lower()
        # Keep simple alphanumeric keywords of length >= 3
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", joined)
        return list(dict.fromkeys(tokens))  # dedupe preserving order

    @staticmethod
    def _score_paragraph(paragraph: str, keywords: List[str]) -> int:
        pl = paragraph.lower()
        score = 0
        for kw in keywords:
            if kw in pl:
                score += 1
        return score

    def enrich_slide_knowledge(self, *, paper_markdown: str, slide: Dict[str, Any], max_snippets: int = 3) -> Tuple[Dict[str, Any], str]:
        content = slide.get("content") or {}
        core_points: List[str] = content.get("core_points") or []
        title: str = (slide.get("slide_title") or "").strip()
        script: str = (content.get("script") or "").strip()
        reason: str = (content.get("scholar_request", {}).get("reason") or "").strip()

        # Direct rewrite using the full paper content and the explicit reason
        new_script = self._rewrite_script_direct(
            title=title,
            core_points=core_points,
            current_script=script,
            reason=reason,
        )

        if new_script and len(new_script.strip()) > 0:
            content["script"] = new_script
        slide["content"] = content
        # We bypass snippet sourcing; mark source as 'full_paper'
        return slide, "full_paper"

    @staticmethod
    def _looks_like_methods_or_background_or_problem(title: str) -> bool:
        tl = title.lower()
        keywords = ["method", "approach", "model", "algorithm", "background", "related", "motivation", "problem", "challenge", "pain point"]
        return any(k in tl for k in keywords)

    def decide_and_apply_enrichment(self, *, paper_markdown: str, slide: Dict[str, Any], save_dir: Path = None) -> Dict[str, Any]:
        """Apply enrichment based on scholar_request in the slide JSON.
        
        Args:
            paper_markdown: The paper content in markdown format
            slide: The slide dictionary to enrich
            save_dir: Optional directory to save generated images
        
        Returns:
            Updated slide dictionary
        """
        content = slide.get("content") or {}
        req = content.get("scholar_request") or {}
        rtype = (req.get("type") or "none").lower()
        reason = (req.get("reason") or "").strip()

        title: str = (slide.get("slide_title") or "").strip()

        print(f"\n=== Processing slide: {title} ===")
        print(f"Scholar request type: {rtype}")
        print(f"Reason: {reason}")

        # Strictly follow the scholar_request type from JSON
        if rtype == "knowledge":
            print("Applying knowledge enrichment based on scholar_request...")
            updated, source = self.enrich_slide_knowledge(paper_markdown=paper_markdown, slide=slide)
            slide.update(updated)
            print(f"Knowledge enrichment completed from {source}")
            
        elif rtype == "image":
            print("Applying image enrichment based on scholar_request...")
            
            # Ensure scholar_request is properly preserved
            content.setdefault("scholar_request", {})
            content["scholar_request"]["type"] = "image"
            content["scholar_request"]["reason"] = reason
            
            slide["content"] = content
            
            # Generate image if save directory is provided
            if save_dir:
                try:
                    print(f"Generating image for slide: {title}")
                    print(f"Using reason as prompt: '{reason}'")
                    
                    save_dir = Path(str(save_dir))
                    save_dir.mkdir(parents=True, exist_ok=True)
                    updated = maybe_generate_visual_for_slide(save_dir=save_dir, slide=slide)
                    if isinstance(updated, dict):
                        slide.update(updated)
                        print("Image generation completed successfully")
                    else:
                        print("Image generation returned no updates")
                except Exception as e:
                    print(f"Image generation failed: {str(e)}")
                    # Continue without failing the whole process
            else:
                print("No save directory provided, image generation will be handled downstream")
                
        elif rtype == "none":
            print("No enrichment requested based on scholar_request")
            
        else:
            print(f"Unknown scholar_request type: {rtype}, skipping enrichment")
        
        return slide

    # (Unified script generation removed)
