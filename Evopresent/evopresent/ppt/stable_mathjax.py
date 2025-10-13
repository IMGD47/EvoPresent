#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import hashlib
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List

class StableMathJaxRenderer:
    """Ensure consistent and reliable MathJax rendering with validation and fallback."""
    
    def __init__(self):
        self.processed_cache = {}
        self.config = self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration with required fields present."""
        return {
            "equation_settings": {
                "enabled": True,
                "render": "mathjax",
                "preferred_scheme": "academic",
                "color_schemes": {
                    "academic": {
                        "base": "rgb(0, 0, 0)",
                        "variable": "rgb(0, 0, 0)", 
                        "operator": "rgb(0, 0, 0)",
                        "number": "rgb(0, 0, 0)",
                        "text": "rgb(0, 0, 0)"
                    }
                },
                "base_sizes": {
                    "default": "18px"
                },
                "importance_scales": {
                    "normal": 1.0,
                    "high": 1.2,
                    "critical": 1.4
                }
            }
        }
    
    def is_mathjax_already_injected(self, html_content: str) -> bool:
        """Check whether MathJax has been injected into HTML."""
        return 'MathJax-script' in html_content and 'window.MathJax' in html_content
    
    def validate_equations(self, html_content: str) -> List[str]:
        """Validate equation syntax within HTML text content."""
        issues = []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()
        
        dollar_count = text_content.count('$')
        if dollar_count % 2 != 0:
            issues.append("Unmatched dollar symbol ($)")
            
        double_dollar_matches = re.findall(r'\$\$', text_content)
        if len(double_dollar_matches) % 2 != 0:
            issues.append("Unmatched double dollar symbols ($$)")
            
        critical_issues = [
            (r'\\frac[^\{]', "\\frac missing brace"),
            (r'\\sqrt[^\{\[]', "\\sqrt missing brace"), 
            (r'\\[a-zA-Z]+\s+[a-zA-Z]', "LaTeX command followed by letter (likely missing braces)")
        ]
        
        for pattern, message in critical_issues:
            if re.search(pattern, text_content):
                issues.append(message)
                
        return issues
    
    def inject_mathjax_safe(self, html_content: str, slide_data: Optional[Dict] = None, 
                           force_reinject: bool = False) -> str:
        """Safely inject MathJax, with validation and fallback."""
        
        # Content hash for caching
        content_hash = hashlib.md5(html_content.encode()).hexdigest()
        
        # Return cached result if available and not forcing reinjection
        if not force_reinject and content_hash in self.processed_cache:
            return self.processed_cache[content_hash]
            
        try:
            if not html_content or not html_content.strip():
                raise ValueError("HTML content is empty")
                
            if self.is_mathjax_already_injected(html_content) and not force_reinject:
                print("MathJax already injected, skipping")
                self.processed_cache[content_hash] = html_content
                return html_content
                
            validation_issues = self.validate_equations(html_content)
            if validation_issues:
                print(f"Equation validation issues: {', '.join(validation_issues)}")
                
            result = self._process_html_safe(html_content, slide_data)
            
            if not self.is_mathjax_already_injected(result):
                raise RuntimeError("MathJax injection failed")
                
            self.processed_cache[content_hash] = result
            return result
            
        except Exception as e:
            print(f"❌ MathJax injection failed: {e}")
            return html_content
    
    def _process_html_safe(self, html_content: str, slide_data: Optional[Dict] = None) -> str:
        """Safely mutate HTML: ensure head, inject MathJax config/styles, group equations."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if not soup.head:
            head_tag = soup.new_tag('head')
            if soup.html:
                soup.html.insert(0, head_tag)
            else:
                html_tag = soup.new_tag('html')
                html_tag.append(head_tag)
                if soup.body:
                    html_tag.append(soup.body)
                else:
                    body_tag = soup.new_tag('body')
                    html_tag.append(body_tag)
                soup.append(html_tag)
        
        self._inject_mathjax_config(soup)
        
        self._inject_equation_styles(soup)
        
        self._process_equations(soup, slide_data)
        
        return str(soup)
    
    def _inject_mathjax_config(self, soup):
        """Inject MathJax configuration and scripts."""
        existing_config = soup.find(id='mathjax-config-stable')
        if existing_config:
            return
            
        config_script = soup.new_tag('script')
        config_script.attrs['type'] = 'text/javascript'
        config_script.attrs['id'] = 'mathjax-config-stable'
        config_script.string = """
        window.MathJax = {
            tex: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                displayMath: [['$$','$$'], ['\\[','\\]']],
                processEscapes: true,
                processEnvironments: true,
                macros: {
                    'R': '\\mathbb{R}',
                    'E': '\\mathbb{E}',
                    'bm': ['\\boldsymbol{#1}',1],
                    'norm': ['\\left\\|#1\\right\\|',1]
                }
            },
            options: {
                skipHtmlTags: ['script','noscript','style','textarea']
            },
            svg: {
                scale: 1,
                mtextInheritFont: true
            },
            chtml: {
                scale: 1,
                mtextInheritFont: true
            },
            responsive: {
                enabled: true
            },
            startup: {
                ready: function () {
                    MathJax.startup.defaultReady();
                    window._mathjaxReady = true;
                }
            }
        };
        """
        soup.head.append(config_script)
        
        existing_script = soup.find(id='MathJax-script-stable')
        if not existing_script:
            script = soup.new_tag('script')
            script.attrs['id'] = 'MathJax-script-stable'
            script.attrs['src'] = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js'
            script.attrs['type'] = 'text/javascript'
            script.attrs['defer'] = ''
            soup.head.append(script)
        
        fallback_script = soup.new_tag('script')
        fallback_script.attrs['type'] = 'text/javascript'
        fallback_script.string = """
        setTimeout(function() {
            if (!window.MathJax) {
                var s = document.createElement('script');
                s.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml.js';
                s.defer = true;
                s.id = 'MathJax-script-fallback';
                document.head.appendChild(s);
            }
        }, 2000);
        """
        soup.head.append(fallback_script)
    
    def _inject_equation_styles(self, soup):
        """Inject CSS styles for equations/variables blocks."""
        existing_styles = soup.find(id='equation-styles-stable')
        if existing_styles:
            return
            
        style_tag = soup.new_tag('style')
        style_tag.attrs['id'] = 'equation-styles-stable'
        style_tag.string = """
        /* Stable equation styles */
        .equation-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 12px auto;
            padding: 0;
            width: 100%;
            --eq-base-color: rgb(0, 0, 0);
            --eq-variable-color: rgb(0, 0, 0);
            --eq-operator-color: rgb(0, 0, 0);
            --eq-number-color: rgb(0, 0, 0);
            --eq-text-color: rgb(0, 0, 0);
            --eq-font-size: 18px;
        }
        
        .equation-group .equation {
            display: block;
            text-align: center;
            margin: 0;
            padding: 0;
            width: 100%;
        }
        
        .equation-group .equation .MathJax {
            display: block !important;
            margin: 0 auto !important;
            font-size: var(--eq-font-size) !important;
            color: var(--eq-base-color) !important;
        }
        
        .equation-group .variables {
            display: block;
            width: 100%;
            margin: 6px auto 0 auto;
            padding: 0;
            text-align: left;
            font-size: var(--eq-font-size);
        }
        
        .equation-group .variables ul {
            margin: 4px 0 0 0;
            padding: 0;
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px 16px;
        }
        
        .equation-group .variables ul li {
            margin: 0;
            white-space: nowrap;
            font-size: 14px;
        }
        """
        soup.head.append(style_tag)
    
    def _process_equations(self, soup, slide_data):
        """Group equations and optionally attach variables list from slide_data."""
        search_root = soup.body if soup.body else soup
        candidate_nodes = search_root.find_all(string=re.compile(r'\$\$[\s\S]*?\$\$'))
        eq_text_nodes = [
            t for t in candidate_nodes
            if t.parent and t.parent.name not in ['script', 'style', 'noscript']
        ]
        
        for text_node in eq_text_nodes:
            eq_parent = text_node.parent
            eq_full = str(text_node)
            
            container = soup.new_tag('div', attrs={'class': 'equation-group equation-normal-academic'})
            eq_div = soup.new_tag('div', attrs={'class': 'equation'})
            eq_div.string = eq_full
            container.append(eq_div)
            
            if slide_data:
                variables_block = self._build_variables_block(soup, eq_full, slide_data)
                if variables_block:
                    container.append(variables_block)
            
            if eq_parent and eq_parent.get_text(strip=True) == eq_full.strip():
                eq_parent.replace_with(container)
            else:
                eq_parent.insert_after(container)
                text_node.replace_with('')
    
    def _build_variables_block(self, soup, equation_text, slide_data):
        """Build a variables block under an equation from slide_data."""
        try:
            equations_data = slide_data.get('content', {}).get('equations', {})
            display_equations = equations_data.get('display', [])
            
            equation_clean = equation_text.strip('$').strip()
            for eq_item in display_equations:
                if isinstance(eq_item, dict):
                    explanation = eq_item.get('explanation', {})
                    variables = explanation.get('variables', {})
                    if variables:
                        v_div = soup.new_tag('div', attrs={'class': 'variables'})
                        title = soup.new_tag('div')
                        title.string = 'Variables:'
                        v_div.append(title)
                        ul = soup.new_tag('ul')
                        for var, desc in variables.items():
                            li = soup.new_tag('li')
                            li.string = f"{var}: {desc}"
                            ul.append(li)
                        v_div.append(ul)
                        return v_div
            return None
        except Exception as e:
            print(f"Failed to build variables block: {e}")
            return None


# Global singleton instance
_stable_renderer = StableMathJaxRenderer()

def inject_mathjax_stable(html_content: str, slide_data: Optional[Dict] = None, 
                         force_reinject: bool = False) -> str:
    """Stable MathJax injector replacing the legacy injector."""
    return _stable_renderer.inject_mathjax_safe(html_content, slide_data, force_reinject)

def validate_mathjax_injection(html_content: str) -> bool:
    """Validate whether MathJax appears correctly injected in HTML."""
    return _stable_renderer.is_mathjax_already_injected(html_content)
