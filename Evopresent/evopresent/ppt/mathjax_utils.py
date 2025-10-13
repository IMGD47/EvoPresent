#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MathJax工具模块
用于在HTML中注入MathJax支持，确保LaTeX公式正确渲染
"""

from bs4 import BeautifulSoup
import re


def determine_equation_importance(equation_text, context=None):
    """基于公式内容和上下文确定公式的重要性"""
    return "normal"  # 统一使用normal重要性

def determine_color_scheme(equation_text, importance):
    """基于公式内容和重要性确定颜色方案"""
    return "academic"  # 统一使用academic方案

def inject_mathjax_and_group_equations(html_content, slide_data=None, config=None):
    """Inject MathJax for LaTeX rendering and group equations with their variable explanations.

    This function ensures that equations written in $...$ or $$...$$ are rendered correctly
    and that any accompanying "Variables:" explanation stays directly under the equation
    to avoid scattered layouts.

    Args:
        html_content: The HTML content to process
        slide_data: Optional slide data containing equation explanations
        config: Optional configuration dictionary containing equation styling settings
    """
    # 设置默认配置
    default_config = {
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
    
    # 合并配置
    if config and isinstance(config, dict):
        eq_settings = config.get("equation_settings", {})
        if eq_settings:
            for key, value in eq_settings.items():
                if isinstance(value, dict):
                    default_config["equation_settings"].setdefault(key, {}).update(value)
                else:
                    default_config["equation_settings"][key] = value

    # 强制仅保留 academic 风格，移除冗余
    academic_colors = default_config["equation_settings"]["color_schemes"].get("academic", {
        "base": "rgb(0, 0, 0)",
        "variable": "rgb(0, 0, 0)",
        "operator": "rgb(0, 0, 0)",
        "number": "rgb(0, 0, 0)",
        "text": "rgb(0, 0, 0)"
    })
    default_config["equation_settings"]["color_schemes"] = {"academic": academic_colors}
    default_config["equation_settings"]["preferred_scheme"] = "academic"
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Ensure <head> exists
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

        # 如果配置关闭渲染，直接返回
        eq_settings = default_config["equation_settings"]
        if not eq_settings.get("enabled", True) or eq_settings.get("render") == "none":
            return str(soup)

        # Preprocess: unwrap <code>/<pre> that contain LaTeX so MathJax will process them
        try:
            for tag_name in ['code', 'pre']:
                for node in soup.find_all(tag_name):
                    txt = node.get_text() if hasattr(node, 'get_text') else ''
                    if not txt:
                        continue
                    if re.search(r'(\\\(|\\\)|\\\[|\\\]|\$\$[\s\S]*?\$\$|\$[^\$]+\$)', txt):
                        # Replace with a span preserving the text content
                        span = soup.new_tag('span')
                        span.string = txt
                        node.replace_with(span)
        except Exception:
            pass

        # Normalize common double-escaped math delimiters (\\( -> \(), (and ] counterparts)
        try:
            for text_node in soup.find_all(string=True):
                s = str(text_node)
                if '\\\\(' in s or '\\\\)' in s or '\\\\[' in s or '\\\\]' in s:
                    s = s.replace('\\\\(', '\\(').replace('\\\\)', '\\)')
                    s = s.replace('\\\\[', '\\[').replace('\\\\]', '\\]')
                    text_node.replace_with(s)
        except Exception:
            pass

        # Inject MathJax configuration if not present
        has_mathjax = bool(soup.find(id='MathJax-script'))
        if not has_mathjax:
            cfg = soup.new_tag('script')
            cfg.attrs['type'] = 'text/javascript'
            cfg.attrs['id'] = 'mathjax-config-custom'
            cfg.string = (
                "window.MathJax = {" +
                "  tex: {" +
                "    inlineMath: [['$','$'], ['\\\\(','\\\\)']], " +
                "    displayMath: [['$$','$$'], ['\\\\[','\\\\]']], " +
                "    processEscapes: true, " +
                "    processEnvironments: true, " +
                "    macros: { " +
                "      'R': '\\\\mathbb{R}', " +
                "      'E': '\\\\mathbb{E}', " +
                "      'bm': ['\\\\boldsymbol{#1}',1], " +
                "      'norm': ['\\\\left\\\\|#1\\\\right\\\\|',1] " +
                "    }" +
                "  }, " +
                "  options: {" +
                "    skipHtmlTags: ['script','noscript','style','textarea']" +
                "  }," +
                "  svg: {" +
                "    scale: 1, " +  # 使用默认大小，让CSS控制缩放
                "    mtextInheritFont: true" +
                "  }," +
                "  chtml: {" +
                "    scale: 1," +  # 使用默认大小，让CSS控制缩放
                "    mtextInheritFont: true" +
                "  }," +
                "  responsive: {" +
                "    enabled: true" +  # 启用响应式布局
                "  }" +
                "};"
            )
            soup.head.append(cfg)

            script = soup.new_tag('script', id='MathJax-script', src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js')
            script.attrs['type'] = 'text/javascript'
            script.attrs['defer'] = ''
            soup.head.append(script)

            # Add a fallback CDN and ensure typesetting is triggered; also expose a readiness flag
            fallback_loader = soup.new_tag('script')
            fallback_loader.attrs['type'] = 'text/javascript'
            fallback_loader.string = (
                "(function(){"  
                "  function loadFallback(){"  
                "    if(window.MathJax) return;"  
                "    var s=document.createElement('script');"  
                "    s.src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml.js';"  
                "    s.defer=true; s.id='MathJax-script-fallback'; document.head.appendChild(s);"  
                "  }"  
                "  setTimeout(function(){ if(!window.MathJax) loadFallback(); }, 1500);"  
                "})();"
            )
            soup.head.append(fallback_loader)

            startup_script = soup.new_tag('script')
            startup_script.attrs['type'] = 'text/javascript'
            startup_script.string = (
                "document.addEventListener('DOMContentLoaded', function(){" 
                "  function typeset(){ if(window.MathJax && window.MathJax.typesetPromise){ window.MathJax.typesetPromise().then(function(){ window._mathjaxReady = true; }); } else { window._mathjaxReady = true; } }" 
                "  try { typeset(); } catch(e) { window._mathjaxReady = true; }" 
                "  try { var obs = new MutationObserver(function(){ typeset(); }); obs.observe(document.body || document.documentElement, {childList:true, subtree:true}); } catch(e) {}" 
                "});"
            )
            soup.head.append(startup_script)

        # Minimal CSS for equations: no boxes, compact spacing, centered layout
        eq_settings = default_config["equation_settings"]
        base_sizes = eq_settings["base_sizes"]
        color_schemes = eq_settings["color_schemes"]
        importance_scales = eq_settings["importance_scales"]
        
        # 生成统一的CSS样式
        importance_css = [
            ".equation-group {",
            "  --eq-base-color: rgb(0, 0, 0);",
            "  --eq-variable-color: rgb(0, 0, 0);",
            "  --eq-operator-color: rgb(0, 0, 0);",
            "  --eq-number-color: rgb(0, 0, 0);",
            "  --eq-text-color: rgb(0, 0, 0);",
            "  --eq-font-size: 18px;",
            "}"
        ]
        
        # 基础样式
        style_css = (
            "/* 基础布局样式 */\n"
            ".equation-group { display: flex; flex-direction: column; align-items: center; margin: 12px auto; padding: 0; width: 100%; }\n"
            ".equation-group .equation { display: block; text-align: center; margin: 0; padding: 0; width: 100%; }\n"
            ".equation-group .equation .MathJax { display: block !important; margin: 0 auto !important; font-size: var(--eq-font-size) !important; }\n"
            ".equation-group .variables { display: block; width: 100%; margin: 6px auto 0 auto; padding: 0; text-align: left; font-size: var(--eq-font-size); }\n"
            ".equation-group .variables ul { margin: 4px 0 0 1.2em; padding: 0; list-style: disc; }\n"
            ".equation-group .variables ul li { margin: 2px 0; }\n"
            "\n/* 颜色应用 */\n"
            ".equation-group .MathJax { color: var(--eq-base-color) !important; }\n"
            ".equation-group .mjx-mi { color: var(--eq-variable-color) !important; }\n"
            ".equation-group .mjx-mo { color: var(--eq-operator-color) !important; }\n"
            ".equation-group .mjx-mn { color: var(--eq-number-color) !important; }\n"
            ".equation-group .mjx-text { color: var(--eq-text-color) !important; }\n"
            "\n/* 保护标题样式 */\n"
            "h1, h2, h3, h4, h5, h6 { color: inherit !important; font-family: inherit !important; font-size: inherit !important; }\n" +
            "\n".join(importance_css) +
            "\n"
        )

        style_tag = soup.new_tag('style')
        style_tag.string = style_css
        soup.head.append(style_tag)

        # Remove any accidental visible dumps of the MathJax config text from body
        try:
            # First find and remove any visible MathJax config text in the body
            for node in soup.find_all(string=lambda text: text and "window.MathJax" in text):
                if not node.parent or node.parent.name != 'script':
                    # If the text is in a block element, remove the entire element
                    parent = node.parent
                    while parent and parent.name not in ['body', 'html']:
                        if parent.name in ['p', 'div', 'pre', 'code', 'span']:
                            parent.decompose()
                            break
                        parent = parent.parent
                    if not parent or parent.name in ['body', 'html']:
                        node.extract()

            # Then remove any duplicate MathJax config scripts not in head
            for script in soup.find_all('script'):
                if script and script.string and 'window.MathJax' in script.string:
                    if not script.parent or script.parent.name != 'head':
                        script.decompose()

            # Finally, ensure there's only one MathJax config in head
            scripts_in_head = soup.head.find_all('script', id='mathjax-config-custom') if soup.head else []
            if len(scripts_in_head) > 1:
                for script in scripts_in_head[1:]:
                    script.decompose()
        except Exception:
            pass

        # Prepare equation explanations (if present in slide data)
        equation_explanations = {}
        try:
            if slide_data and isinstance(slide_data, dict):
                content_eq = slide_data.get('content', {}).get('equations', {})
                for eq_item in content_eq.get('display', []) or []:
                    eq_text = eq_item.get('equation') or ''
                    equation_explanations[eq_text] = eq_item.get('explanation', {})
        except Exception:
            pass

        # Helper for creating variables block from explanation dict
        def build_variables_block(explanation_dict):
            if not explanation_dict:
                return None
            variables = explanation_dict.get('variables') or {}
            if not variables:
                return None
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

        # Helper: strip decorative wrappers (cards, borders) from around a given node
        def strip_decorative_wrappers(target, include_self=False, max_levels=3):
            try:
                suspicious_tokens = ['bg-', 'shadow', 'border', 'rounded', 'card', 'panel', 'box', 'ring', 'backdrop', 'outline']
                # Optionally clean the target node itself
                def clean_node(n):
                    changed_local = False
                    # Clean classes
                    classes = n.get('class', []) or []
                    if classes:
                        filtered = []
                        for c in classes:
                            if any(tok in c for tok in suspicious_tokens):
                                changed_local = True
                                continue
                            # Remove large Tailwind paddings/margins around equations
                            if re.match(r'^(p|px|py|pt|pr|pb|pl|m|mx|my|mt|mr|mb|ml)-(?:\d+|\[)', c):
                                changed_local = True
                                continue
                            filtered.append(c)
                        if filtered != classes:
                            if filtered:
                                n['class'] = filtered
                            else:
                                del n['class']
                    # Clean inline styles
                    style_attr = n.get('style')
                    if style_attr:
                        parts = [p.strip() for p in style_attr.split(';') if p.strip()]
                        kept = []
                        for p in parts:
                            lower = p.lower()
                            if any(k in lower for k in ['background', 'border', 'box-shadow', 'backdrop-filter', 'outline']):
                                changed_local = True
                                continue
                            # aggressive trim of excess padding around equations
                            if lower.startswith('padding') or lower.startswith('border-radius'):
                                changed_local = True
                                continue
                            kept.append(p)
                        if kept:
                            n['style'] = '; '.join(kept)
                        else:
                            del n['style']
                    return changed_local

                current = target if include_self else target.parent
                level = 0
                while current and level < max_levels and current.name in ['div', 'section', 'article', 'figure']:
                    changed = clean_node(current)
                    # If wrapper contains only our target and no other meaningful content, unwrap it
                    try:
                        child_elements = [c for c in current.contents if getattr(c, 'name', None)]
                        text_only = (current.get_text(strip=True) == str(target.get_text(strip=True)) if hasattr(target, 'get_text') else False)
                        if len(child_elements) == 1 and (child_elements[0] == target or text_only):
                            current.unwrap()
                            # after unwrap, target's parent moved one level up
                            level += 1
                            current = target.parent
                            continue
                    except Exception:
                        pass
                    level += 1
                    current = current.parent
            except Exception:
                return

        # Find explicit variables block nodes that the LLM might have created (body only)
        def is_variables_node(node):
            if not getattr(node, 'get_text', None):
                return False
            txt = node.get_text(strip=True).lower()
            return txt.startswith('variables:') or txt == 'variables' or 'variables:' in txt

        search_root = soup.body if soup.body else soup
        variables_nodes = [n for n in search_root.find_all(True) if is_variables_node(n)]

        # Detect display equations in text nodes (body only, skip script/style/noscript)
        candidate_nodes = search_root.find_all(string=re.compile(r'\$\$[\s\S]*?\$\$'))
        eq_text_nodes = [
            t for t in candidate_nodes
            if t.parent and t.parent.name not in ['script', 'style', 'noscript']
        ]

        # If there is exactly one equation and one variables block, group them directly
        if len(eq_text_nodes) == 1 and len(variables_nodes) == 1:
            eq_text_node = eq_text_nodes[0]
            eq_parent = eq_text_node.parent
            eq_container = soup.new_tag('div', attrs={'class': 'equation-group'})

            eq_div = soup.new_tag('div', attrs={'class': 'equation'})
            eq_div.string = eq_text_node
            eq_container.append(eq_div)

            # Move existing variables node inside container
            variables_nodes[0].extract()
            eq_container.append(variables_nodes[0])

            # Replace parent if it only contained the equation text; else insert after
            if eq_parent and eq_parent.get_text(strip=True) == str(eq_text_node).strip():
                eq_parent.replace_with(eq_container)
            else:
                eq_parent.insert_after(eq_container)
                eq_text_node.replace_with('')

            # Remove decorative wrappers around the inserted equation block and its former parent
            strip_decorative_wrappers(eq_container, include_self=True)
            if eq_parent:
                strip_decorative_wrappers(eq_parent, include_self=True)

        else:
            # General case: wrap each equation, append nearest variables block or build one from slide_data
            for text_node in eq_text_nodes:
                eq_parent = text_node.parent
                eq_full = str(text_node)

                # 规范化：将花体的期望符号 \mathcal{E} 替换为标准形式（默认 \mathbb{E}）
                try:
                    eq_full = re.sub(r"\\mathcal\s*\{\s*E\s*\}", r"\\mathbb{E}", eq_full)
                except Exception:
                    pass

                # 分析公式内容和上下文
                eq_text = eq_full.strip('$')
                context = eq_parent.get_text() if eq_parent else None
                
                # 确定公式重要性和颜色方案
                importance = determine_equation_importance(eq_text, context)
                color_scheme = determine_color_scheme(eq_text, importance)
                preferred_scheme = eq_settings.get("preferred_scheme")
                if preferred_scheme and preferred_scheme in color_schemes:
                    color_scheme = preferred_scheme
                
                # 创建带有动态样式类的容器
                container = soup.new_tag('div', attrs={
                    'class': f'equation-group equation-{importance}-{color_scheme}'
                })
                eq_div = soup.new_tag('div', attrs={'class': 'equation'})
                eq_div.string = eq_full
                container.append(eq_div)

                # Try to attach an existing variables block that is closest after the equation
                attached = False
                for cand in variables_nodes:
                    # Find the first variables node that appears after the equation parent in document order
                    # and is not already used
                    if getattr(cand, '_used', False):
                        continue
                    # Heuristic: ensure they share a common ancestor section/container
                    if eq_parent and cand and eq_parent in cand.parents:
                        cand._used = True
                        cand.extract()
                        container.append(cand)
                        attached = True
                        break

                # If none found, try to build variables list from slide_data explanation
                if not attached and equation_explanations:
                    # Fuzzy match by removing whitespace
                    def normalize(s):
                        return re.sub(r'\s+', '', s or '')
                    eq_norm = normalize(eq_full.strip('$'))
                    match_key = None
                    for k in equation_explanations.keys():
                        if normalize(k) in eq_norm or eq_norm in normalize(k):
                            match_key = k
                            break
                    if match_key:
                        v_block = build_variables_block(equation_explanations.get(match_key))
                        if v_block:
                            container.append(v_block)

                # Replace or insert the container (only operate within body)
                try:
                    is_in_body = search_root and (eq_parent == search_root or (eq_parent and eq_parent in search_root.descendants))
                except Exception:
                    is_in_body = True

                if is_in_body:
                    if eq_parent and eq_parent.get_text(strip=True) == eq_full.strip():
                        eq_parent.replace_with(container)
                    else:
                        eq_parent.insert_after(container)
                        text_node.replace_with('')

                    # Remove decorative wrappers around the inserted equation block and its former parent
                    strip_decorative_wrappers(container, include_self=True)
                    if eq_parent:
                        strip_decorative_wrappers(eq_parent, include_self=True)

        return str(soup)
    except Exception as e:
        # On any failure, return the original content unchanged
        print(f"Warning: MathJax injection failed with error: {e}")
        import traceback
        traceback.print_exc()
        return html_content


def _group_equations_with_explanations(soup):
    """Group equations with their variable explanations."""
    try:
        # Find all paragraph elements that contain equations
        for p in soup.find_all('p'):
            text_content = p.get_text()
            if '$' in text_content:
                # Check if the next sibling contains variable explanations
                next_p = p.find_next_sibling('p')
                if next_p and ('Variables:' in next_p.get_text() or 'where' in next_p.get_text().lower()):
                    # Create a wrapper div for the equation and its explanation
                    wrapper = soup.new_tag('div', class_='equation-block')
                    
                    # Insert wrapper before the equation paragraph
                    p.insert_before(wrapper)
                    
                    # Move both paragraphs into the wrapper
                    wrapper.append(p.extract())
                    
                    # Add class to the explanation paragraph
                    next_p['class'] = next_p.get('class', []) + ['equation-vars']
                    wrapper.append(next_p.extract())

    except Exception as e:
        print(f"Warning: Error grouping equations: {e}")
