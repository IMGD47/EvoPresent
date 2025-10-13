import os
import re
import base64
import json
import argparse

def combine_html_slides(presentation_dir: str, output_path: str) -> None:
    try:
        slide_files = sorted(
            [f for f in os.listdir(presentation_dir) if f.startswith('slide_') and f.endswith('.html')],
            key=lambda x: int(re.search(r'slide_(\d+)\.html', x).group(1))
        )

        if not slide_files:
            print("No slide files found to combine.")
            return

        base64_slides = []
        for slide_file in slide_files:
            slide_path = os.path.join(presentation_dir, slide_file)
            with open(slide_path, 'rb') as f:
                content_bytes = f.read()
            b64 = base64.b64encode(content_bytes).decode('ascii')
            base64_slides.append(b64)

        slides_json = json.dumps(base64_slides)

        # Use placeholder replacement to avoid f-string brace escaping issues
        wrapper_html = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Slides</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
            background: #000;
        }
        #frame {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
            background: #fff;
        }
        .nav {
            position: fixed;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 12px;
            z-index: 1000;
        }
        .nav button {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: 1px solid #ccc;
            background: #fff;
            cursor: pointer;
            padding: 0;
            transition: background-color 0.2s, box-shadow 0.2s;
        }
        .nav button:hover {
            background-color: #f0f0f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav button svg {
            width: 24px;
            height: 24px;
            color: #333;
        }
        #counter {
            position: fixed;
            bottom: 16px;
            right: 16px;
            color: #fff;
            background: rgba(0,0,0,0.5);
            padding: 4px 8px;
            border-radius: 4px;
            font-family: sans-serif;
            font-size: 14px;
        }
    </style>
    <script>
        const slidesBase64 = %%SLIDES_JSON%%;
    </script>
</head>
<body>
    <iframe id=\"frame\" src=\"about:blank\"></iframe>
    <div class=\"nav\">
        <button id=\"prev\" title=\"Previous Slide\">
            <svg width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"15 18 9 12 15 6\"></polyline></svg>
        </button>
        <button id=\"next\" title=\"Next Slide\">
            <svg width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"9 18 15 12 9 6\"></polyline></svg>
        </button>
    </div>
    <div id=\"counter\"></div>

    <script>
        let idx = 0;
        const frame = document.getElementById('frame');
        const prevBtn = document.getElementById('prev');
        const nextBtn = document.getElementById('next');
        const counter = document.getElementById('counter');

        function base64ToBlob(b64, contentType = 'text/html') {
            const byteChars = atob(b64);
            const byteNumbers = new Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {
                byteNumbers[i] = byteChars.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: contentType + ';charset=utf-8' });
        }

        let currentObjectUrl = null;

        function show(i) {
            if (!slidesBase64.length) return;
            idx = (i + slidesBase64.length) % slidesBase64.length;
            const blob = base64ToBlob(slidesBase64[idx]);
            if (currentObjectUrl) {
                URL.revokeObjectURL(currentObjectUrl);
            }
            currentObjectUrl = URL.createObjectURL(blob);
            frame.src = currentObjectUrl;
            counter.textContent = `${idx + 1} / ${slidesBase64.length}`;
        }

        function next() { show(idx + 1); }
        function prev() { show(idx - 1); }

        prevBtn.addEventListener('click', prev);
        nextBtn.addEventListener('click', next);

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') next();
            if (e.key === 'ArrowLeft' || e.key === 'PageUp' || e.key === 'Backspace') prev();
            if (e.key === 'Home') show(0);
            if (e.key === 'End') show(slidesBase64.length - 1);
        });

        show(0);
    </script>
</body>
</html>
"""
        wrapper_html = wrapper_html.replace('%%SLIDES_JSON%%', slides_json)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_html)

        print(f"\nCombined presentation saved to {output_path}")

    except Exception as e:
        print(f"Error combining HTML slides: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine per-slide HTML files into a single navigable HTML')
    parser.add_argument('--slides_dir', required=True, help='Directory containing slide_*.html files')
    parser.add_argument('--output', required=True, help='Path to save the combined HTML file')
    args = parser.parse_args()
    combine_html_slides(args.slides_dir, args.output)


