#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import subprocess
import tempfile
import asyncio
import re
import time
import pathlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from PIL import Image, ImageColor


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def _get_project_root():
    """Get project root directory"""
    current_dir = Path(__file__).resolve().parent
    while current_dir.parent != current_dir:
        if (current_dir / "video_generation").exists() or (current_dir / ".git").exists():
            return str(current_dir)
        current_dir = current_dir.parent
    return str(Path(__file__).resolve().parent)


def _save_image_temp(image: Image.Image, prefix: str = "temp") -> str:
    """
    Save PIL Image object to temporary file for FFmpeg use
    
    Args:
        image: PIL Image object
        prefix: Filename prefix
    Returns:
        Temporary file path
    """
    import tempfile
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix=f'{prefix}_')
    try:
        os.close(temp_fd)
        image.save(temp_path, 'PNG', optimize=True)
        return temp_path
    except Exception as e:
        logging.error(f"Failed to save temp image: {e}")
        try:
            os.remove(temp_path)
        except:
            pass
        raise


def _pad_image_canvas(
    image: Image.Image,
    pad_ratio_w: float = 0.06,
    pad_ratio_h: float = 0.06,
    pad_side: str = "right-bottom",
    pad_color: str = "#000000"
) -> Image.Image:
    """
    Expand canvas around image with proportional padding for face overlay space.
    
    Args:
        image: Input PIL Image object
        pad_ratio_w: Canvas width increase ratio (e.g., 0.06 = +6% width)
        pad_ratio_h: Canvas height increase ratio
        pad_side: Expansion direction: "right-bottom" | "right" | "bottom" | "all"
        pad_color: Canvas fill color (#RRGGBB or color name)
    Returns:
        Expanded PIL Image object
    """
    try:
        w, h = image.size
        add_w = int(w * pad_ratio_w)
        add_h = int(h * pad_ratio_h)
        add_w = max(add_w, 0)
        add_h = max(add_h, 0)
        if add_w == 0 and add_h == 0:
            return image

        color = ImageColor.getrgb(pad_color)

        if pad_side == "right-bottom":
            new_w, new_h = w + add_w, h + add_h
            canvas = Image.new("RGB", (new_w, new_h), color)
            canvas.paste(image, (0, 0))
        elif pad_side == "right":
            new_w, new_h = w + add_w, h
            canvas = Image.new("RGB", (new_w, new_h), color)
            canvas.paste(image, (0, 0))
        elif pad_side == "bottom":
            new_w, new_h = w, h + add_h
            canvas = Image.new("RGB", (new_w, new_h), color)
            canvas.paste(image, (0, 0))
        else:  # all
            left = add_w // 2
            top = add_h // 2
            new_w, new_h = w + add_w, h + add_h
            canvas = Image.new("RGB", (new_w, new_h), color)
            canvas.paste(image, (left, top))

        new_w_even = new_w - (new_w % 2)
        new_h_even = new_h - (new_h % 2)
        if new_w_even != new_w or new_h_even != new_h:
            canvas = canvas.resize((new_w_even, new_h_even), Image.Resampling.LANCZOS)

        return canvas
    except Exception as e:
        logging.error(f"Failed to pad image canvas: {e}")
        return image


async def _render_html_file_to_image_async(html_file_path: str, filename_prefix: str = "slide",
                                         size: Tuple[int, int] = (1440, 810)) -> Optional[Image.Image]:
    """Convert HTML file directly to PIL Image object (in-memory processing, no disk save)"""
    try:
        from playwright.async_api import async_playwright
        from pdf2image import convert_from_bytes
    except ImportError as e:
        logging.error(f"Required packages not installed: {e}")
        logging.error("Please install: pip install playwright pdf2image")
        return None
    
    try:
        file_uri = pathlib.Path(html_file_path).resolve().as_uri()
        
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
            
            await page.set_viewport_size({'width': size[0], 'height': size[1]})
            await page.goto(file_uri, wait_until="load")
            await page.wait_for_timeout(2000)
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
            
            pdf_bytes = await page.pdf(
                width=f'{size[0]}px',
                height=f'{size[1]}px',
                print_background=True,
                prefer_css_page_size=True
            )
            await browser.close()
        
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)
            if images:
                logging.info(f"✅ PIL Image created in memory: {images[0].size}")
                return images[0]
            else:
                logging.error("No pages found in PDF")
                return None
        except Exception as e:
            logging.error(f"PDF to PIL Image conversion failed: {e}")
            return None
            
    except Exception as e:
        logging.error(f"HTML file to PIL Image conversion failed: {e}")
        return None


def compose_slide_video_with_face(image_file: str, face_video: str, output_video: str,
                                  ffmpeg_cmd: str = "ffmpeg", face_scale: float = 0.25,
                                  margin_x: int = 30, margin_y: int = 30) -> None:
    """Composite PNG background with face video into a single encoded output video (audio from face video)."""
    filter_complex = (
        f"[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2[bg];"
        f"[1:v]scale=iw*{face_scale}:-1[face];"
        f"[bg][face]overlay=main_w-overlay_w-{margin_x}:main_h-overlay_h-{margin_y}:shortest=1"
    )
    cmd = [
        ffmpeg_cmd, '-y',
        '-loop', '1', '-i', image_file,
        '-i', face_video,
        '-filter_complex', filter_complex,
        '-map', '0:v:0', '-map', '1:a:0?',
        '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        output_video
    ]
    logging.info(f'Compositing slide (single encode, with face audio): {" ".join(cmd)}')
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def compose_slide_video_png_with_audio(image_file: str, tts_audio: str, output_video: str,
                                       ffmpeg_cmd: str = "ffmpeg") -> None:
    """Composite PNG background with TTS audio into a video (no face overlay)."""
    cmd = [
        ffmpeg_cmd, '-y',
        '-loop', '1', '-i', image_file,
        '-i', tts_audio,
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        output_video
    ]
    logging.info(f'Compositing slide (PNG + TTS audio, no face): {" ".join(cmd)}')
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def overlay_face_video(bg_video: str, face_video: str, output_video: str, 
                      ffmpeg_cmd: str = "ffmpeg", face_scale: float = 0.25, 
                      margin_x: int = 30, margin_y: int = 30,
                      face_fixed: bool = False,
                      face_x: int = 1440,
                      face_y: int = 810,
                      face_w: int = 480,
                      face_h: int = 270,
                      face_keep_ar: bool = True) -> None:
    """
    Overlay face_video onto bg_video at bottom-right corner using ffmpeg
    
    Args:
        bg_video: Background video path
        face_video: Face video path  
        output_video: Output video path
        ffmpeg_cmd: FFmpeg command
        face_scale: Face video scale ratio (0.25 = 25% of background width)
        margin_x: Right margin (pixels)
        margin_y: Bottom margin (pixels)
    """
    if face_fixed:
        if face_keep_ar:
            ratio = face_w / face_h if face_h != 0 else 1.0
            filter_complex = (
                f"[1:v]scale='if(gt(a,{ratio}),{face_w},-1)':'if(gt(a,{ratio}),-1,{face_h})'[face];"
                f"[0:v][face]overlay='{face_x}+( {face_w}-overlay_w)/2':'{face_y}+( {face_h}-overlay_h)/2':shortest=1"
            )
        else:
            filter_complex = (
                f"[1:v]scale={face_w}:{face_h}[face];"
                f"[0:v][face]overlay={face_x}:{face_y}:shortest=1"
            )
    else:
        filter_complex = (
            f'[1:v]scale=iw*{face_scale}:-1[face];'
            f'[0:v][face]overlay=main_w-overlay_w-{margin_x}:main_h-overlay_h-{margin_y}:shortest=1'
        )
    
    cmd = [
        ffmpeg_cmd, '-y',
        '-i', bg_video,
        '-i', face_video,
        '-filter_complex', filter_complex,
        '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        output_video
    ]
    
    logging.info(f'Overlaying face video to bottom-right (scale {face_scale*100:.0f}%, margin {margin_x}x{margin_y}): {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def combine_video_chunks(video_files: List[str], output_file: str, 
                        ffmpeg_cmd: str = "ffmpeg") -> None:
    """Merge multiple video chunks into one complete video using ffmpeg concat mode"""
    list_file = "video_list.txt"
    try:
        with open(list_file, "w") as f:
            for video in video_files:
                f.write(f"file '{os.path.abspath(video)}'\n")
        
        cmd = [
            ffmpeg_cmd, '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_file
        ]
        
        logging.info(f'Merging video chunks: {" ".join(cmd)}')
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)


def find_html_files(html_dir: str) -> List[Tuple[int, str]]:
    """Find and sort HTML files"""
    html_files = []
    
    for filename in os.listdir(html_dir):
        if filename.endswith('.html'):
            match = re.search(r'(\d+)', filename)
            if match:
                slide_num = int(match.group(1))
                file_path = os.path.join(html_dir, filename)
                html_files.append((slide_num, file_path))
                logging.info(f"Found HTML {slide_num}: {filename}")
    
    html_files.sort(key=lambda x: x[0])
    
    if not html_files:
        logging.error(f"No HTML files found in {html_dir}")
        return []
    
    logging.info(f"Found {len(html_files)} HTML files")
    return html_files


def find_generated_files(output_dir: str, slide_num: int) -> Tuple[Optional[str], Optional[str]]:
    """Find generated TTS and face video files, prioritize audio/ and face/ subdirectories, fallback to old paths"""
    audio_dir = os.path.join(output_dir, "audio")
    face_dir = os.path.join(output_dir, "face")

    tts_candidates = [
        os.path.join(audio_dir, f"slide_{slide_num}_tts.wav"),
        os.path.join(output_dir, f"slide_{slide_num}_tts.wav"),
    ]
    face_candidates = [
        os.path.join(face_dir, f"slide_{slide_num}_face.mp4"),
        os.path.join(output_dir, f"slide_{slide_num}_face.mp4"),
    ]

    tts_file = next((p for p in tts_candidates if os.path.exists(p)), None)
    face_file = next((p for p in face_candidates if os.path.exists(p)), None)

    return (tts_file, face_file)


async def process_html_folder(html_dir: str, output_dir: str, final_video: str,
                            ffmpeg_cmd: str = "ffmpeg", face_scale: float = 0.25,
                            margin_x: int = 30, margin_y: int = 30, 
                            # 已不使用 quality，固定 1920x1080 捕获
                            pad_ratio_w: float = 0.0,
                            pad_ratio_h: float = 0.0,
                            pad_side: str = "right-bottom",
                            pad_color: str = "#000000",
                            frame_width: int = 1900,
                            frame_height: int = 1000,
                            face_fixed: bool = True,
                            face_x: int = 1440,
                            face_y: int = 810,
                            face_w: int = 480,
                            face_h: int = 270,
                            include_face: bool = True) -> None:
    """处理整个HTML文件夹"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    temp_video_dir = os.path.join(output_dir, "temp_videos")
    os.makedirs(temp_video_dir, exist_ok=True)
    
    # 查找HTML文件
    html_files = find_html_files(html_dir)
    if not html_files:
        raise ValueError("No HTML files found")
    
    final_video_chunks = []
    
    for idx, (slide_num, html_file) in enumerate(html_files):
        target_id = idx + 1
        logging.info(f"Processing HTML slide file_num={slide_num} mapped_id={target_id}: {os.path.basename(html_file)}")
        
        try:
            effective_width, effective_height = (1400, 1000) if not include_face else (frame_width, frame_height)
            size = (effective_width, effective_height)
                
            image = await _render_html_file_to_image_async(
                html_file, 
                f"slide_{slide_num}",
                size
            )
            
            if not image:
                logging.error(f"Failed to convert HTML to PIL Image for slide {slide_num}")
                continue
            
            tts_file, face_file = find_generated_files(output_dir, target_id)
            if (not tts_file or (include_face and not face_file)) and slide_num is not None:
                fallback_id = slide_num
                alt_tts, alt_face = find_generated_files(output_dir, fallback_id)
                if alt_tts and (alt_face or not include_face):
                    logging.warning(f"Primary ID {target_id} missing media. Falling back to ID {fallback_id}.")
                    tts_file, face_file = alt_tts, alt_face

            if not tts_file:
                logging.error(f"Missing TTS ({tts_file}) for slide {slide_num}")
                continue
            if include_face and not face_file:
                logging.error(f"Missing face video ({face_file}) for slide {slide_num}")
                continue
            
            padded_image = _pad_image_canvas(image, pad_ratio_w, pad_ratio_h, pad_side, pad_color)
            temp_image_path = _save_image_temp(padded_image, f"slide_{target_id}")

            final_slide_video = os.path.join(temp_video_dir, f"slide_{target_id}_final.mp4")
            if include_face:
                compose_slide_video_with_face(temp_image_path, face_file, final_slide_video, ffmpeg_cmd,
                                              face_scale=face_scale, margin_x=margin_x, margin_y=margin_y)
            else:
                logging.info(f"Compositing PNG with TTS audio only for slide {slide_num}")
                compose_slide_video_png_with_audio(temp_image_path, tts_file, final_slide_video, ffmpeg_cmd)

            final_video_chunks.append(final_slide_video)
            
            try:
                os.remove(temp_image_path)
            except Exception:
                pass
            
            logging.info(f"✅ Slide {slide_num} completed")
            
        except Exception as e:
            logging.error(f"Error processing slide {slide_num}: {e}")
            continue
    
    if final_video_chunks:
        logging.info(f"Merging {len(final_video_chunks)} video chunks into final video")
        combine_video_chunks(final_video_chunks, final_video, ffmpeg_cmd)
        
        for temp_video in final_video_chunks:
            try:
                os.remove(temp_video)
            except Exception:
                pass
        
        try:
            os.rmdir(temp_video_dir)
        except Exception:
            pass
        
        logging.info(f"✅ Final video created: {final_video}")
    else:
        logging.error("No video chunks were created")
        raise ValueError("Failed to create any video chunks")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert HTML folder to complete video with face overlay")
    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files")
    parser.add_argument("--output-dir", required=True, help="Directory containing generated TTS and face videos")
    parser.add_argument("--final-video", required=True, help="Output final video file path")
    parser.add_argument("--ffmpeg-cmd", default="ffmpeg", help="FFmpeg command path")
    
    parser.add_argument("--face-scale", type=float, default=0.25, 
                       help="Face video scale ratio (0.25 = 25%% of background width)")
    parser.add_argument("--margin-x", type=int, default=30,
                       help="Right margin in pixels (default: 30)")
    parser.add_argument("--margin-y", type=int, default=30,
                       help="Bottom margin in pixels (default: 30)")
    
    parser.add_argument("--pad-ratio-w", type=float, default=0.0,
                       help="Extra width ratio to pad canvas (e.g., 0.06 = +6% width)")
    parser.add_argument("--pad-ratio-h", type=float, default=0.0,
                       help="Extra height ratio to pad canvas (e.g., 0.06 = +6% height)")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--face-fixed", action='store_true', default=True)
    parser.add_argument("--face-x", type=int, default=1440)
    parser.add_argument("--face-y", type=int, default=810)
    parser.add_argument("--face-w", type=int, default=480)
    parser.add_argument("--face-h", type=int, default=270)
    parser.add_argument("--pad-side", choices=["right-bottom", "right", "bottom", "all"], default="right-bottom",
                       help="Which side to pad the canvas")
    parser.add_argument("--pad-color", default="#000000",
                       help="Canvas pad color (e.g., #000000 or 'white')")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if not os.path.isdir(args.html_dir):
        logging.error(f"HTML directory does not exist: {args.html_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.output_dir):
        logging.error(f"Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    final_video_dir = os.path.dirname(os.path.abspath(args.final_video))
    os.makedirs(final_video_dir, exist_ok=True)
    
    try:
        asyncio.run(process_html_folder(
            args.html_dir,
            args.output_dir,
            args.final_video,
            args.ffmpeg_cmd,
            args.face_scale,
            args.margin_x,
            args.margin_y,
            args.pad_ratio_w,
            args.pad_ratio_h,
            args.pad_side,
            args.pad_color,
            args.frame_width,
            args.frame_height,
            args.face_fixed,
            args.face_x,
            args.face_y,
            args.face_w,
            args.face_h
        ))
        
        if os.path.exists(args.final_video):
            file_size = os.path.getsize(args.final_video) / (1024 * 1024)
            logging.info(f"✅ Successfully created final video: {args.final_video} ({file_size:.1f} MB)")
            
            try:
                info_cmd = [args.ffmpeg_cmd, '-i', args.final_video]
                result = subprocess.run(info_cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
                for line in result.stdout.split('\n'):
                    if 'Duration:' in line or 'Video:' in line or 'Audio:' in line:
                        logging.info(f"Video info: {line.strip()}")
            except Exception:
                pass
        else:
            logging.error("Final video was not created")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
