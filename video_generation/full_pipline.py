#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import logging
import subprocess
from video_generation.tts_face_generator import generate_tts_and_face_videos
from video_generation.html_to_video_complete import process_html_folder, setup_logging

SPEED_PRESETS = {
    "fast": {"nfe": 4, "a_cfg_scale": 1.5, "r_cfg_scale": 1.0, "e_cfg_scale": 0.8},
    "balanced": {"nfe": 6, "a_cfg_scale": 2.0, "r_cfg_scale": 1.2, "e_cfg_scale": 1.0},
    "quality": {"nfe": 10, "a_cfg_scale": 2.0, "r_cfg_scale": 1.2, "e_cfg_scale": 1.0},
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline: TTS+face -> HTML composite -> final video")

    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files")
    parser.add_argument("--script-json", required=True, help="JSON with scripts per slide")
    parser.add_argument("--ref-face", required=True, help="Reference face image path")
    parser.add_argument("--output-dir", required=True, help="Working output directory (will contain audio/ and face/)")
    parser.add_argument("--final-video", required=True, help="Output final video file path")
    parser.add_argument("--ckpt-path", required=True, help="Path to the FLOAT model checkpoint (e.g., float.pth)")

    parser.add_argument("--tts-backend", choices=["megatts3", "openai"], default="megatts3", help="TTS backend")
    parser.add_argument("--voice-wav", help="Required: Reference voice WAV for MegaTTS3")
    parser.add_argument("--megatts3-root", help="Required: MegaTTS3 root directory")
    parser.add_argument("--openai-voice", default="alloy", help="OpenAI TTS voice")
    parser.add_argument("--openai-speed", type=float, default=1.0, help="OpenAI TTS speed")
    parser.add_argument("--openai-api-key", help="OpenAI API key. Best practice is to set the OPENAI_API_KEY environment variable instead.")

    parser.add_argument("--fps", type=int, default=25, help="Video FPS for generated face videos")
    parser.add_argument("--speed-preset", choices=["fast", "balanced", "quality"], help="Speed/quality preset for face gen")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Audio sampling rate")
    parser.add_argument("--input-size", type=int, default=512, help="Face image input size")
    parser.add_argument("--wav2vec-model-path", required=True, help="Path to the Wav2Vec2 model directory")
    parser.add_argument("--a-cfg-scale", type=float, default=1.8, help="Audio condition scale")
    parser.add_argument("--r-cfg-scale", type=float, default=1.6, help="Face condition scale")
    parser.add_argument("--e-cfg-scale", type=float, default=1.0, help="Emotion condition scale")
    parser.add_argument("--nfe", type=int, default=4, help="NFE parameter")
    parser.add_argument("--seed", type=int, default=25, help="Random seed")
    parser.add_argument("--emo", default="neutral", choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], help="Emotion")
    parser.add_argument("--no-crop-face", action="store_true", help="Don't crop face")

    parser.add_argument("--face-gpu", help="CUDA device id(s) for face generation")
    parser.add_argument("--tts-concurrency", type=int, default=4, help="Max concurrent TTS tasks")
    parser.add_argument("--face-concurrency", type=int, help="(Unused) Max concurrent face gen tasks; kept for API parity")
    parser.add_argument("--ffmpeg-cmd", default="ffmpeg", help="FFmpeg command path (assumes it's in the system's PATH)")

    parser.add_argument("--face-scale", type=float, default=0.25, help="Face video scale ratio (0.25 = 25% of background width)")
    parser.add_argument("--margin-x", type=int, default=30, help="Right margin in pixels")
    parser.add_argument("--margin-y", type=int, default=30, help="Bottom margin in pixels")
    parser.add_argument("--pad-ratio-w", type=float, default=0.0, help="Extra width ratio to pad canvas")
    parser.add_argument("--pad-ratio-h", type=float, default=0.0, help="Extra height ratio to pad canvas")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--face-fixed", action='store_true', default=True)
    parser.add_argument("--face-x", type=int, default=1440)
    parser.add_argument("--face-y", type=int, default=810)
    parser.add_argument("--face-w", type=int, default=480)
    parser.add_argument("--face-h", type=int, default=270)
    parser.add_argument("--pad-side", choices=["right-bottom", "right", "bottom", "all"], default="right-bottom")
    parser.add_argument("--pad-color", default="#000000")
    parser.add_argument("--no-face", action='store_true', help="Do not overlay face; output PNG+TTS only")

    return parser.parse_args()


async def run_pipeline(args: argparse.Namespace) -> None:
    setup_logging()

    if args.tts_backend == "openai":
        if not args.openai_api_key:
            args.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not args.openai_api_key:
            logging.error("OpenAI API key not found. Please provide it via the --openai-api-key argument or set the OPENAI_API_KEY environment variable.")
            sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.dirname(os.path.abspath(args.final_video))
    os.makedirs(final_dir, exist_ok=True)

    if args.speed_preset:
        preset = SPEED_PRESETS[args.speed_preset]
        args.nfe = preset["nfe"]
        args.a_cfg_scale = preset["a_cfg_scale"]
        args.r_cfg_scale = preset["r_cfg_scale"]
        args.e_cfg_scale = preset["e_cfg_scale"]
        logging.info(f"Applied {args.speed_preset} preset: nfe={args.nfe}, a_cfg={args.a_cfg_scale}, r_cfg={args.r_cfg_scale}")

    await generate_tts_and_face_videos(
        script_json=args.script_json,
        ref_face=args.ref_face,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        voice_wav=args.voice_wav,
        tts_backend=args.tts_backend,
        openai_voice=args.openai_voice,
        openai_speed=args.openai_speed,
        openai_api_key=args.openai_api_key,
        megatts3_root=args.megatts3_root,
        fps=args.fps,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        wav2vec_model_path=args.wav2vec_model_path,
        a_cfg_scale=args.a_cfg_scale,
        r_cfg_scale=args.r_cfg_scale,
        e_cfg_scale=args.e_cfg_scale,
        nfe=args.nfe,
        seed=args.seed,
        emo=args.emo,
        no_crop_face=args.no_crop_face,
        face_gpu=args.face_gpu,
        tts_concurrency=args.tts_concurrency,
        face_concurrency=args.face_concurrency,
        include_face=(not args.no_face),
    )

    await process_html_folder(
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
        args.face_h,
        include_face=(not args.no_face),
    )


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_pipeline(args))
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

    if os.path.exists(args.final_video):
        size_mb = os.path.getsize(args.final_video) / (1024 * 1024)
        logging.info(f"✅ Successfully created final video: {args.final_video} ({size_mb:.1f} MB)")
        try:
            result = subprocess.run([args.ffmpeg_cmd, '-i', args.final_video], capture_output=True, text=True, stderr=subprocess.STDOUT)
            for line in result.stdout.split('\n'):
                if 'Duration:' in line or 'Video:' in line or 'Audio:' in line:
                    logging.info(f"Video info: {line.strip()}")
        except Exception:
            pass
    else:
        logging.error("Final video not found after pipeline run")
        sys.exit(1)


if __name__ == "__main__":
    main()
