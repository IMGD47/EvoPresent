#!/usr/bin/env python3
"""
TTS and face video generator
Generates text-to-speech audio and face videos, without video composition
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import logging
import functools
import tempfile
import subprocess
import wave
from typing import Optional, Dict, List, Any

import cv2
import numpy as np
import torch
import torchvision
import librosa
import face_alignment
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
from transformers import Wav2Vec2FeatureExtractor

from video_generation.models.float.FLOAT import FLOAT
from video_generation.options.base_options import BaseOptions

def _generate_tts_audio_megatts3_sync(
    text: str,
    output_path: str,
    voice_wav: Optional[str] = None,
    megatts3_root: Optional[str] = None,
) -> None:
    """MegaTTS3 TTS generation (synchronous version)"""

    base_dir = os.path.dirname(__file__)
    tts_root = os.path.abspath(megatts3_root) if megatts3_root else os.path.join(base_dir, "MegaTTS3")
    if os.path.isdir(os.path.join(tts_root, "tts")):
        if tts_root not in sys.path:
            sys.path.append(tts_root)
    else:
        raise ImportError(f"MegaTTS3 'tts' package not found at {tts_root}")

    global _MEGATTS3_SINGLETONS, _MEGATTS3_LOCK
    try:
        _MEGATTS3_SINGLETONS  # type: ignore[name-defined]
    except NameError:
        _MEGATTS3_SINGLETONS = {}  # type: ignore[var-annotated]
        _MEGATTS3_LOCK = threading.Lock()  # type: ignore[var-annotated]

    key = (tts_root, os.path.abspath(voice_wav) if voice_wav else os.path.join(tts_root, "assets", "English_prompt.wav"))

    with _MEGATTS3_LOCK:  # type: ignore[name-defined]
        singleton = _MEGATTS3_SINGLETONS.get(key)  # type: ignore[index]
        if singleton is None:
            from tts.infer_cli import MegaTTS3DiTInfer  # type: ignore
            from tts.utils.audio_utils.io import save_wav as _save_wav  # type: ignore

            infer = MegaTTS3DiTInfer(ckpt_root=os.path.join(tts_root, "checkpoints"))
            prompt_audio_path = key[1]
            with open(prompt_audio_path, 'rb') as f:
                audio_bytes = f.read()
            latent_file = None
            potential_npy = os.path.splitext(prompt_audio_path)[0] + '.npy'
            if os.path.isfile(potential_npy):
                latent_file = potential_npy
            resource_context = infer.preprocess(audio_bytes, latent_file)
            try:
                if hasattr(infer, "load_profiles"):
                    infer.load_profiles()
            except Exception:
                pass
            singleton = {
                "infer": infer,
                "resource_context": resource_context,
                "save_wav": _save_wav,
                "infer_lock": threading.Lock(),
            }
            _MEGATTS3_SINGLETONS[key] = singleton  # type: ignore[index]

    infer = singleton["infer"]
    resource_context = singleton["resource_context"]
    save_wav = singleton["save_wav"]

    lock = singleton.get("infer_lock")
    if lock is None:
        lock = threading.Lock()
        singleton["infer_lock"] = lock  # type: ignore[index]

    with lock:
        try:
            wav_bytes = infer.forward(
                resource_context,
                text,
                time_step=32,
                p_w=1.6,
                t_w=2.5
            )
        except Exception as e:
            msg = str(e)
            if "Need to load profiles" in msg and hasattr(infer, "load_profiles"):
                try:
                    infer.load_profiles()  # type: ignore[attr-defined]
                    wav_bytes = infer.forward(
                        resource_context,
                        text,
                        time_step=32,
                        p_w=1.6,
                        t_w=2.5
                    )
                except Exception:
                    raise
            else:
                raise
        save_wav(wav_bytes, output_path)


async def generate_tts_audio(
    text: str,
    output_path: str,
    voice_wav: Optional[str] = None,
    tts_backend: str = "megatts3",
    openai_voice: Optional[str] = None,
    openai_speed: Optional[float] = None,
    openai_api_key: Optional[str] = None,
    megatts3_root: Optional[str] = None,
) -> None:
    """Asynchronous TTS audio generation"""
    try:
        if tts_backend == "openai":
            api_key = openai_api_key
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not provided (pass --openai-api-key or set env var)")
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError("openai package not installed: pip install openai") from e
            client = OpenAI(api_key=api_key)
            voice = openai_voice or "echo"
            speed = openai_speed or 1.0
            resp = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed,
                response_format="wav",
            )
            with open(output_path, "wb") as f:
                f.write(resp.content)
            print(f"TTS(OPENAI): saved {output_path} ({os.path.getsize(output_path)} bytes), voice={voice}, speed={speed}")
            return
        
        voice_wav_path = voice_wav or os.path.join(os.path.abspath(megatts3_root or os.path.join(os.path.dirname(__file__), 'MegaTTS3')), 'assets', 'English_prompt.wav')
        print(f"TTS(MegaTTS3): using voice_wav={voice_wav_path}")
        
        if not os.path.exists(voice_wav_path):
            raise FileNotFoundError(f"Voice reference file not found: {voice_wav_path}")
            
        try:
            with wave.open(voice_wav_path, 'rb') as wav_file:
                wav_file.getparams()
        except Exception as e:
            print(f"Warning: Could not validate voice reference file: {e}")
            voice_wav_path = os.path.join(os.path.dirname(__file__), 'MegaTTS3', 'assets', 'English_prompt.wav')
            print(f"Falling back to default voice: {voice_wav_path}")
            
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _generate_tts_audio_megatts3_sync, text, output_path, voice_wav_path, megatts3_root)
        print(f"TTS(MegaTTS3): saved {output_path} ({os.path.getsize(output_path)} bytes)")
    except Exception as e:
        print(f"TTS ERROR: {e}. Falling back to silence.")
        sample_rate = 22050
        words = len(text.split())
        duration = max(2.0, words * 0.3)
        samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
        print(f"TTS: wrote silent wav {output_path} ({os.path.getsize(output_path)} bytes)")


class DataProcessor:
    """Data preprocessor"""
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device='cpu',
            flip_input=False
        )
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path,
                                                                             local_files_only=True)
        self.transform = A.Compose([
            A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A_pytorch.ToTensorV2(),
        ])
        
        self._face_cache = {}
        self._audio_cache = {}

    @torch.no_grad()
    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Process face image"""
        mult = 360. / img.shape[0]
        resized_img = cv2.resize(img, dsize=(0, 0), fx=mult, fy=mult,
                                 interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
                  for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0]
        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)
        bs = int(max(bsy, bsx) * 1.6)
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs
        crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
        crop_img = cv2.resize(crop_img, dsize=(self.input_size, self.input_size),
                              interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return crop_img

    def default_img_loader(self, path) -> np.ndarray:
        """Default image loader"""
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        """Default audio loader"""
        speech_array, sr = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(speech_array, sampling_rate=sr, return_tensors='pt').input_values[0]

    def preprocess(self, ref_path: str, audio_path: str, no_crop: bool) -> dict:
        """Preprocess face image and audio"""
        cache_key = f"{ref_path}_{no_crop}"
        if cache_key in self._face_cache:
            s = self._face_cache[cache_key]
        else:
            s = self.default_img_loader(ref_path)
            if not no_crop:
                s = self.process_img(s)
            s = self.transform(image=s)['image'].unsqueeze(0)
            self._face_cache[cache_key] = s
        
        if audio_path in self._audio_cache:
            a = self._audio_cache[audio_path]
        else:
            a = self.default_aud_loader(audio_path).unsqueeze(0)
            self._audio_cache[audio_path] = a
            
        return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
    """Face video inference agent"""
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()
        self.data_processor = DataProcessor(opt)

    def load_model(self) -> None:
        """Load FLOAT model"""
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        """Load model weights"""
        checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model_state = self.G.state_dict()
        filtered_state = {
            k: v for k, v in checkpoint_state.items()
            if (k in model_state and model_state[k].shape == v.shape)
        }
        self.G.load_state_dict(filtered_state, strict=False)
        try:
            num_loaded = len(filtered_state)
            num_total = len(model_state)
            logging.info(f"Loaded {num_loaded}/{num_total} parameters from checkpoint (shape-matched only)")
        except Exception:
            pass
        del checkpoint_state

    def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
        """Save generated face video"""
        vid = vid_target_recon.permute(0, 2, 3, 1)
        vid = vid.detach().clamp(-1, 1).cpu()
        vid = ((vid + 1) / 2 * 255).type(torch.ByteTensor)
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        if audio_path is not None:
            temp_video = video_path + '.temp.mp4'
            torchvision.io.write_video(temp_video, vid, fps=self.opt.fps)
            
            command = f"ffmpeg -i {temp_video} -i {audio_path} -c:v copy -c:a aac -shortest -map 0:v:0 -map 1:a:0 {video_path} -y"
            try:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                logging.info(f"FFmpeg merge successful for {video_path}")
                os.remove(temp_video)
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg error: {e.stderr}")
                os.rename(temp_video, video_path)
                logging.warning(f"Keeping video without audio: {video_path}")
        else:
            torchvision.io.write_video(video_path, vid, fps=self.opt.fps)
        
        if os.path.exists(video_path):
            logging.info(f"Successfully saved video to {video_path}")
        else:
            logging.error(f"Failed to save video to {video_path}")
            
        return video_path

    @torch.no_grad()
    def run_inference(self, res_video_path: str, ref_path: str, audio_path: str,
                      a_cfg_scale: float = 2.0, r_cfg_scale: float = 1.0,
                      e_cfg_scale: float = 1.0, emo: str = 'neutral',
                      nfe: int = 10, no_crop: bool = False, seed: int = 25,
                      verbose: bool = False) -> str:
        """Run face video generation inference"""
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        if verbose:
            print("> [Done] Preprocess.")
        d_hat = self.G.inference(
            data=data,
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            emo=emo,
            nfe=nfe,
            seed=seed
        )['d_hat']
        res_video_path = self.save_video(d_hat, res_video_path, audio_path)
        if verbose:
            print(f"> [Done] result saved at {res_video_path}")
        return res_video_path


def _ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


async def generate_tts_and_face_videos(
    script_json: str,
    ref_face: str,
    ckpt_path: str,
    output_dir: str,
    voice_wav: Optional[str] = None,
    tts_backend: str = "megatts3",
    openai_voice: Optional[str] = None,
    openai_speed: Optional[float] = None,
    openai_api_key: Optional[str] = None,
    megatts3_root: Optional[str] = None,
    fps: int = 30,
    sampling_rate: int = 16000,
    input_size: int = 512,
    wav2vec_model_path: str = "facebook/wav2vec2-base-960h",
    a_cfg_scale: float = 2.0,
    r_cfg_scale: float = 1.2,
    e_cfg_scale: float = 1.0,
    nfe: int = 10,
    seed: int = 25,
    emo: str = "neutral",
    no_crop_face: bool = False,
    face_gpu: Optional[str] = None,
    tts_concurrency: int = 4,
    face_concurrency: Optional[int] = None,
    include_face: bool = True,
    max_pages: Optional[int] = None,
) -> None:
    """
    Generate TTS audio and face videos
    
    Args:
        script_json: Script JSON file path
        ref_face: Reference face image path
        ckpt_path: FLOAT model checkpoint path
        output_dir: Output directory
        Other parameters: Various configuration parameters for TTS and face generation
    """
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    _ensure_dir(output_dir)
    audio_dir = os.path.join(output_dir, "audio")
    face_dir = os.path.join(output_dir, "face")
    _ensure_dir(audio_dir)
    _ensure_dir(face_dir)

    with open(script_json, "r", encoding="utf-8") as f:
        data: Any = json.load(f)
    
    scripts_by_key: Dict[str, str] = {}
    if isinstance(data, dict) and "slides" in data:
        for slide in data["slides"]:
            if isinstance(slide, dict):
                slide_id = slide.get("id") or slide.get("slide_id")
                if slide_id is None:
                    continue
                    
                script_text = None
                if "script" in slide:
                    script_text = slide["script"]
                elif "content" in slide and isinstance(slide["content"], dict):
                    script_text = slide["content"].get("script")
                
                if script_text and script_text.strip():
                    scripts_by_key[str(slide_id)] = script_text.strip()
    else:
        raise ValueError("script_json must have 'slides' array")

    face_agent = None
    if include_face:
        class SimpleOptions:
            def __init__(self):
                parser = argparse.ArgumentParser(add_help=False)
                parser = BaseOptions().initialize(parser)
                for action in parser._actions:
                    if action.dest and action.dest not in (argparse.SUPPRESS,):
                        setattr(self, action.dest, action.default)

                self.ref_path = ref_face
                self.ckpt_path = ckpt_path
                self.fps = fps
                self.sampling_rate = sampling_rate
                self.input_size = input_size
                self.wav2vec_model_path = wav2vec_model_path
                self.a_cfg_scale = a_cfg_scale
                self.r_cfg_scale = r_cfg_scale
                self.e_cfg_scale = e_cfg_scale
                self.nfe = nfe
                self.seed = seed
                self.emo = emo
                self.no_crop = no_crop_face

                if not hasattr(self, 'rank'):
                    self.rank = 0
                if not hasattr(self, 'ngpus'):
                    self.ngpus = 1

        face_opts = SimpleOptions()

        if face_gpu is not None and str(face_gpu).strip() != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(face_gpu).strip()
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        ngpus = len([g for g in visible_gpus.split(",") if g != ""]) if visible_gpus else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
        if ngpus <= 0:
            ngpus = 1
        face_opts.rank, face_opts.ngpus = 0, ngpus
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
        face_agent = InferenceAgent(face_opts)

    seq_mode = (tts_backend == "megatts3")
    if seq_mode:
        tts_concurrency = 1
        if include_face:
            face_concurrency = 1
        logging.info("Sequential mode enabled for MegaTTS3: force TTS and face to run one-by-one")

    tts_sem = asyncio.Semaphore(max(1, int(tts_concurrency)))
    if include_face:
        face_workers = max(1, int(face_concurrency)) if face_concurrency else 1
        face_sem = asyncio.Semaphore(face_workers)
        logging.info(f"TTS concurrency: {tts_concurrency}, Face concurrency: {face_workers}")
    else:
        face_sem = None  # type: ignore[assignment]
        logging.info(f"TTS concurrency: {tts_concurrency}, Face generation: DISABLED")

    async def _process_slide(slide_id: str, script_text: str) -> None:
        """Process single slide"""
        logging.info(f"Processing slide {slide_id}: {script_text[:50]}...")
        
        output_tts_path = os.path.join(audio_dir, f"slide_{slide_id}_tts.wav")
        output_face_path = os.path.join(face_dir, f"slide_{slide_id}_face.mp4")
        
        if os.path.exists(output_tts_path):
            logging.info(f"TTS already exists for slide {slide_id}: {output_tts_path}")
            tts_path = output_tts_path
        else:
            async with tts_sem:
                await generate_tts_audio(
                    script_text,
                    output_tts_path,
                    voice_wav=voice_wav,
                    tts_backend=tts_backend,
                    openai_voice=openai_voice,
                    openai_speed=openai_speed,
                    openai_api_key=openai_api_key,
                    megatts3_root=megatts3_root,
                )
            tts_path = output_tts_path

        if include_face:
            if os.path.exists(output_face_path):
                logging.info(f"Face video already exists for slide {slide_id}: {output_face_path}")
            else:
                logging.info(f"Generating face video for slide {slide_id}: {output_face_path}")
                async with face_sem:  # type: ignore[arg-type]
                    loop = asyncio.get_running_loop()
                    fn = functools.partial(
                        face_agent.run_inference,  # type: ignore[union-attr]
                        res_video_path=output_face_path,
                        ref_path=ref_face,
                        audio_path=tts_path,
                        a_cfg_scale=face_agent.opt.a_cfg_scale,  # type: ignore[attr-defined]
                        r_cfg_scale=face_agent.opt.r_cfg_scale,  # type: ignore[attr-defined]
                        e_cfg_scale=face_agent.opt.e_cfg_scale,  # type: ignore[attr-defined]
                        emo=face_agent.opt.emo,  # type: ignore[attr-defined]
                        nfe=face_agent.opt.nfe,  # type: ignore[attr-defined]
                        no_crop=face_agent.opt.no_crop,  # type: ignore[attr-defined]
                        seed=face_agent.opt.seed,  # type: ignore[attr-defined]
                        verbose=True,
                    )
                    await loop.run_in_executor(None, fn)

            logging.info(f"✅ Slide {slide_id} completed: TTS -> {output_tts_path}, Face -> {output_face_path}")
        else:
            logging.info(f"✅ Slide {slide_id} completed: TTS -> {output_tts_path} (face disabled)")

    sorted_slides = sorted([(int(k), v) for k, v in scripts_by_key.items() if v.strip()], key=lambda x: x[0])
    
    if max_pages is not None:
        sorted_slides = sorted_slides[:max_pages]
        logging.info(f"Processing first {max_pages} slides (total available: {len(scripts_by_key)})")
    
    if seq_mode:
        for slide_num, script_text in sorted_slides:
            await _process_slide(str(slide_num), script_text.strip())
    else:
        tasks = []
        for slide_num, script_text in sorted_slides:
            tasks.append(_process_slide(str(slide_num), script_text.strip()))
        await asyncio.gather(*tasks)
    
    processed_count = len(sorted_slides)
    total_count = len(scripts_by_key)
    if max_pages is not None:
        logging.info(f"🎉 Processed {processed_count}/{total_count} slides (limited by max_pages={max_pages})")
    else:
        logging.info(f"🎉 All {processed_count} slides processed successfully!")


SPEED_PRESETS = {
    "fast": {"nfe": 4, "a_cfg_scale": 1.5, "r_cfg_scale": 1.0, "e_cfg_scale": 0.8},
    "balanced": {"nfe": 6, "a_cfg_scale": 2.0, "r_cfg_scale": 1.2, "e_cfg_scale": 1.0}, 
    "quality": {"nfe": 10, "a_cfg_scale": 2.0, "r_cfg_scale": 1.2, "e_cfg_scale": 1.0}
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Generate TTS audio and face videos from script JSON.")
    
    parser.add_argument("--script-json", required=True, help="JSON with scripts per slide")
    parser.add_argument("--ref-face", required=True, help="Reference face image path")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated files")
    parser.add_argument("--ckpt-path", default="./checkpoints/float.pth", help="FLOAT model checkpoint path")
    
    parser.add_argument("--tts-backend", choices=["megatts3", "openai"], default="megatts3", help="TTS backend to use")
    parser.add_argument("--voice-wav", help="Reference voice WAV for MegaTTS3")
    parser.add_argument("--megatts3-root", help="MegaTTS3 root directory")
    parser.add_argument("--openai-voice", help="OpenAI TTS voice")
    parser.add_argument("--openai-speed", type=float, help="OpenAI TTS speed")
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--speed-preset", choices=["fast", "balanced", "quality"], help="Speed/quality preset")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Audio sampling rate")
    parser.add_argument("--input-size", type=int, default=512, help="Face image input size")
    parser.add_argument("--wav2vec-model-path", default="/data/chengzhi/chengzhi/iclr26/Agent/code/video_generation/checkpoints/wav2vec2-base-960h/", help="Wav2Vec2 model path")
    parser.add_argument("--a-cfg-scale", type=float, default=1.7, help="Audio condition scale")
    parser.add_argument("--r-cfg-scale", type=float, default=1.6, help="Face condition scale")
    parser.add_argument("--e-cfg-scale", type=float, default=0.8, help="Emotion condition scale")
    parser.add_argument("--nfe", type=int, default=5, help="NFE parameter")
    parser.add_argument("--seed", type=int, default=25, help="Random seed")
    parser.add_argument("--emo", default="neutral", choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], help="Emotion")
    parser.add_argument("--no-crop-face", action="store_true", help="Don't crop face")
    
    parser.add_argument("--face-gpu", help="CUDA device id(s) for face generation")
    parser.add_argument("--tts-concurrency", type=int, default=4, help="Max concurrent TTS tasks")
    parser.add_argument("--face-concurrency", type=int, help="Max concurrent face generation tasks")
    
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages/slides to process (default: process all)")
    
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = parse_args()
    
    if args.speed_preset:
        preset = SPEED_PRESETS[args.speed_preset]
        args.nfe = preset["nfe"]
        args.a_cfg_scale = preset["a_cfg_scale"]
        args.r_cfg_scale = preset["r_cfg_scale"]
        args.e_cfg_scale = preset["e_cfg_scale"]
        print(f"Applied {args.speed_preset} preset: nfe={args.nfe}, a_cfg={args.a_cfg_scale}, r_cfg={args.r_cfg_scale}")
    
    asyncio.run(
        generate_tts_and_face_videos(
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
            max_pages=args.max_pages,
        )
    )


if __name__ == "__main__":
    main()
