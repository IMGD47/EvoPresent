# *Presenting a Paper is an Art*: Self-Improvement Aesthetic Agents for Academic Presentations

<a href='https://arxiv.org/abs/2510.05571'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://evopresent.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/TobyYang7/EvoPresent'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a> <a href='https://huggingface.co/LCZZZZ/PresAesth'><img src='https://img.shields.io/badge/🤗-Model-purple'></a> <a href='https://evopresent.github.io/'><img src='https://img.shields.io/badge/Demo-Live-orange'></a> <a href="https://x.com/xwang_lk/status/1975917585175642496" target="_blank"><img alt="X (formerly Twitter) URL" src="https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2F_akhaliq%2Fstatus%2F1927721150584390129"></a> <a href='https://mp.weixin.qq.com/s/_UvfTWG2Ub03XWDL7KchgA'><img src='https://img.shields.io/badge/Wexin-Blog-blue'></a>
    
 
 ## 💡 Update
- [X] Thanks QbitAI (量子位) for sharing our project [[link]](https://mp.weixin.qq.com/s/_UvfTWG2Ub03XWDL7KchgA)! 🎉
- [X] Official release of our paper and demo! 🎉
- [X] Training setup and the aesthetic model have been made available.
- [X] Launch of the self-improvement aesthetic agent this week.
  

 ## 🔥 Demo

https://github.com/user-attachments/assets/b614dd22-48cf-41b7-b5b6-7fbc4cad5078

Check out more at [🎨 project web](https://evopresent.github.io/).


## 🚀  EvoPresent Agent Pipeline

![Pipeline](asset/pipeline.png)

Overview of the EvoPresent framework. (a) EvoPresent first performs content extraction and voice generation, then constructs the storyline and script, followed by content enhancement using image generation and knowledge retrieval. Design and rendering are handled next, and the aesthetic checker evaluates the initial slide and provides adjustments. (b) PresAesth is trained on a human-preference aesthetic dataset via multiple tasks (scoring, defect adjustment, and comparison). (c) The PresAesth model guides the agent framework in iterative self-improvement.

## 🛠️  Installation

**Environment**
```bash
conda create -n evopresent python=3.10
pip install -r Evopresent/requirements.txt
python -m playwright install --with-deps chromium
```

**API Installation**

To balance generation speed and quality, the recommended model combinations are:
- For text and image extraction: `gpt-4o-2024-08-06` and `gpt-4.1-2025-04-14`
- For slide generation: `deepseek-chat/deepseek-reasoner`, `gemini-2.5-pro`, and `claude-sonnet-4-20250514`
  
Recommended temperature for `evopresent/ppt/gen_pptx_code.py`: 0.6–0.8. Adjust accordingly for different models.
 
```
export EVOP_DEEPSEEK_API_KEY=
export OPENAI_API_KEY=
export EVOP_GEMINI_API_KEY=
export EVOP_CLAUDE_API_KEY=
```

## ⚡ Quick Inference
-  Create a folder named `{paper_name}` under `{dataset_dir}`, and place your paper inside it as a PDF file named `paper.pdf`.
- `checker-scope` controls the scope of the layout review: `all` (Check all slides), `images` (Check only slides with images/tables), `text` (Check only text-only slides).
`style` specifies the presentation theme and visual style (such as color scheme, fonts, whitespace, animations, etc.). For example, `tech_dark` applies a dark, tech-style theme. 

- Templates in `Evopresent/evopresent/styles` offer various presentation styles. More style-based templates will be added soon for easier customization.
  
```bash
CUDA_VISIBLE_DEVICES=1 python3 -m evopresent.ppt.ppt_gen_pipeline \
  --paper_path="/root/Code/Evopresent/paper_input/paper.pdf" \
  --model_name_t="gpt-4o" \  # or gpt-4.1
  --model_name_v="gpt-4o" \  # or gpt-4.1
  --paper_name="paper" \
  --target_slides=15 \
  --style="tech_dark" \
  --checker=on \   #  You can toggle it to control speed.
  --scholar=on \   # You can toggle it to control speed.
  --checker-scope=images \   # all/images/text/
  --html-model gemini-2.5-pro \   
  --checker-model gpt-4o  \
  --checker-threshold 8.7 \   
  --checker-max-attempts 3 
```

## 🎥 Presentation Video Generation

- **Environment**
  
```
pip install -r Evopresent/generation_requirements.txt
```
For more setup instructions, refer to the [FLOAT](https://github.com/deepbrainai-research/float/tree/main) and [MegaTTS3](https://github.com/bytedance/MegaTTS3).
 
- **Input & Output Directories**:
  - `--html-dir`: Specifies the directory containing HTML files for the presentation.
  - `--script-json`: Points to a JSON file containing scripts for each slide.
  - `--ref-face`: Path to the reference face image for generating facial animations.
  - `--ckpt-path`: Specifies the model weights file for loading (e.g., `float.pth`).
  - `--output-dir`: Directory where output files like audio and videos will be stored.
  - `--final-video`: Path for saving the final composed video.
    
- **Text-to-Speech (TTS) Configuration**:
  - `--tts-backend`: Enables selection between `megatts3` or OpenAI for text-to-speech processing.
  - `--openai-api-key`: API key required if using OpenAI for TTS.
  - `--openai-voice` & `--openai-speed`: Voice selection and speed parameters for OpenAI TTS.
  - `--voice-wav`: Used as a reference WAV file when choosing `megatts3` backend for TTS.
    
 - **Video Parameters**:
   - `--frame-width` & `--frame-height`: Define the dimensions of each slide in the video.
   - `--margin-x` & `--margin-y`: Set the margins for face placement within the video frames.

   
```
python3 -m video_generation.full_pipeline \  
  --html-dir path_to_html_dir \             # slides
  --script-json path_to_script_json \       # script
  --ref-face path_to_ref_face_image \      
  --ckpt-path path_to_checkpoint \          # Path to the float.pth 
  --output-dir output_directory \          
  --final-video path_to_final_video \     
  --wav2vec-model-path path_to_wav2vec_model \ # Path to the wav2vec2-base-960h
  --tts-backend  openai  \        #   megatts3/openai
  --voice-wav /root/video_generation/MegaTTS3/assets/English_prompt.wav \    # Reference WAV file if you choose megatts3
  --openai-api-key  sk....  # Parameters to select voice and speed if you choose openai
  --openai-voice ash \
  --openai-speed 1.3  \
  --frame-width 1920 \        # Width of each slide in the video
  --frame-height 1080  \      # Height of each frame in the video
  --margin-x 100  \              # Horizontal margin for face placement
  --margin-y 100   \           # Vertical margin for face placement
```

- **Presentation Generation Options**:

 1. **OpenAI TTS**:
     - **Recommendation**: If speed is your priority, OpenAI's TTS service is recommended.
     - **Note**: Please refer to OpenAI's official documentation for detailed voice selection options.

  2. **Personalized Voice with MegaTTS3**:
     - **Voice Cloning**: Use the [MegaTTS3 Voice Cloning](https://huggingface.co/spaces/mrfakename/MegaTTS3-Voice-Cloning) tool to mimic your own voice.
     - **Preset Voices**: Explore several preset voices offered by MegaTTS3 in the [Google Drive folder](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr).

## 🏋️‍♂️ PresAesth Training

This section contains the training infrastructure for the **PresAesth** model, which is based on [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and fine-tuned for presentation aesthetics evaluation tasks.

1. **Environment Setup**
   ```bash
   # install uv first
   # pip install uv
   cd train
   bash ./setup.sh
   ```

2. **Configure Environment Variables**
   ```bash
   # Edit .env file with your API keys and configuration (for evaluation and verification)
   cp env-template .env
   ```

3. **Start Training**
   ```bash
   source train_env/bin/activate
   bash ./run.sh
   ```

4. **Evaluation & Inference**
   ```bash
   python eval.py
   python inference.py
   ```

## 📊 EvoPresent Benchmark
![data](asset/data.jpg)

We have released the evaluation dataset on huggingface. Due to potential copyright restrictions, the images cannot be redistributed directly. However, they can be accessed via the links provided in the metadata.

## 🎨 Aesthetic Comparison

![data](asset/compare.jpg)

## Acknowledgement
We appreciate the releasing codes and data of [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [Paper2Poster](https://github.com/Paper2Poster/Paper2Poster/) and [Float](https://github.com/deepbrainai-research/float), [MegaTTS3](https://github.com/bytedance/MegaTTS3).

## Citation
Please kindly cite our paper if you find this project helpful.

```bibtex
@misc{liu2025presentingpaperartselfimprovement,
      title={Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations}, 
      author={Chengzhi Liu and Yuzhe Yang and Kaiwen Zhou and Zhen Zhang and Yue Fan and Yannan Xie and Peng Qi and Xin Eric Wang},
      year={2025},
      eprint={2510.05571},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.05571}, 
}
```
