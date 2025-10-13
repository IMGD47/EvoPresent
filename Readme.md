# *Presenting a Paper is an Art*: Self-Improvement Aesthetic Agents for Academic Presentations

<a href='https://arxiv.org/abs/2510.05571'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://evopresent.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/TobyYang7/EvoPresent'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a> <a href='https://huggingface.co/LCZZZZ/PresAesth'><img src='https://img.shields.io/badge/🤗-Model-purple'></a> <a href='https://evopresent.github.io/'><img src='https://img.shields.io/badge/Demo-Live-orange'></a> <a href="https://x.com/xwang_lk/status/1975917585175642496" target="_blank"><img alt="X (formerly Twitter) URL" src="https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2F_akhaliq%2Fstatus%2F1927721150584390129"></a> <a href='https://mp.weixin.qq.com/s/A5dkmLVmcpD_8DaPhO7bwg'><img src='https://img.shields.io/badge/Wexin-Blog-blue'></a>
    

 ## 💡 Update
- [X] Official release of our paper and demo！ 🎉
- [X] Training setup has been made available. The aesthetic model weights will be released this week.
- [ ] Launch of the self-improvement aesthetic agent this week.
- [ ] Certain benchmark data has been made available; however, due to potential copyright restrictions, the full benchmark will be released at a later time.

 ## 🔥 Demo

 https://github.com/user-attachments/assets/49cd7ab8-3259-4f45-a9eb-a5f8211b9549

Check out more at [🎨 project web](https://evopresent.github.io/).


## 🚀  EvoPresent Agent Pipeline

![Pipeline](asset/pipeline.png)

Overview of the EvoPresent framework. (a) EvoPresent first performs content extraction and voice generation, then constructs the storyline and script, followed by content enhancement using image generation and knowledge retrieval. Design and rendering are handled next, and the aesthetic checker evaluates the initial slide and provides adjustments. (b) PresAesth is trained on a human-preference aesthetic dataset via multiple tasks (scoring, defect adjustment, and comparison). (c) The PresAesth model guides the agent framework in iterative self-improvement.

## 🛠️  Installation

**Environment**
```bash
pip install -r requirements.txt
python -m playwright install --with-deps chromium
```

**API Installation**
To balance generation speed and quality, the recommended model combinations are:
- For text and image extraction: `gpt-4o-2024-08-06` and `gpt-4.1-2025-04-14`
- For slide generation: `deepseek-chat/deepseek-reasoner`, `gemini-2.5-pro`, and `claude-sonnet-4-20250514`
  
Recommended temperature for `evopresent/ppt/gen_pptx_code.py`: 0.6–0.8. Adjust accordingly for different models.
 
```
export EVOP_DEEPSEEK_API_KEY='sk....'
export export OPENAI_API_KEY='sk....'
export EVOP_GEMINI_API_KEY='sk....'
export EVOP_CLAUDE_API_KEY='sk....'
```

## ⚡ Quick Inference
`checker-scope` controls the scope of the layout review: `all` (Check all slides), `images` (Check only slides with images/tables), `text` (Check only text-only slides).
`style` specifies the presentation theme and visual style (such as color scheme, fonts, whitespace, animations, etc.). For example, `tech_dark` applies a dark, tech-style theme. 

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
  --checker-model gpt-4o
```




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

## 🎨 Aesthetic Comparison

![data](asset/compare.jpg)

## Acknowledgement
We appreciate the releasing codes and data of [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [Paper2Poster](https://github.com/Paper2Poster/Paper2Poster/) and [Float](https://github.com/deepbrainai-research/float) .

## Citation

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
