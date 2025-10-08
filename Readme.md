# *Presenting a Paper is an Art*: Self-Improvement Aesthetic Agents for Academic Presentations

[![arXiv](https://img.shields.io/badge/arXiv-paper-red.svg)](https://arxiv.org/abs/2510.05571)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://evopresent.github.io/)



## TO DO 
- [X] Official release of our paper and demo !
- [X] Launch of the self-improvement aesthetic agent this week.
- [X] Training setup has been made available. The aesthetic model weights will be released this week.


## EvoPresent Agent Pipeline

![Pipeline](asset/pipeline.png)

Overview of the EvoPresent framework. (a) EvoPresent first performs content extraction and voice generation, then constructs the storyline and script, followed by content enhancement using image generation and knowledge retrieval. Design and rendering are handled next, and the aesthetic checker evaluates the initial slide and provides adjustments. (b) PresAesth is trained on a human-preference aesthetic dataset via multiple tasks (scoring, defect adjustment, and comparison). (c) The PresAesth model guides the agent framework in iterative self-improvement.

## Training

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

## EvoPresent Benchmark
![Pipeline](asset/pipeline.png)

## Acknowledgement
We appreciate the releasing codes and data of [open-r1](https://github.com/huggingface/open-r1), [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) and [SlideAudit](https://github.com/zhuohaouw/SlideAudit).

## Citation

```bibtex
@article{liu2025presenting,
  title={Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations},
  author={Liu, Chengzhi and Yang, Yuzhe and Zhou, Kaiwen and Zhang, Zhen and Fan, Yue and Xie, Yanan and Qi, Peng and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2510.05571},
  year={2025},
  url={https://arxiv.org/abs/2510.05571}
}
```
