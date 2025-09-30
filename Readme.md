# *Presenting a Paper is an Art*: Self-Improvement Aesthetic Agents for Academic Presentations

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

## Acknowledgement
We appreciate the releasing codes and data of [open-r1](https://github.com/huggingface/open-r1), [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) and [SlideAudit](https://github.com/zhuohaouw/SlideAudit).

## Citation

```bibtex
@article{liu2025presenting,
  title={Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations},
  author={Liu, Chengzhi and Yang, Yuzhe and Zhou, Kaiwen and Zhang, Zhen and Fan, Yue and Xie, Yanan and Qi, Peng and Wang, Xin Eric},
  journal={arXiv preprint},
  year={2025}
}
```
