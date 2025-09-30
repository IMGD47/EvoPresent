uv venv train_env
source train_env/bin/activate

# create .env file
cp env-template .env
uv pip install -e ".[dev]"

# Addtional modules
uv pip install flash-attn --no-build-isolation
uv pip install wandb==0.18.3
uv pip install tensorboardx
uv pip install qwen_vl_utils torchvision
uv pip install transformers==4.51.3
uv pip install openai
uv pip install dotenv
