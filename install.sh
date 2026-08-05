# install packages
pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install \
  transformers==4.57.6 \
  tqdm==4.67.1 \
  ipdb==0.13.13 \
  accelerate==1.12.0 \
  numpy==1.26.4 \
  shortuuid==1.0.13 \
  fschat==0.2.36 \
  fastchat==0.1.0 \
  auto_gptq==0.7.1 \
  requests==2.32.5 \
  fastapi==0.128.0 \
  uvicorn==0.40.0 \
  nvidia-ml-py3

# get fastchat for mt-bench
# git clone https://github.com/lm-sys/FastChat.git
# mv FastChat/fastchat/ fastchat/
# rm -rf FastChat

# get ouroboros
# git clone https://github.com/thunlp/Ouroboros.git
