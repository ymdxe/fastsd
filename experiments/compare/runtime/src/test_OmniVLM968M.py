from transformers import AutoTokenizer, AutoModelForVision2Seq
from PIL import Image
import torch

# 加载模型路径（你已经下载好了）
model_dir = "/home/jianhongbai/gyq/FastSD/OmniVLM-968M"

# 加载 tokenizer 和模型（可以使用 float16 以节省内存）
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForVision2Seq.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")

# 加载图片
image = Image.open("picture.png").convert("RGB")

# 构造 prompt
prompt = "Describe the image."

# 编码文本和图像
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
inputs["pixel_values"] = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(model.device)

# 生成结果
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 解码输出
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("answer:", output)

