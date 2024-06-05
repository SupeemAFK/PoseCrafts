# PoseCrafts: Transforming Text into Dynamic 2D Characters with Openpose 🏃

## What is PoseCrafts?
PoseCrafts is a motion model that generate 2D motion in [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) format from textual description for using it to generate 2d animation spritesheets for 2d characters in [StableDiffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with [Controlnet](https://github.com/lllyasviel/ControlNet)

## Datasets 📦
Create my own dataset by using Pose Estimation Controlnet to estimate spritesheets image collected from [Sprite Resources](https://www.spriters-resource.com/) and editing its result from Pose Estimation to correcting shape and label text description in file name. I collected data around 2 weeks and I got 400 samples then I collect more data to get more 200 samples without sleeping😪. You can see the datasets here: [Openpose Motion Dataset](https://github.com/SupeemAFK/PoseCrafts/tree/main/datasets)

## Process 🧪
I use Sentence Transformer for text encoding then I used encoded text to train my model. I've experimented on many models such as Simple Dense, CNN, RNN and LSTM and from the Evaluation using MAE LSTM perform best
- Sentence Transformer
[sentence-transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Deployment🚀
- Backend
  - Deploy API using Huggingface Spaces🤗 and Docker🐋.
  - Github Repo: [PoseCrafts-API](https://github.com/SupeemAFK/PoseCrafts-API)
- Frontend
  - Fork from [https://github.com/huchenlei/sd-webui-openpose-editor](https://github.com/huchenlei/sd-webui-openpose-editor) to create demo client that can generate poses.
  - Github Repo: [sd-webui-openpose-editor](https://github.com/SupeemAFK/sd-webui-openpose-editor)

## Demo ▶️
### Text Input: "a girl walking forward"
![LSTM_walk forward girly](https://github.com/SupeemAFK/PoseCrafts/assets/83326313/5db425dc-7976-41eb-9468-69b05cec789e)
### Trying Demo: [PoseCrafts Demo Web](https://posecrafts.vercel.app/)
