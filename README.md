# PoseCrafts: Transforming Text into 2D Characters Animation with Openpose Generation🏃

## What is PoseCrafts?
PoseCrafts is a motion model that generate 2D motion in [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) format from textual description for using it to generate 2d animation spritesheets for 2d characters in [StableDiffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with [Controlnet](https://github.com/lllyasviel/ControlNet)

## Demo [▶️](https://posecrafts.vercel.app/)
<table>
  <tr>
    <td valign="top">
      <h3>Text Input: "a girl walking forward"</h3>
      <h4>Pose generated from PoseCrafts</h4>
      <img src="https://github.com/SupeemAFK/PoseCrafts/assets/83326313/dd39c91e-bf44-4bd5-9db8-1701c60faf40"/>
      <h4>Result generated from stable diffusion</h4>
      <img src="https://github.com/SupeemAFK/PoseCrafts/assets/83326313/b0fb81f3-6fbf-4cd5-9073-9747daee3787"/>
    </td>
    <td valign="top">
      <h3>Text Input: "a girl running forward"</h3>
      <h4>Pose generated from PoseCrafts</h4>
      <img src="https://github.com/SupeemAFK/PoseCrafts/assets/83326313/2e74bcfd-f0c8-4ac8-b883-3b0d0c056be5"/>
      <h4>Result generated from stable diffusion</h4>
      <img src="https://github.com/SupeemAFK/PoseCrafts/assets/83326313/825bc72a-4fc6-4aaa-aa12-77f2791e7dd1"/>
    </td>
    <td valign="top">
      <h3>Text Input: "a girl hurting and stagger"</h3>
      <h4>Pose generated from PoseCrafts</h4>
      <img src="https://github.com/SupeemAFK/PoseCrafts/assets/83326313/98206590-7b8e-4ac4-91a7-1ff8f20bce74"/>
      <h4>Result generated from stable diffusion</h4>
      <img src="https://github.com/SupeemAFK/PoseCrafts/assets/83326313/4b48643e-cb66-45ee-9e4e-62ceaa0c14fb"/>
    </td>
  </tr>
</table>

### Trying Demo: [PoseCrafts Demo Web](https://posecrafts.vercel.app/)

### !!! Note📢 !!! 
The demo only generate openpose sequence frames if you want 2d pixel art generated like gif image above please use [StableDiffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to generate

## Datasets 📦
Create my own dataset by using Pose Estimation Controlnet to estimate spritesheets image collected from [Sprite Resources](https://www.spriters-resource.com/) and editing its result from Pose Estimation to correcting shape and label text description in file name. I collected data around 2 weeks and I got 400 samples then I collect more data to get more 200 samples without sleeping😪. You can see the datasets here: [Openpose Motion Dataset](https://github.com/SupeemAFK/PoseCrafts/tree/main/datasets)

## Process 🧪
I use Sentence Transformer for text encoding then I used encoded text to train my model. I've experimented on many models such as Simple Dense, CNN, RNN and LSTM and from the Evaluation using MAE LSTM perform best. For full process of this AI please read full medium blog below
- Sentence Transformer
[sentence-transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Full medium blog
[Medium Blog](https://medium.com/@Supeem/posecrafts-transforming-text-into-dynamic-2d-characters-with-openpose-594861900be6)

## Deployment🚀
- Backend
  - Deploy API using Huggingface Spaces🤗 and Docker🐋.
  - Github Repo: [PoseCrafts-API](https://github.com/SupeemAFK/PoseCrafts-API)
- Frontend
  - Fork from [https://github.com/huchenlei/sd-webui-openpose-editor](https://github.com/huchenlei/sd-webui-openpose-editor) to create demo client that can generate poses.
  - Github Repo: [posecrafts-sd-webui-openpose-editor](https://github.com/SupeemAFK/sd-webui-openpose-editor)
