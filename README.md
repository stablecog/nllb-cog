# Stable Diffusion Cog model

This is an implementation of the [Diffusers Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as a Cog model. [Cog packages machine learning models as standard containers](https://github.com/replicate/cog). This cog also includes [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to upscale images.

The Docker image built using this repo can be found at: [yekta/sc](https://hub.docker.com/r/yekta/sc)

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

    cog run scripts/download-weights <your-hugging-face-auth-token>

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"

Or, build a Docker image:

    cog build

Or, [push it to Replicate](https://replicate.com/docs/guides/push-a-model):

    cog push r8.im/...
