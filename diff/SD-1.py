from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention import Attention

# suppress partial model loading warning
logging.set_verbosity_error()

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
from PIL import Image
import torchvision.utils as vutils

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None
        mask = Image.open('mask2.png')

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

        mask = transform(mask)
        print("mask : ", mask.shape)
        self.attn_mask = mask.cuda()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        print(value.shape)

        # attention_mask = self.attn_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # size = int(math.sqrt(query.size(1)))
        # attention_mask = F.interpolate(attention_mask, size=(size, size)).squeeze(1)
        # attention_mask = attention_mask.reshape(query.size(0))
        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        print(self.attention_probs.shape)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        # self.store_processor = CrossAttnStoreProcessor()
        # self.store_processor1 = CrossAttnStoreProcessor()

        # print(self.unet.mid_block.attentions[0].transformer_blocks[0].attn1)
        # print(self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1)

        # self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = self.store_processor
        # self.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.processor = self.store_processor1

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.image_transforms = T.Compose(
            [
                T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(512),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])

        print(f'[INFO] loaded stable diffusion!')

    def pred_epsilon(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_eps = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_eps = (sample - (alpha_prod_t**0.5) * model_output) / (beta_prod_t**0.5)
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_eps = (beta_prod_t**0.5) * sample + (alpha_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_eps

    def save_attn_amp(self, attn_map, t, indx):
        bh, hw1, hw2 = attn_map.shape
        map_size = int(math.sqrt(hw1))
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[indx]
        b = int(bh/h)
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = (
            attn_mask.reshape(b, map_size, map_size)
            .unsqueeze(1)
            .type(attn_map.dtype)
        )
        T.ToPILImage()(attn_mask[0].cpu()).save("attn_{}.png".format(t))
        # imgs = self.decode_latents(original_latents)
        # img = T.ToPILImage()(imgs[0].cpu())
        # img.save("ori_sample_{}.png".format(str(t)))


    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # print(torch.where(attn_mask[0,1] == attn_mask[0,1].max()))

        T.ToPILImage()(attn_mask[0,1].cpu()).save("attn.png")
        imgs = self.decode_latents(original_latents)
        img = T.ToPILImage()(imgs[0].cpu())
        img.save("ori_sample_{}.png".format(str(t)))

        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)
        # degraded_latents = original_latents * attn_mask + degraded_latents * (1 - attn_mask)

        # Noise it again to match the noise level
        degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t)

        return degraded_latents, attn_mask


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        #print(dir(text_input))
        #print(text_input.data)
        #print(text_input.data.items())
        for key, value in text_input.data.items():
            print(f"{key}: {value}")

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        print("----")
        print(text_embeddings.shape)
        print(text_embeddings)

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        print("----")
        print(text_embeddings.shape)
        print(text_embeddings)

        return text_embeddings

    def pil2tensor(self, img):
        transform = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = transform(img).unsqueeze(0)
        return img

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs 

    @torch.no_grad()
    def text2image(self, input_img, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        tmp_text_embeds = self.get_text_embeds(["a cute cat"], negative_prompts)

        # Define panorama grid and get views
        latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        latent_resize = F.interpolate(latent, size=(height//16, width//16))
        # latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        # latent[:,:,height//16:,:width//16] = latent_resize
        # input image to noise

        if input_img is not None:
            input_img = Image.open(input_img).convert("RGB")
            input_img = self.pil2tensor(input_img).to(self.device)
            input_latent = self.encode_imgs(input_img)
            decode_img = self.decode_latents(input_latent)
            T.ToPILImage()(decode_img[0].cpu()).save("rec.png")
            latent = self.scheduler.add_noise(input_latent, noise=latent, timesteps=self.scheduler.timesteps[0])

        latent_imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        T.ToPILImage()(latent_imgs[0].cpu()).save("latent.png")

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):

                if i < 25:
                    input_text_embeds = tmp_text_embeds
                else:
                    input_text_embeds = text_embeds

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latent] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=input_text_embeds)['sample']

                # perform guidance
                # 7.5배 큼?
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # noise_pred = noise_pred_uncond


                # compute the denoising step with the reference model
                # step_result에는 x0와 xt-1이 저장이 됨
                step_result = self.scheduler.step(noise_pred, t, latent)
                # noise_pred: epsilon
                # latent: Xt-1??

                # uncond_attn, cond_attn = self.store_processor.attention_probs.chunk(2)
                # uncond_attn1, cond_attn1 = self.store_processor1.attention_probs.chunk(2)

                # self.save_attn_amp(cond_attn1, t, 0)

                # map_size = (8, 8)
                # degraded_latents, attn_mask = self.sag_masking(
                #     step_result['pred_original_sample'], cond_attn, map_size, t, self.pred_epsilon(latent, noise_pred_uncond, t)
                # )
                # uncond_emb, cond_emb = text_embeds.chunk(2)
                # degraded_pred = self.unet(degraded_latents, t, encoder_hidden_states=uncond_emb)['sample']
                # noise_pred += noise_pred_uncond
                # noise_pred += 1.0 * (noise_pred_uncond - degraded_pred)
                # noise_pred = degraded_pred + (noise_pred_cond - degraded_pred)
                # # take the MultiDiffusion step
                # latent = 0.75 * (noise_pred_uncond - degraded_pred)
                # print(latent.dtype)                
                latent = step_result['prev_sample']
                # print(latent.shape)
                # latent = self.scheduler.step(noise_pred, t, latent)['prev_sample']

        # Img latents -> imgs
        # latent = latent.to(torch.float32)
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--prompt', type=str, default='a photo of the dolomites')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'],
                        help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version)

    img = sd.text2image(opt.image, opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # save image
    img.save('out.png')