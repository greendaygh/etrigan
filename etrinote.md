# GAN lecture in ETRI
## Installation
- Anaconda
- https://github.com/NVlabs/stylegan3
- https://github.com/clovaai/stargan-v2
- https://github.com/XingangPan/DragGAN


# GPT
프롬프트 지니 설치 (크롬 플러그인)


# cuda 설치 

- 다음 명령어 pytorch 공식사이트에서 환경에 맞게 제시 (현재 노트북, 윈도우)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

- requirement.txt, environment.yml 등으로 설치 후 (stylegan3) cuda 작동 안 될때도 재설치 해봤으나 작동 안 됨 (윈도우)

- 드라이버 설치 참고??

      sudo apt-get install nvidia-utils-535
      sudo reboot now
      sudo apt-get install nvidia-smi

# StyleGAN3

## env
- nvcc: for pytouch. it depends on env
- (base) etriai07@edwk38:~/E6001/stylegan3$ conda env create -f environment.yml

## Nvidia visualizer

- conda 설치
- git clone

      git clone https://github.com/NVlabs/stylegan3
      cd stylegan3

- conda 환경

      conda env create -f environment.yml
      conda activate stylegan3

- 데이터 다운로드 및 실행
      
      wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

      python visualizer.py stylegan2-ffhq-1024x1024.pkl

- 해상도 맞아야함 에트리 실행시 3840 x 2160 (16:9)


# StarGAN v2

- https://github.com/clovaai/stargan-v2
- 환경설정은 위 사이트 Software installation 참고
- 설치 에러시 버전 삭제 후 다시 설치

- git clone 후 개인 폴더로 변경 `etrigan`
- git init 후 git push

- core/checkpoint.py 48번째줄 수정 필요 (multigpu 지원 문제) strict=False 추가
      module.module.load_state_dict(module_dict[name], strict=False)

## 데이터 다운로드

      bash download.sh pretrained-network-celeba-hq
      bash download.sh celeba-hq-dataset
      bash download.sh wing

## 샘플링 실행

      python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 1 \
               --checkpoint_dir expr/checkpoints/celeba_hq \
               --result_dir expr/results/celeba_hq \
               --src_dir assets/representative/celeba_hq/src \
               --ref_dir assets/representative/celeba_hq/ref


## 스타일 여자를 남자로 만들기 

- utils.py 수정

      nets.style_encoder(x_ref, 1-yref)

## 코드리뷰

- solver.py의 train, compute_d_loss, compute_g_loss 리뷰

### mask 설명
- model.py 의 forward 함수
- encoder의 feature 정보를 복사해서 decoder로 바로 사용
- solver의 186 다음과 같으면

      x = self.hpf(mask * cache[x.size(2)])
      #x = x + self.hpf(mask * cache[x.size(2)]) 

- 눈 코 부분 쉐이프 정보 가지고 옴

      mask = masks[0] if x.size(2) in [32] else masks[1]
      mask = masks[0] if x.size(2) in [32] else masks[0]

- 위와 같이 masks[0]으로 하면 입까지


### Face Recognision

- https://github.com/ageitgey/face_recognition
- 얼굴에 점을 그리는 그림

- 코드 수정
- 가시화

            @torch.no_grad()
            def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
            N, C, H, W = x_src.size()
            wb = torch.ones(1, C, H, W).to(x_src.device)
            #x_src_with_wb = torch.cat([wb, x_src], dim=0)

            x_src_ptmap = nets.fan.get_pointmap(x_src).sum(dim=1, keepdim=True) 
            # x_src에서 점을 가져옴, 점 만큼의 차원을 가지고 있으나 sum으로 붙여줌?
            # x_src_ptmap.shape = (b,1,h,w) 
            # C 가 3차원이므로 이를 늘리는 방법은
            x_src_ptmap = x_src_ptmap.repeat(1,3,1,1) #(b, 3, h, w) 가 될거임. 
            x_src_with_wb = torch.cat([wb, x_src_ptmap], dim=0)



# DragGAN

- vs. diffusion 
- 환경 설정은 stylegan3와 같음
- 설치 파일이 너무 많음


# Task 2
A style-based generator architecture for Generative adversarial network

low resolution 4x4 ~ 8x8 포즈, 4개 adain layer 고정하면 포즈와 쉐이프는 바뀌지 않음

middle resolution 16x16 ~ 32x32

High ~

GAN 이 전체적인 이미지를 만들어낼 때 대부분은 low resolution에서 만들어진다

model.py의 Generator.forward 수정




# Task 3

- key point 입력 
- cycle loss 가 쓸모 없어짐
- compute g loss -> cycle-consistency loss 지우기


# hfp 

- hfp 끌 경우 원본 이미지의 전체적인 형태를 먼저 잡고 가기 때문에 형태는 거의 그대로
- hfp 켤 경우 원본 이미지의 키 feature 등은 마스킹 되어서 전체 형태 중 해당 부분은 제외 후 학습
- 사람의 경우 hfp 끌 경우 모습은 거의 그대로 나오고 피부색이나 머리카락 색만 다르게 나옴
- 안경을 쓰는 등의 이미지는 많지 않아서 학습 시킬 경우 썬그라스 위에 눈이 그려지기도 함



# Diffusion 
- 문장을 넣었을 때 이미지를 만들어내는 모형

## DragGAN 
- 업데이트 FF 한 번이면 되므로 빠름
- w (latent code)를 업데이트함
- GAN에서는 z -> G -> Imgf 로 만듦 
- GAN은 노이즈를 어떻게 이미지로

## diffusion
- 1년 안된 기술
- GAN 보다는 더 사용됨
- Dif는 이미지를 노이즈로 
- 120억장 데이터 
- pretraining 모형 가져다 사용 가능

## Stable diffusion
- 44page Z_T 는 Q 값, key와 value는 text의 인코딩된 값
- transformer를 사용
- cross attention: 
- self attention: 픽셀간 상관관계 (전체 이미지에서 quality)
- cross, self 번갈아가면서 나옴
- 잘 학습된 VAE가 필요함
- 인코딩: CLIP github 참고
- 오리지널 제목은 Latent diffusion
- negative prompt 있고 없고에 따라 차이가 있음 (정형화 되어있음)
- openart -> negative prompt
- oppenart.ai/promptbook 
- 오브적트 상관관계 표현하는 것은 어려움
- cat playing with a dog --> cat, dog
- 단일 오브젝트 묘사는 잘함

## 소스 오픈
- LAION 1.45B model 오픈
- DALE 등 나온지 1,2달만에..
- LAION 때문에 확 뜸 (아마존으로 몇 백억 썼으나..)
- Midjorny: 디자이너 (미드저니)
- DALI2: bing 
- anything --> 에니메이션 캐릭터
- 치라옴닉스? chilontmix 
- firefly: 어도비
- https://stability.ai/stablediffusion

## GAN vs DIFF
- GAN은 학습한 데이터 범위 내에서 벗어나지 못 함
- Diffusion은 노이즈를 생성하므로 한계가 없음


## VQGAN
- 48페이지, VQGAN의 Encoder에서 나온 값을 transformer로 만든 codebook 값으로 quantization 한 후 Decoder로 나감
- 코드북의 사이즈에 따라서 
- Dali1은 코드북만 사용, 사이즈 4096, 픽셀당 4096 가지의 표현 가능
- 오픈된 기본 사이즈는 512 x 512
- huggingface에 diffuer library 사용

# 프롬프트엔지니어링
- 택스트를.. 엔지니어링
- 트렌스포머이므로 영어 막 집어넣음
- Gen2 movie 에니메이션 

# diffusion model 실습

        conda create -n diffusers python=3.8
        conda activate diffusers
        pip install torch torchvision transformers diffusers


# 4일차
- Automatic1111 : https://github.com/AUTOMATIC1111/stable-diffusion-webui
- controlnet : https://github.com/lllyasviel/ControlNet
- controlnet model : https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main

# Attention in steps

- 초반 입력한 프롬프트의 영향은 뒷쪽 입력한 프롬프트의 영향보다 적다. 왜냐면 후반으로 갈수록 denoising 영향이 적다. GAN에서도 후반에는 detail 변화
- self attention은 초반 작동 하지 않고 (이미지 품질 담당), cross attention map에서는 초반에 (프롬프트 담당)


# finetuneing 문제
- 내가 입력한 부분만 기억하고 기존 것은 까먹는거 
- 나만의 diffusion 모형 만들 수 있다.. - 3page
- 계산의 어려움 때문에 안 쓰임
- LoRA 나오면서 사용됨 (page4) GPT 타깃 170B, 1/10이하 저장 가능
- 모든 커스터마이즈드 모형에 사용 
- 모형 10개라도 DeltaW 만 업데이트 하면 됨

# 아이덴티티 유지하면서 일부만 변형

- 입만 벌리기, - feature 를 그대로 쓰고 입만 벌리게..
- 구부린 다리 --> 펼친 다리
- 키포인트 

# Hugging face
- github과 같은데 데모 페이지를 만들어서 더 유용



# automatic1111
- Automatic1111 : https://github.com/AUTOMATIC1111/stable-diffusion-webui

      bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)

- 위와 같이 하면 설치 완료
- 127.0.0.1:7860 접속
- 그라지오 webUI
- 다시 실행시

      webui.sh 

- txt2img 로 만들고 inpainting으로 맘에 안드는 부분 지우고 img2img로 
- dreambooth 설치시 extension -> available -> search에서 dreambooth
- ControlNet for Stable Diffusion WebUI 같은 방법으로 설치
- xformers 설치 이슈
            (diffsers) etriai07@edwk38:~/E6001/stable-diffusion-webui$ bash webui.sh --xformers

- midjourney 
- bing image
- firefly




# Question
- 특정 weigh 0으로 이용할 수 있는지..
- false 가 많은 데이터에 대해서는...
- Stygen3 이후 diffusion으로 
  - 최소사양 Titan A100 8개

