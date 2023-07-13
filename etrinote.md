# GAN lecture in ETRI
## Installation
- Anaconda
- https://github.com/NVlabs/stylegan3
- https://github.com/clovaai/stargan-v2
- https://github.com/XingangPan/DragGAN


# GPT
프롬프트 지니 설치 (크롬 플러그인)


# StyleGAN3

## env
- nvidia-smi not available
- nvcc: for pytouch. it depends on env
- (base) etriai07@edwk38:~/E6001/stylegan3$ conda env create -f environment.yml


Nvidia visualizer

conda 설치

git clone https://github.com/NVlabs/stylegan3

cd stylegan3

conda env create -f environment.yml

conda activate stylegan3

python visualizer.py stylegan2-ffhq-1024x1024.pkl


# StarGAN v2

- https://github.com/clovaai/stargan-v2

환경설정은 위 사이트 Software installation 참고

설치 에러시 버전 삭제 후 다시 설치

core/checkpoint.py 48번째줄 수정 필요

strict = False 추가

## 데이터 다운로드

bash download.sh pretrained-network-celeba-hq

bash download.sh wing

## 코드리뷰

- solver.py의 train, compute_d_loss, compute_g_loss 리뷰

mask 설명
model.py 의 forward 함수

encoder의 feature 정보를 복사해서 decoder로 바로 사용

solver의 186 다음과 같으면

x = self.hpf(mask * cache[x.size(2)]) # 
#x = x + self.hpf(mask * cache[x.size(2)]) # 
눈 코 부분 쉐이프 정보 가지고 옴

이 때는

mask = masks[0] if x.size(2) in [32] else masks[1]
mask = masks[0] if x.size(2) in [32] else masks[0]
위와 같이 masks[0]으로 하면 입까지

[ ]

Face Recognision
https://github.com/ageitgey/face_recognition
점을 그리는 그림

코드 수정
가시화

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
더블클릭 또는 Enter 키를 눌러 수정




# DragGAN

- vs. diffusion 



# Task 2
A style-based generator architecture for Generative adversarial network

low resolution 4x4 ~ 8x8 포즈, 4개 adain layer 고정하면 포즈와 쉐이프는 바뀌지 않음

middle resolution 16x16 ~ 32x32

High ~

GAN 이 전체적인 이미지를 만들어낼 때 대부분은 low resolution에서 만들어진다

model.py의 Generator.forward 수정

sudo apt-get install nvidia-utils-535

sudo reboot now

sudo apt-get install nvidia-smi


# Task 3

- key point 입력 
- cycle loss 가 쓸모 없어짐
- compute g loss -> cycle-consistency loss 지우기

