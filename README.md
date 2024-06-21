# MixT5
Development of MoE T5 for Software Capstone Design Class

# Overview
MoE Finetuning을 T5-v1.1-base 모델에 적용해, SQuAD task의 성능 향상을 확인   
추가로 서로 다른 Expert간에 다른 noise를 주어 랜덤성을 부여하는 방법을 탐구

# Setup
```bash
cd transformers
pip install -e .
cd ..
pip install -r requirements.txt
```

# Train
```bash
# moe 아키텍처를 사용하려면 2개의 path가 필요합니다.
# 베이스로 삼을 모델을 미리 로컬에 저장 후, 해당 모델의 config에 "activate_moe": true를 추가해주세요.
# 이후 reference path로 베이스로 삼은 모델의 path를,
# pretrained_path로 config를 바꿔준 로컬 path를 입력해주시면 됩니다.

python train_t5.py \
--pretrained_model_name_or_path YOUR_T5_PATH \
--reference_model_name_or_path google/t5-v1_1-base \
--save_dir YOUR_SAVE_DIR \
--save_strategy epoch \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 10 \
--wandb_model_name t5_v1_1_moe_6layer_10epoch_seed42_010 \
--project_name moe \
--use_wandb \
--enable_moe \ # moe 아키텍처로 학습하지 않으려면 지워주세요.
--gradient_accumulation_steps 2 \
--seed 42 \
```

# Inference
```bash
# inference.py의 't5_path = YOUR_PATH'에서 YOUR_PATH를 수정해주세요.

python inference.py
```

# Results
MoE FT 적용에 따른 결과 비교

| Model         | EM score | F1 score | 
|---------------|----------|----------|
| Pretrain-Only | 0.01892  | 5.152231 |
| Simple FT     | 82.5767  | 89.8063  |
| MoE FT        | **83.3050** | **90.2933** |

NoisyTune 적용에 따른 결과 비교

| Num MoE layers | lambda | EM score | F1 score |
|----------------|--------|----------|----------|
| 0              | -      | 82.5767  | 89.8063  |
| 6              | 0      | **83.3050** | **90.2933** |
| 6              | 0.1    | 83.2927  | 90.2017  |
| 6              | 0.2    | 83.2703  | 90.2127  |
| 6              | 0.3    | 83.2577  | 90.1413  |

# Conclusion
- 본 연구에서는 MoE 구조를 사용하지 않은 채 Pretrain된 언어모델을 Fine-Tuning 이전에 MoE 구조로 바꾸어서 학습하는 MoE Fine-Tuning 방법을 SQuAD 데이터셋에 대하여 T5-v1.1 모델에 적용하여 실험
- 실험 결과 Simple Fine-Tuning에 비해 MoE Fine-Tuning의 EM score와 F1 score가 모두 높았으며, 정성적으로도 더 나은 답변을 제공하는 점을 확인
- 뿐만 아니라 모든 Expert를 동일하게 복사하여 초기화하는 것이 아닌, Expert별로 NoisyTune 기법 등을 적용해 다른 Noise를 주어 다양성을 확보하는 실험을 진행했으나 4가지 다른 노이즈를 준 결과가 노이즈를 주지 않은 결과보다 좋지 않았으며, 적절한 randomness를 부여하는 방법을 향후 추가로 탐구할 계획
- 제안한 방법론은 task에 국한 받지 않고, 다양한 모델에 적용할 수 있으므로 향후 다른 task나 model 구조에 대한 일반화 실험을 진행할 예정



# Reports
- Midterm: [Report](https://drive.google.com/file/d/1Ar1Pbj_3qY3P5EWIZMD5LJF3SJEWkbaO/view?usp=drive_link)
- Final: [Report](https://drive.google.com/file/d/1kZkIwC6inuX6WhX6mXib3cUs5SpfBtFU/view?usp=drive_link), [Poster](https://drive.google.com/file/d/1cbbvpvdyp06UBvmwrmRqCq3B3Ny4QnqM/view?usp=drive_link), [Demo video](https://youtu.be/cN1T0--fnjY)