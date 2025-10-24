```bash
# use dataset "elastic" to train model with a pretrained model state
python3 train.py --model_choice resnext101_64x4d_v1 --model_state_path model/resnext101_64x4d_v1/2025-10-23T06-05-23_96.61423414130995.pth --dataset elastic

# predict on test data
python3 prediction.py --model_choice resnext101_64x4d_v1

# perform training data argumentation
python3 preprocess.py
``