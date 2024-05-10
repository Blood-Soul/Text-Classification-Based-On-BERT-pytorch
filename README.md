# Text-Classification-Based-On-BERT-pytorch
本项目为入门新手项目，只适合学习  
使用BERT进行文本分类  
以kaggle比赛[Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)为例  
本项目输入、输出格式，评价指标等均以此为例
# Envirenment
python~=3.11.9  
torch~=2.3.0  
transformers~=4.40.2  
pandas~=2.2.2  
scikit-learn~=1.4.2  
tqdm~=4.66.4  
# Dataset
在train.py和evaluate.py中可以修改TRAIN_PATH和TEST_PATH，指向训练集和测试集路径  
# Predictions
在执行evaluate.py后会生成预测结果，具体文件路径可在evaluate.py中的PREDICTION_PATH修改  
# BERT-base-cased
需要提前下载bert预训练文件，并放入项目中，具体路径可在train.py和evaluate.py中BERT_PATH修改
#Best_Weights
训练完毕后，会将最佳权重存放在BEST_MODEL_WEIGHTS_PATH指定路径中，预测时读取路径也是BEST_MODEL_WEIGHTS_PATH
