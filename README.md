# asr_cn_tf1
中文自动语音识别


模型：

cnn + ctc

gru + ctc


训练环境：

tensorflow 1

keras


数据：

将训练数据整理成kaldi中的格式

至少需要 text 和 wav.scp 两个文件


训练： 

python train_am.py 


测试：

python test_am.py
