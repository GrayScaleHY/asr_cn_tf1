import numpy as np
from utils_am import decode_ctc, compute_mfcc
from gru_ctc import Am, am_hparams
import math
import argparse

# 动态参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '-v', '--vocab_file', type=str, default='config/conf_am.yaml')
parser.add_argument(
    '-m', '--model_file', type=str, default='save_models/11-12-14-14-34-logs/model_05.h5')
parser.add_argument(
    '-w', '--wav_file', type=str, default='test.wav')
cmd_args = parser.parse_args()

# 加载vocab
am_vocab = []
for s in open(cmd_args.vocab_file):
    am_vocab.append(s.strip())

## 加载语音识别模型
am_args = am_hparams()
am_args.vocab_size = len(am_vocab)
am = Am(am_args) 
print('loading acoustic model...')
am.ctc_model.load_weights(cmd_args.model_file)

# 生成mfcc并扩展成四维矩阵输入到模型中
mfcc = compute_mfcc(cmd_args.wav_file)
x = np.zeros((1,8*math.ceil(mfcc.shape[0]/8),mfcc.shape[1],1))
x[0,:mfcc.shape[0],:,0] = mfcc

# 预测结果
result = am.model.predict(x, steps=1)
_, text = decode_ctc(result, am_vocab)
text = ' '.join(text)
print('预测结果：', text)



