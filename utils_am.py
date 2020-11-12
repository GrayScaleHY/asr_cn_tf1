import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from keras import backend as K
import re
import math

def dic2args(dic):
    """
    将dict转换成args的格式。
    """
    params = tf.contrib.training.HParams()
    for keys in dic:
        params.add_hparam(keys, dic[keys])
    return params

def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type='train',
        data_path='data/',
        batch_size=1,
        data_length=None)
    return params


class get_data():

    def __init__(self, args):
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.source_init()

    def source_init(self):
        print('get source list...')

        # 将kaldi形式的data文件转成[text]和[wav path]的两个list
        pny_list, wav_list = data_2_list(self.data_path) 

        self.wav_lst = []
        self.pny_lst = []
        for i in range(len(wav_list)):
            self.wav_lst.append(wav_list[i])
            self.pny_lst.append(pny_list[i].split(' '))
        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
        print('make am vocab...')
        self.am_vocab = self.mk_am_vocab(self.pny_lst)
        self.batch_num = len(self.pny_lst) // self.batch_size # 分多少个batch

    def get_am_batch(self):
        """
        迭代器，生成输入到网络的input和output。
        """
        while 1:
            for i in range(self.batch_num):
                wav_data_lst = []
                label_data_lst = []
                for index in range(i*self.batch_size,(i+1)*self.batch_size):
                    # 计算音频特征
                    # feature = compute_fbank(self.wav_lst[index])
                    feature = get_mfcc(self.wav_lst[index])

                    ## feature的长度要大于ctc label长度的8倍。
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if feature.shape[0] // 8 >= label_ctc_len: # 之所以要8的倍数是因为卷积模型做了三次步长为2的strite
                        wav_data_lst.append(feature)
                        label_data_lst.append(label)

                # 将一个batch的feature整合成一个四维矩阵array
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)

                # 将input和output打包成dict
                inputs = {'the_inputs': pad_wav_data,
                            'the_labels': pad_label_data,
                            'input_length': input_length,
                            'label_length': label_length,
                            }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )} # 一个batch的大小

                yield inputs, outputs

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    def han2id(self, line, vocab):
        return [vocab.index(han) for han in line]

    def wav_padding(self, wav_data_lst):
        """
        输入一个2维矩阵的list(该list中的每个矩阵第二维相同)。
        将改矩阵整合成一个（batch, h, w, channel）的四维矩阵。
        改矩阵的h为8的倍数。
        返回array矩阵和原来list中每个矩阵的长度除以八的列表。
        """
        wav_lens = [math.ceil(len(data)/8) for data in wav_data_lst]
        new_wav_data_lst = np.zeros((len(wav_data_lst), max(wav_lens)*8, wav_data_lst[0].shape[1], 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, np.array(wav_lens)

    def label_padding(self, label_data_lst):
        """
        输入一个label的list，将其转换成二维的array矩阵。
        """
        label_lens = np.array([len(label) for label in label_data_lst])
        new_label_data_lst = np.zeros((len(label_data_lst), max(label_lens)))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            line = line
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    def ctc_len(self, label):
        """
        如果有连续两个相同label，则返回长度加一。
        """
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len


def data_2_list(data_path):
    """
    将kaldi形式的text和wav.scp转换成一一对应的[wav path]和[text]列表。
    """
    text_dic = {}
    pny_list = []
    wav_list = []
    for s in open(os.path.join(data_path,"text")):
        s_s = re.split(" ",s.strip())
        text_dic[s_s[0]] = " ".join(s_s[1:])
    for s in open(os.path.join(data_path,"wav.scp")):
        s_s = re.split(" ",s.strip())
        pny_list.append(text_dic[s_s[0]])
        wav_list.append(s_s[1])
    return pny_list, wav_list

def get_mfcc(file):
    """
    返回维度为32的mfcc
    """
    fs, data = wav.read(file)
    mfcc_ = mfcc(data,fs,winlen=0.032,winstep=0.016,numcep=32,nfilt=64)
    return mfcc_

def compute_mfcc(file):
    """
    对音频文件提取mfcc特征
    """
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
    mfcc_feat = mfcc_feat[::3] ## 跳帧
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat  

def compute_fbank(file):
    """
    获取信号的时频图
    """
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input

# word error rate------------------------------------
def GetEditDistance(str1, str2):
    """
    字错率
    """
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost

def decode_ctc(num_result, num2word):
    """
    ctc解码
    """
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype = np.int32)
    in_len[0] = result.shape[1]
    r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])
    return r1, text
