import os
import tensorflow as tf
from utils_am import get_data, data_hparams, dic2args
from keras.callbacks import ModelCheckpoint
from keras.backend import tensorflow_backend
import yaml
import time
import argparse
# from cnn_ctc import Am, am_hparams
from gru_ctc import Am, am_hparams

# 自动分配GPU内存，以防止Gpu内存不够的情况
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tensorflow_backend.set_session(tf.Session(config=config))

# 动态参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config', type=str, default='./config/conf_am.yaml')
parser.add_argument(
    '-s', '--save_path', type=str, default='save_models/'+time.strftime("%m-%d-%H-%M-%S")+"-logs")
cmd_args = parser.parse_args()

## 加载整个训练必要的config
f = open(cmd_args.config, 'r', encoding='utf-8')
parms_dict = yaml.load(f, Loader=yaml.FullLoader)
f.close()

## 训练数据参数
data_args = dic2args(parms_dict['data'])
train_data = get_data(data_args)
batch_num = train_data.batch_num
train_batch = train_data.get_am_batch()

## 准备验证所需数据
validation_data = None
validation_steps = None
if parms_dict['data']["dev_path"]:
    dev_args = dic2args(parms_dict['data'])
    dev_args.data_path = parms_dict['data']["dev_path"]
    dev_data = get_data(dev_args)
    dev_batch = dev_data.get_am_batch()
    validation_data = dev_batch
    validation_steps = 50

## 声学模型训
am_args = dic2args(parms_dict['model'])
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args) 

## 训练参数
epochs = parms_dict['train']['epochs']
save_path = parms_dict['train']['save_path'] = cmd_args.save_path
retrain_dir = parms_dict['train']['retrain_dir']

## save vocab and config
os.makedirs(save_path,exist_ok=True)
# 保存vocab
am_vocab = train_data.am_vocab
f = open(os.path.join(save_path,"vocab"),"w")
f.write("\n".join(am_vocab))
f.close()
# 保存config
parms_dict["data"]["am_vocab_file"] = os.path.join(save_path,"vocab")
f = open(os.path.join(save_path,"config.yaml"),"w",encoding='utf-8')
yaml.dump(parms_dict,f)
f.close()

## 是否加载预训练模型
if retrain_dir:
    print('load acoustic model...')
    am.ctc_model.load_weights(retrain_dir)

## checkpoint，保存模型信息
ckpt = "model_{epoch:02d}-{loss:.2f}.h5"
checkpoint = ModelCheckpoint(os.path.join(save_path, ckpt), monitor='val_loss', save_weights_only=True, verbose=1, save_best_only=False)

## 开始训练
am.ctc_model.fit_generator(
    train_batch, # 打包好的迭代型训练数据
    steps_per_epoch=batch_num, # 一个epoch训练多少个batch
    epochs=epochs, # 训练多少个epoch
    callbacks=[checkpoint], # 保存的model形式
    workers=1, 
    use_multiprocessing=False, 
    validation_data=validation_data, 
    validation_steps=validation_steps)

## 保存最后一个训练epoch的模型
am.ctc_model.save_weights(os.path.join(save_path,'model_'+str(epochs).zfill(2)+'.h5'))





