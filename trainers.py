import tensorflow as tf
import keras.backend as K
import os
import numpy as np
import genomelake
from genomelake.extractors import ArrayExtractor
from pybedtools import BedTool
from io_utils import *
from sklearn import metrics
from keras.layers import Input
import numpy as np
from models import*
from keras.utils.generic_utils import Progbar
import logging
from models import *
from sklearn import metrics
from metrics import *





DEFER_DELETE_SIZE=int(250 * 1e6) 
def create_tensorflow_session(visiblegpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    session = tf.Session(config=session_config)
    K.set_session(session)
    return session









def cross_entropy_loss(y_true,logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=logits))


def focal_loss(y_true,logits):
    y_pred=tf.nn.sigmoid(logits)
    gamma=3
    alpha=0.25
    eps=K.epsilon()
    y_pred=K.clip(y_pred,eps,1.-eps)
    alpha_wts=K.ones_like(y_true) * alpha
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_wts=tf.where(tf.equal(y_true,1),alpha_wts,1.-alpha_wts)
    return -K.sum(alpha_wts * K.pow(1. - pt, gamma) * K.log(pt))


class ClassifierTrainer(object):

    def __init__(self,train_validation_dict,extractors_dict,logdir,batch_size=128,
                 epoch_size=10000, num_epochs=10,
                 early_stopping_metric='auPRC', early_stopping_patience=5,
                 logger=None):
        
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.batches_per_epoch=int(self.epoch_size /self.batch_size) + 1
        self.num_epochs = num_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.logger = logger
        self.train_intervals_file=train_validation_dict['train_data']
        self.validation_intervals_file=train_validation_dict['validation_data']
        self.train_intervals_bedtool=BedTool(train_validation_dict['train_data'])
        self.train_labels=np.load(train_validation_dict['train_labels'])
        self.validation_intervals_bedtool=BedTool(train_validation_dict['validation_data'])
        self.validation_labels=np.load(train_validation_dict['validation_labels'])
        self.Model=SequenceDNAseClassifier()
        self.extractors_dict=extractors_dict
        ##Define tensorflow session
        self.sess=create_tensorflow_session(visiblegpus='1')


    def predict_on_batch(self,batch_data_dict):
        """ batch_data_dict is a dictionary of data type to numpy array
        This is completely equivalent to doing model.model.predict_on_batch

        """
        feed_dict = {seq_placeholder:batch_data_dict['data/genome_data_dir'], dnase_placeholder:batch_data_dict['data/dnase_data_dir']}
        preds_=self.sess.run(tf.nn.sigmoid(self.logits),feed_dict=feed_dict)

        return preds_

    def predict_on_intervals(self,intervals,extractor_dict,batch_size=128):
        """ Args:
            intervals: raw intervals file 
            extractor_dict: dictionary of data type to corresponding genomelake extractor
        """    
        batch_generator=generate_from_intervals(intervals=intervals,data_extractors_dict=extractor_dict,indefinitely=False,batch_size=batch_size)
        predict_list=[]
        target=len(intervals)
        progbar=Progbar(target=target)
        total=0
        for batch_dict in batch_generator:
            total+=batch_size
            if total>target:
                total=target
            predict_list.append(self.predict_on_batch(batch_dict))
            progbar.update(total)
        return np.vstack(predict_list)

 


    def train(self):
        """
            The input placeholders for the data get created here as they are imported from models
            They are global variables in the models

        """


        ###Create Tensorflow Graph
        self.logits=self.Model.get_logits()
        labels_placeholder=tf.placeholder(tf.float32,shape=(None,1))
        self.loss_op=cross_entropy_loss(labels_placeholder,self.logits)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=.0003)
        self.train_op=self.optimizer.minimize(self.loss_op)

        
        ###Initialize all global variables
        init_op=tf.global_variables_initializer()
        self.sess.run(init_op)


        ##Define the data types to extractor dictionary
        ##Start the train loop
        progbar=Progbar(target=self.epoch_size)
        batch_generator = generate_from_intervals_and_labels(intervals=self.train_intervals_bedtool ,labels=self.train_labels, data_extractors_dict=self.extractors_dict)
        
        for epoch in xrange(self.num_epochs):

            validation_preds = self.predict_on_intervals(self.validation_intervals_bedtool, self.extractors_dict)
            validation_metrics=ClassificationResult(labels=self.validation_labels,predictions=validation_preds)
            #val_auc_roc = metrics.roc_auc_score(self.validation_labels,validation_preds)

            print(validation_metrics)
            for batch_indx in range(1,self.batches_per_epoch+1):
                data_dict, labels_batch  = next(batch_generator)
                #print labels_batch.shape
                feed_dict={seq_placeholder:data_dict['data/genome_data_dir'],dnase_placeholder:data_dict['data/dnase_data_dir'],labels_placeholder:labels_batch}
                logits_,loss_,_=self.sess.run([self.logits,self.loss_op,self.train_op],feed_dict=feed_dict)
        #         if batch_indx%10==0:
        #             print("AUC is %s"%(str(metrics.roc_auc_score(y_true=labels_batch,y_score=logits_))))
                    
                start = (batch_indx-1)*self.batch_size
                stop = batch_indx*self.batch_size
                if stop > self.epoch_size:
                    stop = self.epoch_size
                progbar.update(stop)    
            print("Finished epoch %s"%(str(epoch)))     
          







        


if __name__ == '__main__':
    train_intervals_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/train_intervals.bed'
    validation_intervals_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/validation_intervals.bed'
    train_labels_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/train_labels.npy'
    validation_labels_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/validation_labels.npy'
    # train_intervals=BedTool(train_intervals_file)
    # train_labels=np.load(train_labels_file)
    # validation_intervals=BedTool(validation_intervals_file)
    # validation_labels=np.load(validation_labels_file)
    
    def create_small_validation_set(size=100000):
        data=np.loadtxt(validation_intervals_file,dtype=str)
        labels=np.load(validation_labels_file)
        indices=np.random.choice(len(labels),size=size,replace=False)
        sub_set_data=data[indices]
        sub_set_labels=labels[indices]
        np.save('./validation_subset_labels',sub_set_labels)
        np.savetxt(fname='./validation_subset_data.bed', X=sub_set_data,fmt='%s',delimiter='\t')

        
        print("Created sub set validation data")

    #create_small_validation_set()
    # validation_intervals=BedTool('./validation_subset_data.bed')
    # validation_labels=np.load('validation_subset_labels.npy')
    validation_intervals_subset_file='./validation_subset_data.bed'
    validation_labels_subset_file='./validation_subset_labels.npy'
    train_validation_dict={'train_data':train_intervals_file,'train_labels':train_labels_file,'validation_data':validation_intervals_subset_file,'validation_labels':validation_labels_subset_file}
    

    ##Path to the extracted the Genome and DNAse
    genome_file_path='/srv/scratch/manyu/Baseline_TF_binding_models/extracted_data/GRCh38.p3.genome.fa/'
    dnase_file_path='/srv/scratch/manyu/memmap_bcolz/DNASE.K562.fc.signal.hg38.bigwig/'
    genome_extractor=ArrayExtractor(genome_file_path)
    dnase_extractor=ArrayExtractor(dnase_file_path)
    extractors_dict={'data/genome_data_dir':genome_extractor,'data/dnase_data_dir':dnase_extractor}

    ##Create the logdir for saving models
    logdir ='./logdir_SeqDnase_CEBPB_CEloss'
    if not os.path.exists(logdir):
        os.mkdir(logdir)

        
    Trainer=ClassifierTrainer(train_validation_dict,extractors_dict,logdir)
    Trainer.train()



    


    #test_validation_intervals=[validation_intervals[i] for i in range(1000)]
    #train_validation_dict={'train_data':train_inter}

        
    










