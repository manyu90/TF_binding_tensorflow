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


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, mode='w')
    logger.addHandler(file_handler)






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



def recall_at_precision_loss(labels,logits,lambda_):
    alpha=0.9
    
    l_stop_grad=2*tf.stop_gradient(lambda_)-lambda_
    pos = tf.boolean_mask(logits, tf.cast(labels, tf.bool))
    neg = tf.boolean_mask(logits, ~tf.cast(labels, tf.bool))
    total_pos=tf.cast(tf.reduce_sum(tf.shape(pos)),dtype=tf.float32)
    #total_neg=tf.reduce_sum(tf.shape(neg))
    pos_labels=tf.ones_like(pos,dtype=tf.float32)
    neg_labels=tf.zeros_like(neg,dtype=tf.float32)
    L_plus=tf.losses.hinge_loss(pos_labels,pos)
    L_minus=tf.losses.hinge_loss(neg_labels,neg)
    loss=(1+l_stop_grad)*L_plus+l_stop_grad*(alpha/(1-alpha))*L_minus -l_stop_grad*total_pos
    return loss


def roc_auc_score(y_true,logits):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        logits: `Tensor`. Predicted values.
        logits: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(logits, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(logits, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))

    
loss_dict={'cross_entropy':cross_entropy_loss,'focal_loss':focal_loss,'r@p_loss':recall_at_precision_loss,'roc_auc_loss':roc_auc_score}

class ClassifierTrainer(object):

    def __init__(self,train_validation_dict,extractors_dict,logdir,batch_size=128,
                 epoch_size=100000, num_epochs=100,
                 early_stopping_metric='auPRC', early_stopping_patience=5,
                 logger=None,save_best_model_prefix='SeqDnaseModel',visiblegpus='1',loss='cross_entropy'):
        
        self.logdir=logdir
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
        self.save_best_model_prefix=save_best_model_prefix
        self.loss=loss
        assert self.loss in loss_dict.keys()
        self.loss_function=loss_dict[self.loss]
        ##Define tensorflow session
        self.sess=create_tensorflow_session(visiblegpus=visiblegpus)

        ##Create some basic information logging about the training 

        self.logger.info('Train File: {}'.format(self.train_intervals_file))
        self.logger.info('Train Labels: {}'.format(train_validation_dict['train_labels']))
        self.logger.info('Validation File: {}'.format(self.validation_intervals_file))
        self.logger.info('Validation Labels: {}'.format(train_validation_dict['validation_labels']))
        self.logger.info('logdir: {}'.format(self.logdir))
        self.logger.info('early_stopping_metric: {}'.format(self.early_stopping_metric))
        self.logger.info('early_stopping_patience: {}'.format(self.early_stopping_patience))
        self.logger.info('batch_size: {}'.format(self.batch_size))
        self.logger.info('Epoch Size: {}'.format(self.epoch_size))
        self.logger.info('Model Type: {}'.format(self.Model.__class__.__name__))
        self.logger.info('Loss type used: {}'.format(self.loss))

        



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

 


    def train(self,learning_rate=.0003):
        """
            The input placeholders for the data get created here as they are imported from models
            They are global variables in the models

        """


        ###Create Tensorflow Graph
        self.lambda_=tf.Variable(1.0,trainable='True')
        self.logits=self.Model.get_logits()
        labels_placeholder=tf.placeholder(tf.float32,shape=(None,1))
        if self.loss=='r@p_loss':
            self.loss_op=self.loss_function(labels_placeholder,self.logits,self.lambda_)
        else:
            self.loss_op=self.loss_function(labels_placeholder,self.logits)    
        
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op=self.optimizer.minimize(self.loss_op)


        ##Loggint the optimizer information
        self.logger.info('optimizer: {}'.format(self.optimizer.__class__.__name__))
        self.logger.info('learning_rate: {}'.format(str(learning_rate)))

        ###Initialize all global variables
        init_op=tf.global_variables_initializer()
        self.sess.run(init_op)


        ##Define the data types to extractor dictionary
        ##Start the train loop
        
        current_best_metric = -np.inf 
        progbar=Progbar(target=self.epoch_size)
        batch_generator = generate_from_intervals_and_labels(intervals=self.train_intervals_bedtool ,labels=self.train_labels, data_extractors_dict=self.extractors_dict)
        print("Starting to Train model \n")
        logger.info("\nStarting to Train model \n")
        for epoch in xrange(self.num_epochs):

    
            for batch_indx in range(1,self.batches_per_epoch+1):
                data_dict, labels_batch  = next(batch_generator)
                
                feed_dict={seq_placeholder:data_dict['data/genome_data_dir'],dnase_placeholder:data_dict['data/dnase_data_dir'],labels_placeholder:labels_batch}
                self.sess.run(self.train_op,feed_dict=feed_dict)
                start = (batch_indx-1)*self.batch_size
                stop = batch_indx*self.batch_size
                if stop > self.epoch_size:
                    stop = self.epoch_size
                progbar.update(stop)    
            print("Finished epoch %s"%(str(epoch))) 
            logger.info('Finished epoch: {}'.format(str(epoch)))

            validation_preds = self.predict_on_intervals(self.validation_intervals_bedtool, self.extractors_dict)
            validation_metrics = ClassificationResult(labels=self.validation_labels,predictions=validation_preds)
            current_metric_of_consideration = validation_metrics.results[self.early_stopping_metric]
            if current_metric_of_consideration > current_best_metric:
                current_best_metric=current_metric_of_consideration
                best_epoch = epoch
                early_stopping_wait = 0
                print("Found new best model. Saving weights to %s \n"%(self.logdir))
                logger.info("Found new best model. Saving weights to {} ".format(os.path.join(self.logdir,self.save_best_model_prefix+'.weights.h5')))
                self.Model.save(os.path.join(self.logdir,self.save_best_model_prefix))
            else :
                if early_stopping_wait >= self.early_stopping_patience:
                    break
                early_stopping_wait += 1
                    
            

            print(validation_metrics)
            logger.info(str(validation_metrics))
            logger.info('\n\n\n')


          
        print("Finished training after {} epochs \n".format(epoch))
        logger.info("Finished training after {} epochs \n".format(epoch))

        if self.save_best_model_prefix is not None:
                print("The best model's architecture and weights (from epoch {0}) "
                                 'were saved to {1}.arch.json and {1}.weights.h5'.format(
                                     best_epoch, self.save_best_model_prefix))

                logger.info("The best model's architecture and weights (from epoch {0}) "
                                 'were saved to {1}.arch.json and {1}.weights.h5'.format(
                                     best_epoch, self.save_best_model_prefix))






        


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
    logdir ='./logdir_SeqDnase_ZBTB33_roc_auc_loss'

    logdir=os.path.abspath(logdir)
    assert(not os.path.exists(logdir))
    os.makedirs(logdir)

    setup_logger('train_logger',os.path.join(logdir,'metrics.log'))
    logger=logging.getLogger('train_logger')

    train_intervals_file_dnase_regions='/srv/scratch/manyu/TF_binding_tensorflow/data/train_intervals.bed'
    train_intervals_file_dnase_regions_labels='/srv/scratch/manyu/TF_binding_tensorflow/data/train_labels.npy'
    validation_intervals_file_dnase_regions='/srv/scratch/manyu/TF_binding_tensorflow/data/validation_intervals.bed'
    validation_intervals_file_dnase_regions_labels='/srv/scratch/manyu/TF_binding_tensorflow/data/validation_labels.npy'
    train_validation_dict_dnase={'train_data':train_intervals_file_dnase_regions,'train_labels':train_intervals_file_dnase_regions_labels,'validation_data':validation_intervals_file_dnase_regions,'validation_labels':validation_intervals_file_dnase_regions_labels}
        
    Trainer=ClassifierTrainer(train_validation_dict_dnase,extractors_dict,logdir,logger=logger,visiblegpus=1,batch_size=256,loss='roc_auc_loss')
    Trainer.train()



    


    #test_validation_intervals=[validation_intervals[i] for i in range(1000)]
    #train_validation_dict={'train_data':train_inter}
    

        
    










