from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import numpy as np
import sys

from keras import backend as K
from keras.layers import (
    Activation, AveragePooling1D, BatchNormalization,
    Conv1D, Dense, Dropout, Flatten, Input,
    MaxPooling1D, Merge, Permute, Reshape,
    PReLU, Multiply, Lambda
)
from keras.models import Model

#Definging some global Input placeholder variables
#These are the inputs to the model via the placeholder dictionary and the feed dictionary to be defined while training
seq_placeholder = Input(shape=(1000,4))
meth_placeholder = Input(shape=(1000,1))
dnase_placeholder = Input(shape=(1000,1))

##Define a mapping from data source to placeholder variable
placeholder_dict={
	"data/genome_data_dir":seq_placeholder,
	"data/dnase_data_dir":dnase_placeholder,
	"data/methylation_data_dir":meth_placeholder
}   


model_inputs = {
    "SequenceClassifier": ["data/genome_data_dir"],
    "SequenceDNAseClassifier":["data/genome_data_dir","data/dnase_data_dir"],
    "SequenceMethylationClassifier":["data/genome_data_dir","data/methylation_data_dir"],
    "SequenceMethylationRevCompClassifier":["data/genome_data_dir","data/methylation_data_dir"]
    }


shapes_dict={
	"data/genome_data_dir":(1000,4),
	"data/dnase_data_dir":(1000,1),
	"data/methylation_data_dir":(1000,1)

}




class Classifier(object):
    """
    Classifier interface.

    Args:
        
    Attributes:
        get_inputs (list): a list of input names.
            Derived from model_inputs unless implemented.
    """
    @property
    def get_inputs(self):
        return model_inputs[self.__class__.__name__]

    def __init__(self, **hyperparameters):
        pass

    def save(self, prefix):
        arch_fname = prefix + '.arch.json'
        weights_fname = prefix + '.weights.h5'
        open(arch_fname, 'w').write(self.model.to_json())
        self.model.save_weights(weights_fname, overwrite=True)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def get_placeholder_inputs(self):
        """Returns dictionary of named keras inputs"""
        return collections.OrderedDict(
            [(name, placeholder_dict[name])
             for name in self.get_inputs])




class SequenceClassifier(Classifier):

    def __init__(self,num_tasks=1,
                 num_filters=(15, 15, 15), conv_width=(15, 15, 15),
                 pool_width=35, dropout=0, batch_norm=False):
        assert len(num_filters) == len(conv_width)

        # # configure inputs
        ##Dictionary of data type to placeholder
        self.inputs = self.get_placeholder_inputs()
        

        # convolve sequence
        seq_preds = self.inputs["data/genome_data_dir"]
        for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
            seq_preds = Conv1D(
                nb_filter, nb_col, kernel_initializer='he_normal')(seq_preds)
            if batch_norm:
                seq_preds = BatchNormalization()(seq_preds)
            seq_preds = Activation('relu')(seq_preds)
            if dropout > 0:
                seq_preds = Dropout(dropout)(seq_preds)

        # pool and fully connect
        seq_preds = AveragePooling1D((pool_width))(seq_preds)
        seq_preds = Flatten()(seq_preds)
        seq_preds = Dense(output_dim=num_tasks)(seq_preds)
        self.logits=seq_preds
        preds = Activation('sigmoid')(self.logits)
        self.model = Model(inputs=self.inputs.values(), outputs=preds)

    def get_logits(self):
    	return self.logits



class SequenceMethylationClassifier(Classifier):
    def __init__(self, num_tasks=1,
                 num_filters=(55,50,50,50),conv_width=(25, 25, 25,25),

                 pool_width=25,
                 fc_layer_widths=(100,),
                 conv_dropout=0.0,
                 fc_layer_dropout=0.0,
                 batch_norm=False):
        assert len(num_filters)==len(conv_width)
        #assert len(num_combined_filters)==len(combined_conv_width)


        #configure inputs
        self.inputs = self.get_placeholder_inputs()
        
        seq_preds=self.inputs["data/genome_data_dir"]
        methylation_preds=self.inputs["data/methylation_data_dir"]
        logits = Merge(mode='concat', concat_axis=2)([seq_preds, methylation_preds])
        for num_filter,width in zip(num_filters,conv_width):
            logits=Conv1D(num_filter,width,kernel_initializer='he_normal')(logits)
            if batch_norm:
                logits=BatchNormalization()(logits)
            logits=Activation('relu')(logits)
            if conv_dropout>0:
                logits=Dropout(conv_dropout)(logits)

        logits=AveragePooling1D(pool_width)(logits)
        logits=Flatten()(logits)
        

        for fc_layer_width in fc_layer_widths:
                logits = Dense(fc_layer_width)(logits)
                if batch_norm:
                    logits = BatchNormalization()(logits)
                logits = Activation('relu')(logits)
                if fc_layer_dropout > 0:
                    logits = Dropout(fc_layer_dropout)(logits)

        # from IPython import embed
        # embed()

        #logits = Merge(mode='concat', concat_axis=-1)([logits, methylation_preds])
        logits = Dense(num_tasks)(logits)
        self.logits=logits
        preds = Activation('sigmoid')(self.logits)
        self.model = Model(inputs=self.inputs.values(), outputs=preds)

    
    def get_logits(self):
    	return self.logits






class SequenceDNAseClassifier(Classifier):

    def __init__(self,num_tasks=1,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 pool_width=25,
                 fc_layer_widths=(100,),
                 seq_conv_dropout=0.0,
                 dnase_conv_dropout=0.0,
                 combined_conv_dropout=0.0,
                 fc_layer_dropout=0.0,
                 batch_norm=False):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        # configure inputs
        self.inputs = self.get_placeholder_inputs()
        

        # convolve sequence
        # from IPython import embed
        # embed()
        seq_preds = self.inputs["data/genome_data_dir"]

        for nb_filter, nb_col in zip(num_seq_filters, seq_conv_width):
            seq_preds = Conv1D(
                nb_filter, nb_col, kernel_initializer='he_normal')(seq_preds)
            if batch_norm:
                seq_preds = BatchNormalization()(seq_preds)
            seq_preds = Activation('relu')(seq_preds)
            if seq_conv_dropout > 0:
                seq_preds = Dropout(seq_conv_dropout)(seq_preds)

        # convolve dnase
        dnase_preds = self.inputs["data/dnase_data_dir"]
        for nb_filter, nb_col in zip(num_dnase_filters, dnase_conv_width):
            dnase_preds = Conv1D(
                nb_filter, nb_col, kernel_initializer='he_normal')(dnase_preds)
            if batch_norm:
                dnase_preds = BatchNormalization()(dnase_preds)
            dnase_preds = Activation('relu')(dnase_preds)
            if dnase_conv_dropout > 0:
                dnase_preds = Dropout(dnase_conv_dropout)(dnase_preds)

        # stack and convolve
        logits = Merge(mode='concat', concat_axis=-1)([seq_preds, dnase_preds])
        for nb_filter, nb_col in zip(num_combined_filters, combined_conv_width):
            logits = Conv1D(nb_filter, nb_col, kernel_initializer='he_normal')(logits)
            if batch_norm:
                logits = BatchNormalization()(logits)
            logits = Activation('relu')(logits)
            if combined_conv_dropout > 0:
                logits = Dropout(combined_conv_dropout)(logits)

        # pool and fully connect
        logits = AveragePooling1D((pool_width))(logits)
        logits = Flatten()(logits)
        for fc_layer_width in fc_layer_widths:
            logits = Dense(fc_layer_width)(logits)
            if batch_norm:
                logits = BatchNormalization()(logits)
            logits = Activation('relu')(logits)
            if fc_layer_dropout > 0:
                logits = Dropout(fc_layer_dropout)(logits)
        logits = Dense(num_tasks)(logits)
        self.logits=logits
        preds = Activation('sigmoid')(logits) 
        self.model = Model(inputs=self.inputs.values(), outputs=preds)



    def get_logits(self):
    	return self.logits



class SequenceMethylationRevCompClassifier(Classifier):
    def __init__(self, num_tasks=1,
                 num_filters=(55,50,50,50),conv_width=(25, 25, 25,25),

                 pool_width=25,
                 fc_layer_widths=(100,),
                 conv_dropout=0.0,
                 fc_layer_dropout=0.0,
                 batch_norm=False):
        assert len(num_filters)==len(conv_width)
        #assert len(num_combined_filters)==len(combined_conv_width)


        #configure inputs
        self.inputs = self.get_placeholder_inputs()
        
        seq_preds=self.inputs["data/genome_data_dir"]
        methylation_preds=self.inputs["data/methylation_data_dir"]

        def reverse_comp(x):
        	return K.reverse(K.reverse(x,axes=-1),axes=-2)
        def reverse(x):
        	return K.reverse(x,axes=-2)	
        def get_C_locations(x):
        	return K.expand_dims(x[:,:,1],axis=-1)
        
        reverse_comp_seq = Lambda(reverse_comp)(seq_preds)
        reverse_methylation = Lambda(reverse)(methylation_preds)
        C_locations_forward_strand = Lambda(get_C_locations)(seq_preds)
        C_locations_reverse_strand = Lambda(get_C_locations)(reverse_comp_seq)
        forward_methylation_masked = Multiply()([methylation_preds,C_locations_forward_strand])  #Using the mask that is C
        reverse_methylation_masked = Multiply()([reverse_methylation,C_locations_reverse_strand])  #Using the reverse comp mask on C




        # reverse_comp_seq=K.reverse(K.reverse(seq_preds,axes=-1),axes=-2)
        # reverse_methylation=K.reverse(methylation_preds,axes=-2)
        # C_locations_forward_strand=K.expand_dims(seq_preds[:,:,1],axis=-1)  #Remember the order A:0,C:1,G:2,T:3
        # C_locations_reverse_strand=K.expand_dims(reverse_comp_seq[:,:,1],axis=-1)
        # forward_methylation_masked=Multiply()([methylation_preds,C_locations_forward_strand])  #Using the mask that is C
        # reverse_methylation_masked=Multiply()([reverse_methylation,C_locations_reverse_strand])  #Using the reverse comp mask on C

        logits = Merge(mode='concat', concat_axis=2)([seq_preds,forward_methylation_masked,reverse_comp_seq,reverse_methylation_masked])
 
        #logits = Merge(mode='concat', concat_axis=2)([seq_preds, methylation_preds])
        for num_filter,width in zip(num_filters,conv_width):
            logits=Conv1D(num_filter,width,kernel_initializer='he_normal')(logits)
            if batch_norm:
                logits=BatchNormalization()(logits)
            logits=Activation('relu')(logits)
            if conv_dropout>0:
                logits=Dropout(conv_dropout)(logits)

        logits=AveragePooling1D(pool_width)(logits)
        layer_shape=K.get_variable_shape(logits)
        flattened_shape=layer_shape[1]*layer_shape[2]
        logits=Reshape((flattened_shape,))(logits)     
        #logits=Flatten()(logits)
        

        for fc_layer_width in fc_layer_widths:
                logits = Dense(fc_layer_width)(logits)
                if batch_norm:
                    logits = BatchNormalization()(logits)
                logits = Activation('relu')(logits)
                if fc_layer_dropout > 0:
                    logits = Dropout(fc_layer_dropout)(logits)

        # from IPython import embed
        # embed()

        #logits = Merge(mode='concat', concat_axis=-1)([logits, methylation_preds])
        logits = Dense(num_tasks)(logits)
        self.logits=logits
        preds = Activation('sigmoid')(self.logits)
        self.model = Model(inputs=[seq_preds,methylation_preds], outputs=preds)

    
    def get_logits(self):
    	return self.logits

        

    