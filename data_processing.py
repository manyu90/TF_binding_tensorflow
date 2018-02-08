import numpy as np
import argparse





def create_train_validation_split(intervals_file,labels_file,valid_chroms,save_dir):
    data=np.loadtxt(intervals_file,dtype=str)
    labels=np.load(labels_file)
    idix=np.in1d(data[:,0],valid_chroms)
    train_idx=np.where(~idix)
    validation_idx=np.where(idix)
    train_labels=labels[train_idx]
    validation_labels=labels[validation_idx]
    train_intervals=data[train_idx]
    validation_intervals=data[validation_idx]
    np.savetxt(fname='%s/train_intervals.bed'%(save_dir),X=train_intervals,fmt='%s',delimiter='\t')
    np.savetxt(fname='%s/validation_intervals.bed'%(save_dir),X=validation_intervals,fmt='%s', delimiter='\t')
    np.save(arr=train_labels,file='%s/train_labels'%(save_dir))
    np.save(arr=validation_labels,file='%s/validation_labels'%(save_dir))
    
    
    
if __name__=='__main__':
    import json
    f=open('ZBTB33_dnase_AND_chipseq_regions_labels_stride200_flank400.json','r')
    dict_=json.load(f)
    f.close()
    intervals_file=dict_['K562']['regions']
    labels_file=dict_['K562']['labels']
    valid_chroms=['chr9']
    savedir='/srv/scratch/manyu/TF_binding_tensorflow/'
    create_train_validation_split(intervals_file,labels_file,valid_chroms,savedir)
