import numpy as np
if __name__ == '__main__':
#    train_intervals_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/train_intervals.bed'
    validation_intervals_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/validation_intervals.bed'
#    train_labels_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/train_labels.npy'
    validation_labels_file='/srv/scratch/manyu/Baseline_TF_binding_models/data/validation_labels.npy'
    # train_intervals=BedTool(train_intervals_file)
    # train_labels=np.load(train_labels_file)
    # validation_intervals=BedTool(validation_intervals_file)
    # validation_labels=np.load(validation_labels_file)
    np.random.seed(42)    
    def create_small_validation_set(size=100000):
        data=np.loadtxt(validation_intervals_file,dtype=str)
        labels=np.load(validation_labels_file)
        indices=np.random.choice(len(labels),size=size,replace=False)
        sub_set_data=data[indices]
        sub_set_labels=labels[indices]
        np.save('./validation_subset_labels',sub_set_labels)
        np.savetxt(fname='./validation_subset_data.bed', X=sub_set_data,fmt='%s',delimiter='\t')

        
        print("Created sub set validation data")
    create_small_validation_set(10000)
