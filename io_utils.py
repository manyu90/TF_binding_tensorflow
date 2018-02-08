

def batch_iterator(iterable,batch_size=128):
    it=iter(iterable)
    try:
        while True:
            batch=[]
            for i in xrange(batch_size):
                batch.append(next(it))
            yield batch
    except StopIteration:
        yield batch
        

        
def get_fixed_number_of_intervals(intervals_bedtool,num_batches,batch_procesing_size):
    
    total_processed_batches=0
    return_list=[]
    
    for batch in batch_iterator(intervals_bedtool,batch_procesing_size):
        return_list.append(copy.deepcopy(batch))
        total_processed_batches+=1
        if total_processed_batches==num_batches:
            break
    return return_list        
    
    #Vstack all the elements of the list
    features_tensor=np.vstack(features_list)
    np.save(file=save_file_path,arr=features_tensor)
    print("Saved features file to %s \n"%(save_file_path))
    
