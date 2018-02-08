from data_processing import *
import genomelake
import numpy as np
from genomelake.extractors import FastaExtractor
from genomelake.extractors import BigwigExtractor
from genomelake.extractors import ArrayExtractor
import pybedtools
from pybedtools import BedTool


if __name__=='__main__':	
    train_intervals_file='/srv/scratch/manyu/TF_binding_tensorflow/data/train_intervals.bed'
    genome_file_path='/srv/scratch/manyu/Baseline_TF_binding_models/extracted_data/GRCh38.p3.genome.fa/'
    print("Extracting Genome \n")


    extracted_genome=ArrayExtractor(genome_file_path)
    print("Creating training Bedtool \n")
    train_intervals_bedtool=BedTool(train_intervals_file)
    import IPython
    IPython.embed()




