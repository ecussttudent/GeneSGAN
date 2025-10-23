import pandas as pd
import os
os.environ['R_HOME'] = r"C:\Program Files\R\R-4.5.1"
os.environ['PATH'] = r"C:\Program Files\R\R-4.5.1\bin\x64;" + os.environ.get('PATH', '')
from GeneSGAN.Gene_SGAN_clustering import cross_validated_clustering

if __name__ == '__main__':
	output_dir = './test_output'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# current_directory = os.getcwd()
	# print("Current working directory:", current_directory)

	image_data = pd.read_csv('./datasets/toy_data_imaging.csv')
	gene_data = pd.read_csv('./datasets/toy_data_gene.csv')
	#print(gene_data.dtypes)

	fold_number = 2
	ncluster = 3
	start_saving_epoch = 2000

	max_epoch = 3000

	WD = 0.11
	AQ = 30
	cluster_loss = 0.01
	genelr = 0.0002


	cross_validated_clustering(image_data, gene_data, ncluster, fold_number, 0.8, start_saving_epoch, max_epoch, output_dir, WD, AQ, cluster_loss,\
		genelr = 0.0002, batchSize=25, lipschitz_k=0.5)
