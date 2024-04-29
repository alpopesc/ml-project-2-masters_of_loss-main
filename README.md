# Deep learning-based dysarthric speech classification in adverse acoustic environments

Our Data Preparation code is in the chunk_computation folder.
 - In feature_extractor.py, the functions needed to calculate the mel and stft spectrograms of audio files are present.
 - In csv_generator.py, we calculate the mel and stft spectrograms, divide all the audio into chunks and save them as csv files.
 - In noise.ipynb, we add noises to clean data, calculate mel and stft spectrograms of noisy data, divide them into chunks and save them as csv files.
 
 Note that for both csv_generator.py and noise.ipynb one needs to specify the path at the beginning in accordance to where the data folder is saved

Our network training and testing are in network.ipynb. Our results are shown separately for clean training-clean testing, clean training-noisy testing and noisy training-noisy testing cases.

The audio recordings should be stored in ../data/HC for the healthy controls and ../data/PC for the Parkinson's Disease patients. 

ml_grp_1.yml is the Conda environment we used for data preparation. We used Google Colab to run the noise.ipynb and network.ipynb files to be able to use GPU. 

