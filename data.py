from os import listdir
from os.path import isfile, join
import librosa
import numpy as np

class VCTK:
    """Sound abstraction, does feature extraction"""

    """
        Load the data from the wav bank
        Speakers must be specified

        Speakers are (ints) ids between 225 and 
        376 (see VCTK corpus for available speakers ids).
    """
    features = []
    labels = []

    def __init__(self, speakers):
        print('Loading wav files for ' + str(len(speakers)) + ' ....')
        speaker_index = 0
        for speaker in speakers:
            # Read the files in the directory
            dir_name = './asset/data/wav48/p' + str(speaker) + '/'
            waveFilesNames = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
            speaker_one_hot = np.zeros(len(speakers))
            np.put(speaker_one_hot, speaker_index, 1)
            print('Loading ' + str(len(waveFilesNames)) + ' wav files for speaker ' + str(speaker))
            for waveFileName in waveFilesNames[:20]:
                filename = join(dir_name, waveFileName)
                y, sr = librosa.load(filename)
                print(filename)
                mfccFeatures = self.complete_features(librosa.feature.mfcc(y=y, sr=sr), 500).flatten()
                print(mfccFeatures.shape)
                self.features.append(mfccFeatures)
                # Create the onehot
                self.labels.append(speaker_one_hot)
            # Increment speaker index
            speaker_index = speaker_index + 1;
        print('Loaded ' + str(len(self.features)) + ' examples.')

    def complete_features(self, f, times_units):
        """Pads and truncates the frames to match the classifier's format (i.e 20x500)"""
        # Dimension is 20 * x (x is the length of the audio)
        if(f.shape[1] > times_units): # Remove values on first dimension
            return f[:,range(0,500)]
        
        # Less then 500 frames
        diff = times_units - f.shape[1] # Number of 0s missing on first dimension
        completed = np.zeros([20, diff])
        stacked = np.hstack((f, completed))
        return stacked

    def next_batch(self, batch_size):
        """Returns a batch from the data"""
        return self.features, self.labels
