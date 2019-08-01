import numpy as np
import h5py

# for numpy files
# for i in range(55):
#     data = np.load('/Users/sophia/Documents/Studium/Mathematik/Master/AdvancedDeepLearning4ComputerVision/data/sampleBeachData/sceneChunks/train0000_00_{}.npy'.format(i))
#     np.savetxt('/Users/sophia/Documents/Studium/Mathematik/Master/AdvancedDeepLearning4ComputerVision/data/sampleBeachData/sceneChunks_txt/train0000_00_{}.txt'.format(i), data, delimiter=',')

# for hdf5 files
for i in range(22):
    f = h5py.File('/Users/sophia/Documents/Studium/Mathematik/Master/AdvancedDeepLearning4ComputerVision/data/train0600_00/train0600_00_{}.hdf5'.format(i))
    points = f["points"][:]
    print("sample", i, "with corresponding images", f["corresponding_images"][:])
    np.savetxt('/Users/sophia/Documents/Studium/Mathematik/Master/AdvancedDeepLearning4ComputerVision/data/train0600_00txt/train0600_00_{}.txt'.format(i), points, delimiter=',')