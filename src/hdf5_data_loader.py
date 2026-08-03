import h5py
import numpy as np
import os

class HDF5DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File does not exist: {self.file_path}")

        try:
            with h5py.File(self.file_path, 'r') as f:
                X = np.array(f['X'])
                y = np.array(f['y'])
                # datetimes are stored as bytes, decode to str
                datetimes = np.array([dt.decode('utf-8') for dt in f['datetimes']])
                # Optional: convert to datetime objects if you prefer
                # datetimes = np.array([datetime.fromisoformat(dt.decode('utf-8')) for dt in f['datetimes']])
        except OSError as e:
            raise OSError(f"Failed to read HDF5 file: {e}")
        except KeyError as e:
            raise KeyError(f"Missing required dataset in HDF5 file: {e}")

        return X, y, datetimes

if __name__ == "__main__":
    loader = HDF5DataLoader("/home/erick.ramirez/repo/Project-PSI/hdf5_paper_datasets/dataset_sensor_range_1440_1690.h5")
    try:
        X, y, datetimes = loader.load()
        print(X.shape, y.shape, datetimes.shape)
        print("First datetime:", datetimes[0])
        print("First y:", y[0])
    except Exception as e:
        print(f"[ERR] {e}")
        exit(1)
