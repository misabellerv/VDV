import numpy as np
import pandas as pd
from preprocessing.transforms import Normalize, ApplyHOG, GaussianBlur, DenoiseWavelet
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def create_pipeline(config):
    if not config.Preprocessing.WaveletDenoising.Use:
        return Pipeline([
            ('normalize', Normalize()),
            ('gaussian_blur', GaussianBlur(config.Preprocessing.GaussianBlur.KernelSize)),
            ('hog', ApplyHOG(config.Preprocessing.HOG.Visualize))
        ])
    else:
        return Pipeline([
            ('normalize', Normalize()),
            ('denoise_wavelet', DenoiseWavelet(
                config.Preprocessing.WaveletDenoising.Wavelet,
                config.Preprocessing.WaveletDenoising.Level,
                config.Preprocessing.WaveletDenoising.Threshold
            )),
            ('hog', ApplyHOG(config.Preprocessing.HOG.Visualize))
        ])

def process_image(img, pipeline):
    return pipeline.transform(img)

def preprocess_pipeline(dataframe, config):
    print('Starting Pre-processing Pipeline...')
    
    start_time = time.time()
    
    width, height = config.InputImages.Width, config.InputImages.Height
    total_imgs = config.InputImages.TotalTrainingImages
    num_jobs = config.Joblib.NumJobs

    array_df = dataframe.values.reshape(-1, width, height).astype(np.float32)

    pipeline = create_pipeline(config)
    
    proc_df = Parallel(n_jobs=num_jobs)(
        delayed(process_image)(img, pipeline) for img in tqdm(array_df, desc='Processing Images', total=len(array_df))
    )
    
    proc_df = np.array(proc_df).reshape(total_imgs, width * height)
    proc_df = pd.DataFrame(proc_df)
    
    print(f'Pre-processing Finished in {time.time() - start_time:.2f} Seconds!')
    
    return proc_df  