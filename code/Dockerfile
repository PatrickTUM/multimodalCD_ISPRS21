FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# install dependencies 
RUN conda install -c conda-forge cupy
RUN conda install -c conda-forge opencv
RUN pip install scipy sklearn rasterio natsort matplotlib scikit-image pandas tqdm natsort

# add directories and files
RUN mkdir -p ./aux
ADD aux ./aux
ADD data_loader.py ./data_loader.py 
ADD test.py ./test.py
