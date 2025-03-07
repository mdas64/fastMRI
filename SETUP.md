# Setup Instructions for Deep Learning Project

## Clone the Repo and Install Dependencies

Clone the repo.

```bash
git clone https://github.com/mdas64/fastMRI.git
```

Run the following commands:

```bash
conda create -n fastmri_env python=3.9 -y
conda activate fastmri_env
conda install pytorch torchvision torchaudio -c pytorch
pip install fastmri
conda install -c conda-forge jupyter
conda install numpy=1.26
conda install -c conda-forge h5py matplotlib
pip install pytorch-lightning==1.9.5
```

## Download Data

Visit the [fastMRI dataset page](https://fastmri.med.nyu.edu/). Scroll to the bottom of the webpage to apply for access. This will send the data to your email address. Download the following files:

- knee_singlecoil_train (~72.7 GB)
- knee_singlecoil_val (~14.9 GB)
- knee_singlecoil_test (~1.4 GB)

This took me ~1 hour to download all the data. 

Move these .tar files into the fastMRI directory (the top level directory in the repository).

Run the following commands:

```bash
tar -xvf knee_singlecoil_test.tar.xz
tar -xvf knee_singlecoil_val.tar.xz
tar -xvf knee_singlecoil_train.tar.xz
```

Not sure how long this step took since I ran the last command overnight. 

After completing the above steps, you should have the 
following directories in the top-level folder:

- /singlecoil_test
- /singlecoil_val
- /singlecoil_train

## Work through the tutorial .ipynb notebook

Run the following command:

```bash
jupyter notebook
```

Open the fastMRI_tutorial.ipynb file. Work through the notebook up until the section on the Root-Sum-of-Squares Transform. The last cell you execute should visualize the knee MRI. 

## Train the baseline UNet Model

Exit jupyter notebook.

Run the following command to begin the training process. Remember to replace '/path/to/repo' with the absolute path to your fastMRI repo.

```bash
python3 fastmri_examples/unet/train_unet_demo.py --challenge singlecoil --data_path /path/to/repo/fastMRI/singlecoil_train --mask_type random
```

I configured the training process to work on my personal setup. You might need to modify the file fastmri_examples/unet/train_unet_demo.py to work with your own setup if this command isn't working for you. Text me if you can't figure it out. 
