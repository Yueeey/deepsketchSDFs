# DeepSketchSDF

This is a PyTorch implementation of the Computers & Graphics paper [A Study of Deep Single Sketch-Based Modeling: View/Style Invariance, Sparsity and Latent Space Disentanglement](https://www.sciencedirect.com/science/article/abs/pii/S0097849322001078).


## Installation

To get started, simply clone the repo and run the setup bash script, which will take care of installing all packages and dependencies.

```
./setup.sh
```

Sometimes, you may need to install the following packages manually.

```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```


## Data

In our project, we store data according to the following structure:
```
data/
  <chairs>/
      samples/
          <instance_name>.npz
      meshes/
          <instance_name>.obj
      renders/
        <instance_name>/
            naive_mad/
                base/
                    azi_0_elev_10_0001.jpg
                    ...
                bias/
                    azi_-5_elev_15_0001.jpg
                    ...
            sty_mad/
                base/
                    azi_0_elev_10_0001.jpg
                    ...
                bias/
                    azi_-5_elev_15_0001.jpg
                    ...
            sil_mad/
                base/
                    azi_0_elev_10__sil.png0001.png
                    ...
                bias/
                    azi_-5_elev_15__sil.png0001.png
                    ...
```

We provide pre-processed and subsampled ShapeNet data for [chairs](https://drive.google.com/file/d/1_ESc98RNIkXV0lHOw0q8LFtStgcQf5kB/view?usp=sharing) to get you started (124GB).

Simply download it and unzip it in the `data/` folder and make sure the folder is arranged according to the above structure to get going.

## Single-view reconstruction

You can train a single-view reconstruction model for chairs with regression loss by running

```
python train_svr_reg.py -e experiments/chairs_svr_reg/
```

Note that, running the script above will overwrite the pretrained checkpoint.

Once the model is trained, you can generate the 3D shape by running

```
python reconstruct_svr_reg.py -e experiments/chairs_svr_reg/
```

## Single-view reconstruction with mask

You can train a single-view reconstruction model for chairs with mask by running

```
python train_svr_mask.py -e experiments/chairs_svr_mask/
```

Note that, running the script above will overwrite the pretrained checkpoint.


Once the model is trained, you can generate the 3D shape by running

```
python reconstruct_svr_reg.py -e experiments/chairs_svr_reg/
```

## Contact

If you have any questions, please contact Yue Zhong: <zysyly@163.com>
