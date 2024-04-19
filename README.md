# Restoration by Generation with Constrained Priors

<br>CVPR 2024 (Highlight)<br>

[Zheng Ding](), [Xuaner Zhang](https://ceciliavision.github.io), [Zhuowen Tu](https://pages.ucsd.edu/~ztu)
[Zhihao Xia](https://likesum.github.io)

[Paper](https://arxiv.org/pdf/2312.17161.pdf) / [arXiv](https://arxiv.org/abs/2312.17161) / [Project Page](https://gen2res.github.io/)

![teaser](figs/teaser.png)

## Environment Setup

```bash
conda create -n gen2res python=3.8
conda activate gen2res
conda install pytorch=1.11 cudatoolkit=11.3 torchvision -c pytorch
conda install dlib mpi4py scikit-learn scikit-image tqdm -c conda-forge
pip install blobfile==2.0.2 tqdm
```

## Personal Restoration - Finetuning

For finetuning, please prepare a personal dataset contains several images first (we use around 20 images). Put all images into a folder and then align them by running:

```bash
python scripts/align.py -i PATH_TO_PERSONAL_PHOTO_ALBUM -o personal_images_aligned -s 256
```

Then we can run the following command to finetune the model. Please download the pretrained diffusion model trained on FFHQ from [here](). Feel free to try other pretrained diffusion models.

```bash
python scripts/finetune.py 
        --resume_checkpoint model.pt \
        --batch_size 4 \
        --lr 1e-5 \
        --lr_anneal_steps 5000 \
        --log_dir log_personal \
        --data_dir personal_images_aligned
```

## Personal Restoration - Inference

Before we restore the low-quality images, we also need to restore the blind restoration

After we have the personalized model, we can run the following command to restore the image. 

```bash
python scripts/ref_sample_single.py \
            --noise_step 200 \
            --data_dir PATH_TO_INPUT_IMAGES \
            --outputdir restored_images \
            --model_path PATH_TO_FINETUNED_MODEL \
```

## Citation

If you find this work helpful, please consider citing using the following BibTeX entry.

```BibTeX
@inproceedings{ding2024restoration,
  title={Restoration by Generation with Constrained Priors},
  author={Ding, Zheng and Zhang, Xuaner and Tu, Zhuowen and Xia, Zhihao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
