# dcnn
A PyTorch implementation of the Dynamic CNN, from the paper [A Convolutional Neural Network for Modeling Sentences](https://arxiv.org/abs/1404.2188) by Kalchbrenner et al.

## requirements
- python>=3.6
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for optional Docker support.

## usage

```
python3 dcnn.py --help
```

## docker support
```
docker run --runtime=nvidia -it --rm --ipc=host mwilliammyers/dcnn
```

## datasets

[kaggle yelp review dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset/downloads/yelp_review.csv)

[twitter US airline sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment/data)
