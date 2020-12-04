# Semi-supervised Learning with Deep Generative Models     

[![Paper](http://img.shields.io/badge/paper-arxiv.1406.5298-B31B1B.svg)](https://arxiv.org/pdf/1406.5298.pdf)
[![Conference](http://img.shields.io/badge/NeurIPS-2014-4b44ce.svg)](https://nips.cc/Conferences/2014/Schedule?showEvent=4448)

## Description   
[PyTorch](https://pytorch.org/) implementation of Semi-supervised Learning with Deep Generative Models using [PyTorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/). This repository implements only the Semi-supervised model M2.

#### CPU   
```bash   
python ssvae_trainer.py     
```


Hyperparameters configuration in [conf](conf) 

Tensorboard

```bash
tensorboard --logdir outputs/
```

## Implementations      
- [SSLVAE - M2](sslvae)
- [see full derivation of equations](derivation)  

### Citation   
```
@article{kingma2014semi,
  title={Semi-supervised learning with deep generative models},
  author={Kingma, Durk P and Mohamed, Shakir and Jimenez Rezende, Danilo and Welling, Max},
  journal={Advances in neural information processing systems},
  volume={27},
  pages={3581--3589},
  year={2014}
}
```   



