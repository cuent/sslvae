# Semi-supervised Learning with Deep Generative Models     

[![Paper](http://img.shields.io/badge/paper-arxiv.1406.5298-B31B1B.svg)](https://arxiv.org/pdf/1406.5298.pdf)
[![Conference](http://img.shields.io/badge/NeurIPS-2014-4b44ce.svg)](https://nips.cc/Conferences/2014/Schedule?showEvent=4448)



## Description   
[PyTorch](https://pytorch.org/) implementation of Semi-supervised Learning with Deep Generative Models using [PyTorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/). This repository implements only the Semi-supervised model M2.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/cuent/sslvae   

# install project   
cd sslvae
pip install -e .   
pip install -r requirements.txt
 ```   
 SSLVAE - M2:   
 ```bash
# module folder
cd src/sslvae/   

ssvae
python sslvae_trainer.py    
```

Modify hyperparameters in `sslvae/conf`

See logging in tensorboard

```bash
tensorboard --logdir src/ssvae/outputs/
```

## Implementations      
- [SSLVAE - M2](src/sslvae)
- [see full derivation of equations](src/sslvae/derivation)  

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
