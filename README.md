# Non Linear Independent Components Estimation (NICE)

This project implements the models proposed in [NICE: Non-linear Independent Components Estimation paper](https://arxiv.org/abs/1410.8516)
written by Laurent Dinh, David Krueger and Yoshua Bengio.

The model is trained separately on two datasets:
- MNIST
- MNIST Fashion

The results of the models can be found in [results.pdf](./results.pdf).

<img src="samples/fashion-mnist_batch128_coupling4_coupling_typeadditive_mid1000_hidden5_.ptepoch0.png" >
<img src="samples/mnist_batch128_coupling4_coupling_typeadditive_mid1000_hidden5_.ptepoch0.png" >

---

## Setup

### Prerequisites
- Python 3.x
- NumPy
- torch
- torchvision
- matplotlib

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Code

To train the NICE model on MNIST or Fashion-MNIST:

```bash
python train.py
```

You can customize the training using the following command-line arguments:

| Argument           | Description                                                   | Default         |
|--------------------|---------------------------------------------------------------|-----------------|
| `--dataset`        | Dataset to train on: `mnist` or `fashion-mnist`              | `mnist`         |
| `--prior`          | Latent distribution (`logistic`, `gaussian`, etc.)           | `logistic`      |
| `--batch_size`     | Number of images per mini-batch                               | `128`           |
| `--epochs`         | Number of training epochs                                      | `50`            |
| `--sample_size`    | Number of images to generate during sampling                  | `64`            |
| `--coupling-type`  | Type of coupling layers (`additive`, etc.)                    | `additive`      |
| `--coupling`       | Number of coupling layers                                     | `4`             |
| `--mid-dim`        | Dimensionality of the hidden intermediate layer               | `1000`          |
| `--hidden`         | Number of hidden layers in each coupling function             | `5`             |
| `--lr`             | Initial learning rate for the optimizer                       | `1e-3`          |

#### Example with Custom Parameters:

```bash
python train.py --dataset fashion-mnist --coupling 6 --coupling-type additive --mid-dim 512 --hidden 4 --lr 0.0005
```

---

### Output Files

- Generated Samples: After training, sample images are saved in the `./samples/` folder with the following filename format:

  ```
  samples/{dataset}_batch{batch_size}_coupling{coupling}_coupling_type{coupling_type}_mid{mid_dim}_hidden{hidden}epoch{epoch}.png
  ```

  Example:
  ```
  samples/fashion-mnist_batch128_coupling4_coupling_typeadditive_mid1000_hidden5epoch0.png
  ```
- **Loss Logs**:
  - Training losses are saved as: `loss_train_{dataset}_{coupling_type}.pkl`
  - Testing losses are saved as: `loss_test_{dataset}_{coupling_type}.pkl`
