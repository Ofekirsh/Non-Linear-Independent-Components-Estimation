"""Training procedure for NICE.
"""

import argparse
import pickle

import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
#from tqdm import trange
#import matplotlib.pyplot as plt
import nice


def train(flow, trainloader, optimizer, epoch, device):
    loss_epoch = 0
    flow.train()  # set to training mode
    for inputs,_ in trainloader:
        optimizer.zero_grad()
        inputs = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2]*inputs.shape[3]) #change  shape from BxCxHxW to Bx(C*H*W)
        inputs = inputs.to(device)
        #TODO Fill in
        loss = - flow(inputs).mean()
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()

    loss_epoch /= len(trainloader)
    return loss_epoch


def test(flow, testloader, filename, epoch, sample_shape, device, should_we_sample=False):
    loss_inference = 0
    flow.eval()  # set to inference mode
    with torch.no_grad():
        if should_we_sample:
          samples = flow.sample(100).to(device)
          a,b = samples.min(), samples.max()
          samples = (samples-a)/(b-a+1e-10) 
          samples = samples.view(-1,sample_shape[0],sample_shape[1],sample_shape[2])
          torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                       './samples/' + filename + 'epoch%d.png' % epoch)
        #TODO full in
        for xs, _ in testloader:
            xs = xs.view(xs.shape[0],xs.shape[1]*xs.shape[2]*xs.shape[3]).to(device) #change  shape from BxCxHxW to Bx(C*H*W)
            loss = - flow(xs).mean()
            loss_inference += loss.item()
        loss_inference /= len(testloader)
        return loss_inference


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)) #dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'coupling%d_' % args.coupling \
             + 'coupling_type%s_' % args.coupling_type \
             + 'mid%d_' % args.mid_dim \
             + 'hidden%d_' % args.hidden \
             + '.pt'
    full_dim = 28 * 28
    flow = nice.NICE(
                prior=args.prior,
                coupling=args.coupling,
                coupling_type=args.coupling_type,
                in_out_dim=full_dim,
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    #TODO fill in
    loss_train = []
    loss_test = []
    for epoch in range(args.epochs):
        loss_epoch_train = train(flow, trainloader, optimizer, epoch, device)
        print(f"Loss Train: {loss_epoch_train}")
        loss_train.append(loss_epoch_train)

        loss_epoch_test = test(flow, testloader, model_save_filename, epoch, sample_shape, device)
        print(f"Loss Test: {loss_epoch_test}")
        loss_test.append(loss_epoch_test)

    # sample    
    test(flow, testloader, model_save_filename, 0, sample_shape, device, should_we_sample=True)

    with open(f"loss_train_{args.dataset}_{args.coupling_type}.pkl", 'wb') as f:
        pickle.dump(loss_train, f)

    with open(f"loss_test_{args.dataset}_{args.coupling_type}.pkl", 'wb') as f:
        pickle.dump(loss_test, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)