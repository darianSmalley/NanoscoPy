from __future__ import print_function
import argparse
import os
import numpy as np 
import torch

def progbar(curr, total, full_progbar, accuracy):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', 
          '#'*filled_progbar + '-'*(full_progbar-filled_progbar), 
          '[{:>7.2%}]'.format(frac), 
          'Accuracy: [{:>7.2%}]'.format(accuracy),
          end='')
    
def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, num_cats):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data.float())
        
        # Compute loss based on criterion
        loss = criterion(output, target)

        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()

        # Get predicted class by rounding 
        pred = output.round()
        pred = torch.argmax(pred, dim=1)
        
        # Count correct predictions overall 
        # Get element-wise equality between the preds and the targets for this batch,
        # finally sum the equalities and convert to a python float
        n_equal = pred.eq(target).sum().item()
        correct += n_equal

        # Update progress bar
        batch_accuracy = n_equal / torch.numel(target)
        progbar(batch_idx, len(train_loader), 10, batch_accuracy)
    
    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / ((batch_idx+1) * batch_size * num_cats)
    
    print('\nTrain set\t Average loss: {:.4f}\t Average Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, (batch_idx+1) * batch_size * num_cats, train_acc))
          
    return train_loss, train_acc

def test(model, device, test_loader, criterion, num_cats):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
                
            # Predict for data by doing forward pass
            output = model(data.float())
            
            # Compute loss based on same criterion as training
            loss = criterion(output, target)

            # Append loss to overall test loss
            losses.append(loss.item())

            # Get predicted class by rounding 
            pred = output.round()
            pred = torch.argmax(pred, dim=1)
            
            # Count correct predictions overall 
            # Get element-wise equality between the preds and the targets for this batch,
            # finally sum the equalities and convert to a std python float
            n_equal = pred.eq(target).sum().item()
            correct += n_equal

    test_loss = float(np.mean(losses))
    test_acc = (100. * correct) / (len(test_loader.dataset) * num_cats)
    print('Test set\t Average loss: {:.4f}\t Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset) * num_cats, test_acc))
    
    return test_loss, test_acc

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    debug = FLAGS.debug
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    sts_trainset = STSDataset(data_path=train_path)
    sts_valset = STSDataset(data_path=val_path)
    
    # Seperate validation set for testing
    test_split = 0.4
    batch_size = FLAGS.batch_size
    idxs = list(range(len(sts_valset)))
    val_idx, test_idx = train_test_split(idxs, test_size=test_split)
    sts_valset = Subset(sts_valset, val_idx)
    sts_testset = Subset(sts_valset, test_idx)
    
    train_dataloader = DataLoader(sts_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(sts_valset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(sts_testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize the model and send to device 
    model = ConvNet1D(n=n, num_classes=num_classes, debug=False).to(device)

    if debug: 
        print(model) 
        print('mode {}'.format(FLAGS.mode))
    
    EPOCHS = FLAGS.num_epochs 
    learning_rate = FLAGS.learning_rate
    num_classes = FLAGS.num_classes
    
    if debug: print(f'{num_classes} classes')
    
    # Use Cross Entropy as the loss function since we want to choose 1-of-N classes 
    # which are inter-related. 
    # Cross Entropy will map the networks predictions to a probabilities in range [0,1] 
    criterion = model.criterion
    
    # Define optimizer function.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define tracked network performance metrics
    best_accuracy = 0.0
    train_losses = np.zeros(EPOCHS)
    train_accuracies = np.zeros(EPOCHS)
    test_losses = np.zeros(EPOCHS)
    test_accuracies = np.zeros(EPOCHS)
    
    # Define the tensorboard writer to track netowrk training
#     writer = SummaryWriter('./runs/CIFAR10/Dropout/{}'.format(FLAGS.name))
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, EPOCHS + 1):
        print("Epoch {}".format(epoch))
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, batch_size)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        
        # Store epoch metrics in memory
        i = epoch - 1
        train_losses[i] = train_loss
        train_accuracies[i] = train_accuracy
        test_losses[i] = test_loss
        test_accuracies[i] = test_accuracy 
            
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    
      
        # Log the epoch metrics in tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)  
        
    # Flush all writer logs and close resource
    writer.flush()
    writer.close()

    # Print final results to console
    print("accuracy is {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Enable debug mode.')
    parser.add_argument('--name',
                        type=str,
                        default='model',
                        help='Set model name')
    parser.add_argument('--num_classes',
                        type=str,
                        default='4',
                        help='Set number of classes')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)