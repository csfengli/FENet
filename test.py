import numpy as np
import torch
from barbar import Bar
def test_model(model, dataloader, device):
    #Initialize and accumalate ground truth, predictions, and image indices

    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # index = index.to(device)
    
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            # Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / (len(dataloader.sampler))
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], #'Index':Index[1:],
                 'test_acc': np.round(test_acc.cpu().numpy()*100,2)}
    
    return test_dict