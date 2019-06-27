"""
testing phase module. implements methods for visualising results

"""

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch

from torch.autograd import Variable

from sklearn.metrics import confusion_matrix

from Callback import Callback
from train import place_and_unwrap


class test_images(Callback):
    """ Predicts on a subset of images and displays prediction vs actual
    Implements confusion matrix on test set data

    """

    def __init__(self, path="", batch_size=5, classes=None):
        """instantiates variables

        Keyword Arguments:
            path {str}: Current path to experiment 
            batch_size {int}: test size for visual display
            classes: training classes in dataset
        """

        self.training_path = path
        self.batch_size = batch_size
        self.classes = classes

    def training_ended(self, phases, model, **kwargs):
        """At training end get test data and plot confusion matrix and for 5 images display output vs correct results

        Arguments:
            phases: training phases
            model: trained model
        """

        device = 'cuda'
        y_tot = []
        pred_tot = []
        for phase in phases:

            if phase.name == 'test':
                for i, batch in enumerate(phase.loader):
                    x, y = place_and_unwrap(batch, device)  # unwrap data

                    prediction = self.predict(x, model, y)  # predict result
                    y, pred = zip(*prediction)  # unzip result and correct
                    x = x.cpu().data.numpy()

                    if i == 1:
                        self.display_predictions(
                            x, pred, y)  # display results
                        y_tot = np.asarray(y)

                        pred_tot = np.asarray(pred)

                    y_tot = np.append(y_tot, y)

                    pred_tot = np.append(pred_tot, pred)

                self.plot_confusion_matrix(y_tot, pred_tot)

    def display_predictions(self, x, pred, y):
        """Display the predictions for the 5 images
        and saves to training folder

        Arguments:
            x: input data
            pred: predicted output
            y: ground truth
        """
        fig = plt.figure(figsize=(15, 15))

        for i in range(self.batch_size):
            vals = x[i, :, :, :]
            sub = fig.add_subplot(1, self.batch_size, i + 1)
            val = pred[i]
            val2 = y[i]
            res = self.classes[val]
            res2 = self.classes[val2]

            sub.set_title("predicted = " + res + "\n" + "Actual = " + res2)
            plt.axis('off')
            img = np.asarray(vals)
            img = np.transpose(img, (1, 2, 0))
            # Get Specific channels for rgb
            rgbimg = self.get_rgb(img, 61, 38, 19)

            # Normalize Inputs
            imgmin, imgmax = rgbimg.min(), rgbimg.max()
            rgbimg = (rgbimg - imgmin) / (imgmax - imgmin)
            plt.imshow(rgbimg)
        file_loc = [str(self.training_path) + '\\checkpoints' +
                    '\\' + 'predictions.jpg']
        s = ""
        s = s.join(file_loc)
        pred_path = Path(s)
        plt.savefig(pred_path)
        plt.show()

    def predict(self, img, model, y):
        """[summary]

        Arguments:
            img: input data
            model: trained model
            y:  ground truth

        Returns:
            results: zipped tuple of prediction and input target
        """

        device = 'cuda'
        img1 = img.clone().detach()
        #img = torch.tensor(img).float()
        # image_tensor = image_tensor.unsqueeze_(0)
        inpt = Variable(img1)
        inpt = inpt.to(device)
        output = model(inpt)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu().data.numpy()
        y = y.cpu().data.numpy()
        results = list(zip(predicted, y))

        return results

    def get_rgb(self, img, r, g, b):
        """ Takes the image and three band values as inputs and outputs a stacked array of these bands for display in imgshow


        Arguments:
            img: input data
            r: Red band for display
            g : green band for display
            b: blue band for display

        Returns:
           img: (m,n, b) output x bby y by channel.
        """

        # Get specific bands of hyperspectral image
        red_channel = img[:, :, r]
        green_channel = img[:, :, g]
        blue_channel = img[:, :, b]

        img = np.stack((red_channel, green_channel, blue_channel), axis=2)
        img = img.astype('float32')
        return img

    def plot_confusion_matrix(self, y_true, y_pred, title=None):
        """       This function prints and plots the confusion matrix.

        Arguments:
            y_true: ground truth
            y_pred: predicted output

        Keyword Arguments:
            title: title of confusion matric (default: {None})

        Returns:
            ax: returns plot
        """

        if not title:
            title = 'confusion matrix'

        # Compute confusion matrix

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = self.classes
        print('Confusion matrix')

        print(cm)
        fig2, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig2.tight_layout()
        file_loc = [str(self.training_path) +
                    '\\checkpoints\\confusion_matrix.jpg']  # NEED TO FIX
        s = ""
        s = s.join(file_loc)
        conf_path = Path(s)
        plt.savefig(conf_path)
        plt.show()

        return ax
