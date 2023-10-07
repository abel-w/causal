import random

import torch

import helper
from differential_evolution import differential_evolution
import os
import numpy as np
from resnet import ResNet,Bottleneck
import PIL.Image as Image


#############################################################################

def imgbxyc2tensorbcxy(img):
    if len(np.shape(img)) ==4:
        img = np.float32(np.transpose(img,[0,3,1,2])/255)
        img = torch.tensor(img).cuda()

    elif len(np.shape(img)):
        img = np.float32(np.transpose(img,[2,0,1]))/255
        img = torch.tensor(img).cuda()
        img = torch.unsqueeze(img,0)

    return img


class DDEmodel:
    def __init__(self, models, data, class_names, dimensions=(224, 224)):
        self.model = models
        self.x_test, self.y_test = data
        self.class_name = class_names
        self.dimensions = dimensions

    def attack(self, img_id, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False, DE='DE', epsilon=0.5, LS=0, no_stop=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (-1, 1), (-1, 1), (-1, 1)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, model, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, model, targeted_attack, verbose,
                                       epsilon=epsilon, no_stop=no_stop)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False, disp=True, DE=DE, LS=LS)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        with torch.no_grad():
            prior_probs = model(imgbxyc2tensorbcxy(self.x_test[img_id]))[0]
            # prior_probs = model(np.array([self.x_test[img_id]]))[0]
            predicted_probs = model(imgbxyc2tensorbcxy([attack_image]))[0]
            # predicted_probs = model(np.array([attack_image]))[0]
        prior_probs = prior_probs.detach().cpu().numpy()
        predicted_probs= predicted_probs.detach().cpu().numpy()
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 1]
        success = predicted_probs[actual_class] < epsilon and (predicted_class != actual_class)
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_name, predicted_class)
            # helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)
        if np.argmax(prior_probs) != np.argmax(predicted_probs):
            im_s = Image.fromarray(attack_image)
            im_s.save(os.path.join('./output/pixels_attack/%d_pix'%pixel_count,
                                   '%04d@%.3e_%.3e@%.3e_%.3e.png'
                                   %(im_id,prior_probs[0],prior_probs[1],
                                     predicted_probs[0],predicted_probs[1])))

        return [model, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x]

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        imgs_perturbed = imgbxyc2tensorbcxy(imgs_perturbed)
        with torch.no_grad():
            predictions = model(imgs_perturbed)[:, target_class]
        predictions = predictions.detach().cpu().numpy()
        # predictions = model.predict(imgs_perturbed)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions


    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False, epsilon=0.5, no_stop=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)
        with torch.no_grad():
            confidence = model(imgbxyc2tensorbcxy(attack_image))[0]
        confidence = confidence.detach().cpu().numpy()
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if (confidence[target_class] <= epsilon and not no_stop and
                ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class))):
            return True


########################################################################################
if __name__ == '__main__':
    num_pix = 5

    ############################################
    # print(torch.cuda.memory_summary())
    # torch.cuda.empty_cache()
    img_out = './output/pixels_attack/%d_pix'%num_pix
    if not os.path.exists(img_out):
        os.makedirs(img_out)

    model = torch.load('./model_resnet_breast.pt')
    model.eval()

    idx = np.load('./index_sample_test.npy')

    x0 = np.load('./x_breast.npy')
    y_true0 = np.load('./y_true_breast.npy')
    y_pred0 = np.load('./y_pred_breast.npy')

    # idx = [i for i in range(len(x0))]
    # random.shuffle(idx)
    # idx = idx[0:500]

    x =x0[idx]
    y_true = y_true0[idx]
    y_pred = y_pred0[idx]


    data = [x, y_true]
    class_names = ['negative','positive']
    n_suc = 0

    dde = DDEmodel(models=[model], data=data, class_names=class_names)
    for im_id in range(len(x)):
    #model, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result.x
        att = dde.attack(img_id=im_id, model=model, target=None, pixel_count=num_pix,
                       maxiter=50, popsize=400, verbose=False, plot=False,
                         DE='DE', epsilon=0.5, LS=0, no_stop=False)
        n_suc+=int(att[5])
        print(im_id+1,'/',len(x),'-number of success:',n_suc)


    with open(os.path.join(img_out,'pixels_attack_%d.txt'%num_pix),'w+') as f:
        f.write('total: %d,sucess: %d-percent: %.5f'%(len(x),n_suc,n_suc/len(x)))

    # with open('./pixels_attack_%d.txt'%num_pix,'w+') as f:
    #     f.write('total: %d,sucess: %d-percent: %.5f'%(len(x),n_suc,n_suc/len(x)))




    print('--------------------finish--------------------------')