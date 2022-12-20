import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from PIL import ImageFile, ImageDraw, Image, ImageFont
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3


def get_data_prepared(data_path):

    image = []
    mask = []

    path_image = f"{data_path}images/"
    path_label = f"{data_path}masks/"
    print(path_image, end="\n")
    image_names = next(os.walk(path_image))[2]

    print(f"Begin to read images from {path_image}")
    # for n,name in tqdm(enumerate(image_names), total=len(image_names)): 
    for name in image_names:  
        path = path_image + name
        img = Image.open(path)
        img = img.convert("L")  
        img = np.asarray(img)
        img = img / 255    
        img = np.expand_dims(img, axis=-1)
        image.append(img)

    print(f"Begin to read masks from {path_label}")
    # for n,name in tqdm(enumerate(image_names), total=len(image_names)): 
    for name in image_names:  
        name = name.replace("image", "mask")    
        path = path_label + name
        img = Image.open(path)
        img = img.convert("L")
        img = np.asarray(img)
        img = img // 200 
        img = np.expand_dims(img, axis=-1)
        mask.append(img)

    image = np.array(image)
    mask = np.array(mask)

    return image, mask


def get_train_dataset(data_path):
    
    image, mask = get_data_prepared(data_path)

    # Label One-Hot Coding for Softmax Activation
    mask_float = mask.astype(np.float32)
    mask_one_hot = mask2onehot(mask_float, 2)

    X_train, X_val, Y_train, Y_val = train_test_split(image, mask_one_hot, test_size=0.2, random_state=42, shuffle=True)

    return X_train, Y_train, X_val, Y_val

def overlay_info(image, text):
    
    draw = ImageDraw.Draw(image)

    draw.text((10,10),f"{text[0]}")
    draw.text((10,25),f"{text[1]}")
    draw.text((10,40),f"{text[2]}")
    draw.text((10,55),f"{text[3]}")

    return image


def overlay_softmax(image, mask, label=np.array([]), h=np.array([])):
    """
    Overlay Image, Mask, Label and Entropy in one output pic
    :param image: Original Pic, Gray Scale 2D Pic
    :param mask: Predicted Mask, 2D only contain 0 and 1
    :param label: Optional, Original Label, 2D only contain 0 and 1
    :param h: Optional, Entropy, 2D only contain float between 0 and 1
    :return: Overlay Pic
    """

    image = image * 255
    image1 = Image.fromarray(image).convert("RGB")

    if label.any():
        intersection = mask * label
        only_in_label = label - intersection
        only_in_mask = mask - intersection

        intersection = intersection * 170
        only_in_label = only_in_label * 170
        only_in_mask = only_in_mask * 170

        r = np.expand_dims(only_in_mask, axis=-1)
        r = r + np.expand_dims(only_in_label, axis=-1)
        g = np.expand_dims(intersection, axis=-1)
        g = g + np.expand_dims(only_in_label, axis=-1)
        b = np.zeros([label.shape[0], label.shape[1], 1])

        rgb = np.concatenate((r, g, b), axis=2)
        image2 = Image.fromarray(np.uint8(rgb)).convert("RGB")

        overlayimage = Image.blend(image1, image2, alpha=0.3)

        if h.any():
            image_h = Image.fromarray(h * 255).convert("RGB")
            overlayimage = Image.blend(overlayimage, image_h, alpha=0.3)

    else:
        mask = mask * 170
        mask_r = np.expand_dims(mask, axis=-1)
        mask_gb = np.zeros([mask.shape[0], mask.shape[1], 2])
        mask = np.concatenate((mask_r, mask_gb), axis=2)

        image2 = Image.fromarray(np.uint8(mask)).convert("RGB")

        overlayimage = Image.blend(image1, image2, alpha=0.3)

        if h.any():
            image_h = Image.fromarray(h * 255).convert("RGB")
            overlayimage = Image.blend(overlayimage, image_h, alpha=0.3)

    return overlayimage
    
def mask2onehot(mask, num_classes):
    """
    convert picture mask in form [N, H, W, 1] to one hot mask in form [N, H, W, ONEHOT]
    """
    mask = mask.squeeze()
    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.float32)

    if len(_mask.shape) == 4:
        mask_return = np.transpose(_mask, (1, 2, 3, 0))
    else:
        mask_return = np.transpose(_mask, (1, 2, 0))

    return mask_return


def total_loss(weights, Pretrained_Weight_Path, X_train):

    def add_loss(y_true, y_pred):
        def s_weighted_crossentropy(y_true, y_pred):
            ce = K.categorical_crossentropy(y_true, y_pred)

            # weighted calc
            weight_map_tmp = y_true * 1.0
            weight_map = K.sum(weight_map_tmp, axis=-1)
            weighted_ce = weight_map * ce

            return K.mean(weighted_ce)

        image_width = 512
        image_height = 512

        #t_model = unet_vanilla(input_size=(image_width, image_height, 1), base=2, uncertainty=False, trainable=False)
        #model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        #t_model.load_weights(Pretrained_Weight_Path)
        #t_model.summary()
        #prediction = t_model.predict(X_train)

        def t_weighted_crossentropy(y_true, y_pred):
            ce = K.categorical_crossentropy(y_true, y_pred)

            # weighted calc
            weight_map_tmp = y_true * 1.0
            weight_map = K.sum(weight_map_tmp, axis=-1)
            weighted_ce = weight_map * ce

            return K.mean(weighted_ce)

        loss_ce = s_weighted_crossentropy(y_true, y_pred)
        loss_dist = t_weighted_crossentropy(y_true, y_pred)
    
        return loss_ce + loss_dist
    
    return add_loss

def weighted_crossentropy(weights):
    def weighted_crossentropy(y_true, y_pred):
        ce = K.categorical_crossentropy(y_true, y_pred)

        # weighted calc
        weight_map_tmp = y_true * weights
        weight_map = K.sum(weight_map_tmp, axis=-1)
        weighted_ce = weight_map * ce

        return K.mean(weighted_ce)

    return weighted_crossentropy

def iou(y_true, y_pred):
    intersection = K.sum(y_true[:, :, :, 1] * y_pred[:, :, :, 1])
    union = K.sum(y_true[:, :, :, 1]) + K.sum(y_pred[:, :, :, 1]) - intersection
    iou = K.mean((intersection) / (union))
    return iou

def iou_binarized(y_true, y_pred):
    y_pred_b = K.argmax(y_pred, axis=-1)
    y_pred_b = tf.cast(y_pred_b, 'float32')
    intersection = K.sum(y_true[:, :, :, 1] * y_pred_b)
    union = K.sum(y_true[:, :, :, 1]) + K.sum(y_pred_b) - intersection
    iou_binarized = K.mean((intersection) / (union))
    return iou_binarized

def recall(y_true, y_pred):
    tp = K.sum(y_true[:, :, :, 1] * y_pred[:, :, :, 1])
    tp_plus_fn = K.sum(y_true[:, :, :, 1]) 
    recall = K.mean((tp) / (tp_plus_fn))
    return recall

def precision(y_true, y_pred):
    tp = K.sum(y_true[:, :, :, 1] * y_pred[:, :, :, 1])
    tp_plus_fp = K.sum(y_pred[:, :, :, 1]) 
    precision = K.mean((tp) / (tp_plus_fp))
    return precision

def F1_score(y_true, y_pred):
    F1_score = 2*recall(y_true, y_pred)*precision(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred))
    return F1_score

def iou_prediction(label, mask):
    intersection = K.sum(label[:,:,1] * mask[:,:,1])
    union = K.sum(label[:,:,1]) + K.sum(mask[:,:,1]) - intersection
    iou = K.mean((intersection) / (union))
    return iou

def recall_prediction(label, mask):
    tp = K.sum(label[:,:,1] * mask[:,:,1])
    tp_plus_fn = K.sum(label[:,:,1])
    recall = K.mean((tp) / (tp_plus_fn))
    return recall

def precision_prediction(label, mask):
    tp = K.sum(label[:,:,1] * mask[:,:,1])
    tp_plus_fp = K.sum(mask[:,:,1])
    precision = K.mean((tp) / (tp_plus_fp))
    return precision

def F1_score_prediction(label, mask):
    F1_score = 2*recall_prediction(label, mask)*precision_prediction(label, mask)/(recall_prediction(label, mask)+precision_prediction(label, mask))
    return F1_score
    
def plot_metrics(history, n_epochs, save_path=""):
    data = history.history
    for i in data.keys():
        plt.plot(np.arange(n_epochs), data[i], label=i)
    plt.xlabel("number of epochs")
    plt.ylabel("metrics")
    plt.title("Metrics of training")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path + "/" + "train_metrics.svg")
    plt.show()


def plot_metrics(history, n_epochs, save_path=""):
    data = history.history
    for i in data.keys():
        plt.plot(np.arange(n_epochs), data[i], label=i)
    plt.xlabel("number of epochs")
    plt.ylabel("metrics")
    plt.title("Metrics of training")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path + "/" + "train_metrics.svg")
    #plt.show()

def predict(X,Y,prediction, overlay_with_label=True, overlay_with_info=True):

    mask = np.argmax(prediction, axis=2).astype('float32').reshape(512, 512)
    Pred_End_onehot = mask2onehot(mask, 2)
    IOU = iou_prediction(Y, Pred_End_onehot)
    RECALL = recall_prediction(Y, Pred_End_onehot)
    PRECISION = precision_prediction(Y, Pred_End_onehot)
    F1_SCORE = F1_score_prediction(Y, Pred_End_onehot)
    if overlay_with_label:
        label_overlay = np.argmax(Y, axis=2).astype('float32')
        image_pred = overlay_softmax(image=X, mask=mask, label=label_overlay)
    else:
        image_pred = overlay_softmax(X, mask)
    if overlay_with_info:
        text = [
                    f"IOU        {IOU}\n",
                    f"RECALL     {RECALL}\n",
                    f"PRECISION  {PRECISION}\n",
                    f"F1_SCORE   {F1_SCORE}\n",
                ]   
                      
        image_pred = overlay_info(image_pred,text)
    else:
        pass

    return image_pred


