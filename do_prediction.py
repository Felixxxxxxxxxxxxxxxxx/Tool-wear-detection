from PIL import Image
from functions import *
from model import *
import random

SEED = 20
random.seed(SEED)

class data():

    def __init__(self, **kw):

        data_path =[
                r'E:\Bachelorarbeit/Data/AVM_2022/Turning/Train_aug/',
                r'E:\Bachelorarbeit/Data/AVM_2022/Turning/Inference_aug/',
                r"E:\Bachelorarbeit/Data/AVM_2022/Milling/Train_aug/",
                r"E:\Bachelorarbeit/Data/AVM_2022/Milling/Inference_aug/", 
                ]

        self.X_train_0, self.Y_train_0, self.X_val_0, self.Y_val_0 = get_train_dataset(data_path[0])
        self.X_train_1, self.Y_train_1, self.X_val_1, self.Y_val_1 = get_train_dataset(data_path[1])
        self.X_train_2, self.Y_train_2, self.X_val_2, self.Y_val_2 = get_train_dataset(data_path[2])
        self.X_train_3, self.Y_train_3, self.X_val_3, self.Y_val_3 = get_train_dataset(data_path[3])

    def choose_data(self, i):

        if i==0:
            self.X_train = self.X_train_0
            self.Y_train = self.Y_train_0
        elif i==1:
            self.X_train = self.X_train_1
            self.Y_train = self.Y_train_1
            
        elif i==2:
            self.X_train = self.X_train_2
            self.Y_train = self.Y_train_2
            
        elif i==3:
            self.X_train = self.X_train_3
            self.Y_train = self.Y_train_3


model = unet_vanilla(input_size=(512, 512, 1), base=2, uncertainty=False, trainable=False)
#model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.load_weights("E:\Bachelorarbeit\WP1\model_retrained_with_inference_img\model_trained_with_random_mixed\Model_weights=[0.5, 2.0]_ini_lr=0.0001\model_weights/")
image_pred_path = "./prediction/"            

dataset = data()
dataset.choose_data(1)
X_train, Y_train = dataset.X_train, dataset.Y_train
X_train_0, Y_train_0, X_val_0, Y_val_0 = dataset.X_train_0, dataset.Y_train_0, dataset.X_val_0, dataset.Y_val_0 
_, _, X_val_1, Y_val_1 = dataset.X_train_1, dataset.Y_train_1, dataset.X_val_1, dataset.Y_val_1 
_, _, X_val_2, Y_val_2 = dataset.X_train_2, dataset.Y_train_2, dataset.X_val_2, dataset.Y_val_2 
_, _, X_val_3, Y_val_3 = dataset.X_train_3, dataset.Y_train_3, dataset.X_val_3, dataset.Y_val_3


index = 12
X_show = np.expand_dims(X_train[index], 0)
Y_show = np.expand_dims(Y_train[index], 0)

index = 14
X_show = np.append(X_show,np.expand_dims(X_train[index], 0),axis = 0)
Y_show = np.append(Y_show,np.expand_dims(Y_train[index], 0),axis = 0)

index = 16
X_show = np.append(X_show,np.expand_dims(X_train[index], 0),axis = 0)
Y_show = np.append(Y_show,np.expand_dims(Y_train[index], 0),axis = 0)

for idx in range (0,len(X_show[:,0,0,0])):
    Pred_Dim = Y_show.shape
    prediction = model.predict(np.expand_dims(X_show[idx], 0))
    prediction = prediction.reshape(Pred_Dim[1], Pred_Dim[2], Pred_Dim[3])
    X = X_show[idx].reshape(512, 512)
    Y = Y_show[idx]
    image_pred = predict(X, Y, prediction,True, True)
    #image_pred.show()
    folder = os.path.exists(image_pred_path)

    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(image_pred_path)            #makedirs 创建文件时如果路径不存在会创建这个
    else:
        pass
    image_pred.save(image_pred_path + f"{idx}.1.png")
    #wx_push.Push("important","image",name=image_pred_path + f"validation{idx}.png")