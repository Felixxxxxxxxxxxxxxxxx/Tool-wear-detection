from functions import *
from model import *
from wx_push import wx_push
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  ModelCheckpoint, TensorBoard
import datetime,time
from multiprocessing import Process


PERCENT = 0.1

class data():

    def __init__(self,**kw):
        
        self.past_X = []
        self.past_Y = []

        data_path =[
                r"/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Turning/Train_aug/",
                r'/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Turning/Inference_aug/',
                r"/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Milling/Train_aug/",
                r"/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Milling/Inference_aug/",
                ]

        self.X_train_0, self.Y_train_0, self.X_val_0, self.Y_val_0 = get_train_dataset(data_path[0])
        self.X_train_1, self.Y_train_1, self.X_val_1, self.Y_val_1 = get_train_dataset(data_path[1])
        self.X_train_2, self.Y_train_2, self.X_val_2, self.Y_val_2 = get_train_dataset(data_path[2])
        self.X_train_3, self.Y_train_3, self.X_val_3, self.Y_val_3 = get_train_dataset(data_path[3])

    def generate_data(self,i):

        if i==0:
            self.X_train = self.X_train_0
            self.Y_train = self.Y_train_0
        elif i==1:
            self.past_X = np.concatenate((self.X_train_0, self.X_val_0))
            self.past_Y = np.concatenate((self.Y_train_0, self.Y_val_0))
            self.X_train = self.X_train_1
            self.Y_train = self.Y_train_1
        elif i==2:
            self.past_X = np.concatenate((self.X_train_0, self.X_val_0, self.X_train_1, self.X_val_1))
            self.past_Y = np.concatenate((self.Y_train_0, self.Y_val_0, self.Y_train_1, self.Y_val_1))
            self.X_train = self.X_train_2
            self.Y_train = self.Y_train_2
        elif i==3:
            self.past_X = np.concatenate((self.X_train_0, self.X_val_0, self.X_train_1, self.X_val_1, self.X_train_2, self.X_val_2))
            self.past_Y = np.concatenate((self.Y_train_0, self.Y_val_0, self.Y_train_1, self.Y_val_1, self.Y_train_2, self.Y_val_2))
            self.X_train = self.X_train_3
            self.Y_train = self.Y_train_3
        
        if i!=0:

            print("X_past  ",np.shape(self.past_X))
            print("Y_past  ",np.shape(self.past_Y))

            sample_num = int(PERCENT * len(self.X_train)) 
            sample_list = [i for i in range(len(self.past_X))] 
            sample_list = random.sample(sample_list, sample_num) 

            print("sample_list  ",sample_list)

            self.past_X = self.past_X[sample_list] 
            self.past_Y = self.past_Y[sample_list]

            time.sleep(3)

        else:
            print("X_past  ",np.shape(self.past_X))
            print("Y_past  ",np.shape(self.past_Y))

            time.sleep(3)


    def replay(self,i):

        self.generate_data(i)

        self.Pretrained_Weight_Path = f"/rwthfs/rz/cluster/home/de532237/cancer/Bachelorarbeit/WP2/Methods/replay_p={PERCENT}/Model_{i-1}/checkpoint/model.ckpt"

        _, _, X_val_0, Y_val_0 = self.X_train_0, self.Y_train_0, self.X_val_0, self.Y_val_0 
        _, _, X_val_1, Y_val_1 = self.X_train_1, self.Y_train_1, self.X_val_1, self.Y_val_1 
        _, _, X_val_2, Y_val_2 = self.X_train_2, self.Y_train_2, self.X_val_2, self.Y_val_2 
        _, _, X_val_3, Y_val_3 = self.X_train_3, self.Y_train_3, self.X_val_3, self.Y_val_3


        if i!=0:
            past_X = self.past_X
            past_Y = self.past_Y
            X_train = np.concatenate((self.past_X, self.X_train))
            Y_train = np.concatenate((self.past_Y, self.Y_train))
        else:
            X_train = self.X_train_0
            Y_train = self.Y_train_0

        # Data Shuffle for Training
        SEED = 20
        random.seed(SEED)
        train_index = np.arange(len(X_train))
        random.shuffle(train_index)
        Dat_train = X_train[train_index]
        Lbl_train = Y_train[train_index]
        X_train  = Dat_train
        Y_train  = Lbl_train

        print("X_train  ",np.shape(X_train))
        print("Y_train  ",np.shape(Y_train))

        time.sleep(3)

        if_test = False

        if if_test:
            Model_Name = "Model_Test"
        else :
            Model_Name = f"Model_{i}"

        Result_Path = f"/home/de532237/cancer/Bachelorarbeit/WP2/Methods/replay_p={PERCENT}/" + Model_Name + '/'

        ###############################################################################
        
        """
        Define Dir
        """

        weight_1  = 1
        
        if i == 0:
            n_epochs = 200
            batch_size = 8
            ini_learning_rate = 1e-4
            weight_1  = 1
        elif i == 1:
            n_epochs = 200
            batch_size = 8
            ini_learning_rate = 1e-4
            weight_1  = 1
        elif i == 2:
            n_epochs = 200
            batch_size = 8
            ini_learning_rate = 1e-4
            weight_1  = 1
        elif i == 3:
            n_epochs = 200
            batch_size = 8
            ini_learning_rate = 1e-4
            weight_1  = 1

        weight_2 = 1/weight_1
        weights = [weight_1,weight_2]

        image_width = 512
        image_height = 512

        Save_and_Show_Result = True
        overlay_with_label = True
        overlay_with_info = True

        checkpoint_path = Result_Path + '/checkpoint/model.ckpt'
        image_pred_path = Result_Path + "/image_output/"
        log_dir = Result_Path  + "/logs/"
        txt_path = Result_Path

        ###############################################################################
        """
        Model Setting for Training/Validation
        """

        optimizer = Adam(
                            learning_rate=ini_learning_rate,
                            beta_1=0.9,
                            beta_2=0.999,)

        loss = weighted_crossentropy(weights)

        class CustomCallback(tf.keras.callbacks.Callback):

            eval_0 = []
            eval_1 = []
            eval_2 = []
            eval_3 = []

            def on_train_begin(self, epoch, logs=None):
                print("train begins.")
                eval_0 = model.evaluate(X_val_0, Y_val_0, verbose=0)
                CustomCallback.eval_0 = np.expand_dims(eval_0, axis = 0)
                eval_1 = model.evaluate(X_val_1, Y_val_1, verbose=0)
                CustomCallback.eval_1 = np.expand_dims(eval_1, axis = 0)
                eval_2 = model.evaluate(X_val_2, Y_val_2, verbose=0)
                CustomCallback.eval_2 = np.expand_dims(eval_2, axis = 0)
                eval_3 = model.evaluate(X_val_3, Y_val_3, verbose=0)
                CustomCallback.eval_3 = np.expand_dims(eval_3, axis = 0)

            def on_epoch_end(self, epoch, logs=None):
                print("epoch ends.")
                eval_0 = model.evaluate(X_val_0, Y_val_0, verbose=0)
                CustomCallback.eval_0 = np.append(CustomCallback.eval_0,np.expand_dims(eval_0, axis = 0), axis = 0)
                eval_1 = model.evaluate(X_val_1, Y_val_1, verbose=0)
                CustomCallback.eval_1 = np.append(CustomCallback.eval_1,np.expand_dims(eval_1, axis = 0), axis = 0)
                eval_2 = model.evaluate(X_val_2, Y_val_2, verbose=0)
                CustomCallback.eval_2 = np.append(CustomCallback.eval_2,np.expand_dims(eval_2, axis = 0), axis = 0)
                eval_3 = model.evaluate(X_val_3, Y_val_3, verbose=0)
                CustomCallback.eval_3 = np.append(CustomCallback.eval_3,np.expand_dims(eval_3, axis = 0), axis = 0)
            
        metrics = [iou_binarized, iou, recall, precision, F1_score]

        if i!=0:
            model = unet_vanilla(input_size=(image_width, image_height, 1), base=2, uncertainty=False, trainable=True)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            Pretrained_Weight_Path = self.Pretrained_Weight_Path
            model.load_weights(Pretrained_Weight_Path)
        else:
            model = unet_vanilla(input_size=(image_width, image_height, 1), base=2, uncertainty=False, trainable=True)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            pass

        ###############################################################################
        """
        Train Model 
        """
        
        tf.keras.backend.clear_session()

        callbacks = [
                ModelCheckpoint(filepath=checkpoint_path, monitor="loss", verbose=0, save_best_only=False,save_weights_only=True,mode="min"),
                CustomCallback(),
                TensorBoard(log_dir=log_dir)]

        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs, callbacks=callbacks, verbose=2, shuffle = False)
        plot_metrics(history=history, n_epochs=history.epoch[-1]+1, save_path=Result_Path)
        np.save(Result_Path + "histroy.npy", history.history)

        evaluation = {}
        evaluation["eval_0"] = CustomCallback.eval_0
        evaluation["eval_1"] = CustomCallback.eval_1
        evaluation["eval_2"] = CustomCallback.eval_2
        evaluation["eval_3"] = CustomCallback.eval_3
        np.save(Result_Path + "evaluation.npy", evaluation)

        txt_file = open(txt_path + f"Info_{i}.txt", "w", encoding="utf-8")

        data = history.history
        for key in data.keys():
            txt_file.write(f"{key} = {round(data[key][-1],4)}\n")

        txt_file.write(f"\n\n{model.metrics_names}\n")
        data = evaluation
        
        for index in evaluation.keys():
            txt_file.write("\nevaluation length = {}\n".format(len(data[index])))
            txt_file.write(f"{index} first = {data[index][0]}\n")
            txt_file.write(f"{index} last = {data[index][-1]}\n")

        txt_file.write("\nepochs = {}\n".format(history.epoch[-1]+1))
        txt_file.write("weights = {}\n".format(weights))
        txt_file.write("learning_rate_ini = {}\n".format(ini_learning_rate))
        txt_file.write("time = {}\n".format(datetime.datetime.now()))
        txt_file.close()

        model.save_weights(Result_Path+'model_weights/')
        model.save(Result_Path+'model_save/model.h5')
        # tensorboard --logdir=E:\Bachelorarbeit\WP1\Model_avm2022\logs

        ###############################################################################
        """
        Output Segmentation Result
        """

        if Save_and_Show_Result:

            index_0 = random.randint(0,len(X_val_0[:,0,0,0])-1)
            X_show = np.expand_dims(X_val_0[index_0], 0)
            Y_show = np.expand_dims(Y_val_0[index_0], 0)
            index_1 = random.randint(0,len(X_val_1[:,0,0,0])-1)
            X_show = np.append(X_show,np.expand_dims(X_val_1[index_1], 0),axis = 0)
            Y_show = np.append(Y_show,np.expand_dims(Y_val_1[index_1], 0),axis = 0)
            index_2 = random.randint(0,len(X_val_2[:,0,0,0])-1)
            X_show = np.append(X_show,np.expand_dims(X_val_2[index_2], 0),axis = 0)
            Y_show = np.append(Y_show,np.expand_dims(Y_val_2[index_2], 0),axis = 0)
            index_3 = random.randint(0,len(X_val_3[:,0,0,0])-1)
            X_show = np.append(X_show,np.expand_dims(X_val_3[index_3], 0),axis = 0)
            Y_show = np.append(Y_show,np.expand_dims(Y_val_3[index_3], 0),axis = 0)
            if i!=0:
                index_4 = random.randint(0,len(past_X[:,0,0,0])-1)
                X_show = np.append(X_show,np.expand_dims(past_X[index_4], 0),axis = 0)
                Y_show = np.append(Y_show,np.expand_dims(past_Y[index_4], 0),axis = 0)
                index_5 = random.randint(0,len(past_X[:,0,0,0])-1)
                X_show = np.append(X_show,np.expand_dims(past_X[index_5], 0),axis = 0)
                Y_show = np.append(Y_show,np.expand_dims(past_Y[index_5], 0),axis = 0)

            for idx in range (0,len(X_show[:,0,0,0])):
                Pred_Dim = Y_show.shape
                prediction = model.predict(np.expand_dims(X_show[idx], 0))
                prediction = prediction.reshape(Pred_Dim[1], Pred_Dim[2], Pred_Dim[3])
                X = X_show[idx].reshape(512, 512)
                Y = Y_show[idx]
                image_pred = predict(X, Y, prediction,overlay_with_label, overlay_with_info)
                #image_pred.show()
                folder = os.path.exists(image_pred_path)

                if not folder:                   #???????????????????????????????????????????????????????????????
                    os.makedirs(image_pred_path)            #makedirs ???????????????????????????????????????????????????
                else:
                    pass

                image_pred.save(image_pred_path + f"test{idx}.png")
                #wx_push.Push("important","image",name=image_pred_path + f"validation{idx}.png")
        
        wx_push.Push("important","file",name=txt_path + f"Info_{i}.txt")

    


if __name__ == '__main__' :

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    dataset = data()
     
    for i in range (1,4):
        p = Process(target = dataset.replay, args = (i,))
        p.start()
        p.join() 
        time.sleep(1) 

    wx_push.Push("important","text",content="Finish.")


