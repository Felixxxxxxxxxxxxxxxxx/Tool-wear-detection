
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from functions import *
from model import *
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.callbacks import  ModelCheckpoint, TensorBoard
import datetime
from wx_push import *

I = 3
alpha = 0.7
ini_learning_rate = 1e-4
weights_student = 1
weights_dist = 1
epochs = 300
batch_size = 8

Save_and_Show_Result = True
overlay_with_label = True
overlay_with_info = True

class data():

    def __init__(self, **kw):

        data_path =[
                r'/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Turning/Train_aug/',
                r'/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Turning/Inference_aug/',
                r"/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Milling/Train_aug/",
                r"/home/de532237/cancer/Bachelorarbeit/Data/AVM_2022/Milling/Inference_aug/", 
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

class CustomCallback(tf.keras.callbacks.Callback):

    eval_0 = []
    eval_1 = []
    eval_2 = []
    eval_3 = []

    def on_train_begin(self, epoch, logs=None):
        print("train begins.")
        eval_0 = student.evaluate(X_val_0, Y_val_0, verbose=0)
        CustomCallback.eval_0 = np.expand_dims(eval_0, axis = 0)
        eval_1 = student.evaluate(X_val_1, Y_val_1, verbose=0)
        CustomCallback.eval_1 = np.expand_dims(eval_1, axis = 0)
        eval_2 = student.evaluate(X_val_2, Y_val_2, verbose=0)
        CustomCallback.eval_2 = np.expand_dims(eval_2, axis = 0)
        eval_3 = student.evaluate(X_val_3, Y_val_3, verbose=0)
        CustomCallback.eval_3 = np.expand_dims(eval_3, axis = 0)

    def on_epoch_end(self, epoch, logs=None):
        print("epoch ends.")
        eval_0 = student.evaluate(X_val_0, Y_val_0, verbose=0)
        CustomCallback.eval_0 = np.append(CustomCallback.eval_0,np.expand_dims(eval_0, axis = 0), axis = 0)
        eval_1 = student.evaluate(X_val_1, Y_val_1, verbose=0)
        CustomCallback.eval_1 = np.append(CustomCallback.eval_1,np.expand_dims(eval_1, axis = 0), axis = 0)
        eval_2 = student.evaluate(X_val_2, Y_val_2, verbose=0)
        CustomCallback.eval_2 = np.append(CustomCallback.eval_2,np.expand_dims(eval_2, axis = 0), axis = 0)
        eval_3 = student.evaluate(X_val_3, Y_val_3, verbose=0)
        CustomCallback.eval_3 = np.append(CustomCallback.eval_3,np.expand_dims(eval_3, axis = 0), axis = 0)

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha,
    ):

        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            teacher_loss = self.student_loss_fn(y, teacher_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = self.distillation_loss_fn(teacher_predictions, student_predictions)

            loss =  student_loss +  self.alpha * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"s_loss": student_loss, "t_loss": teacher_loss, "dist_loss": distillation_loss, "loss": loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

if __name__ == '__main__' :

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Start.")

    # Dir define
    Model_Name = f"Model_{I}"
    Result_Path = f"/home/de532237/cancer/Bachelorarbeit/WP2/Methods/lwf_ef/" + Model_Name + '/'
    checkpoint_path = Result_Path + '/checkpoint/model.ckpt'
    image_pred_path = Result_Path + "/image_output/"
    log_dir = Result_Path  + "/logs/"
    txt_path = Result_Path
    # Create the teacher
    image_width = 512
    image_height = 512
    teacher = unet_vanilla(input_size=(image_width, image_height, 1), base=2, uncertainty=False, trainable=False)
    # Create the student
    student = unet_vanilla_fe(input_size=(image_width, image_height, 1), base=2, uncertainty=False, trainable=True)

    # Prepare the train and test dataset.
    
    dataset = data()
    dataset.choose_data(I)
    X_train, Y_train = dataset.X_train, dataset.Y_train
    _, _, X_val_0, Y_val_0 = dataset.X_train_0, dataset.Y_train_0, dataset.X_val_0, dataset.Y_val_0 
    _, _, X_val_1, Y_val_1 = dataset.X_train_1, dataset.Y_train_1, dataset.X_val_1, dataset.Y_val_1 
    _, _, X_val_2, Y_val_2 = dataset.X_train_2, dataset.Y_train_2, dataset.X_val_2, dataset.Y_val_2 
    _, _, X_val_3, Y_val_3 = dataset.X_train_3, dataset.Y_train_3, dataset.X_val_3, dataset.Y_val_3

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

    # Load and compile model
    optimizer = Adam(
                        learning_rate=ini_learning_rate,
                        beta_1=0.9,
                        beta_2=0.999,)

    loss_student = weighted_crossentropy([weights_student, 1/weights_student])
    loss_dist = weighted_crossentropy([weights_dist, 1/weights_dist])

    metrics = [iou_binarized, iou, recall, precision, F1_score]

    Pretrained_Weight_Path = f"/home/de532237/cancer/Bachelorarbeit/WP2/Methods/lwf_ef/Model_{I-1}/model_weights/"
    teacher.load_weights(Pretrained_Weight_Path)
    student.load_weights(Pretrained_Weight_Path)
    student.compile(optimizer=optimizer, loss=loss_student, metrics=metrics)

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=optimizer,
        metrics=metrics,
        student_loss_fn=loss_student,
        distillation_loss_fn=loss_dist,
        alpha=alpha,
    )
    callbacks = [
                ModelCheckpoint(filepath=checkpoint_path, monitor="loss", verbose=0, save_best_only=False,save_weights_only=True,mode="min"),
                CustomCallback(),
                TensorBoard(log_dir=log_dir)]
    # Distill teacher to student
    history = distiller.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2, shuffle=False,)
    plot_metrics(history=history, n_epochs=history.epoch[-1]+1, save_path=Result_Path)
    np.save(Result_Path + "histroy.npy", history.history)

    evaluation = {}
    evaluation["eval_0"] = CustomCallback.eval_0
    evaluation["eval_1"] = CustomCallback.eval_1
    evaluation["eval_2"] = CustomCallback.eval_2
    evaluation["eval_3"] = CustomCallback.eval_3
    np.save(Result_Path + "evaluation.npy", evaluation)

    txt_file = open(txt_path + f"Info_{I}.txt", "w", encoding="utf-8")

    data = history.history
    for key in data.keys():
        txt_file.write(f"{key} = {round(data[key][-1],4)}\n")
    
    txt_file.write(f"alpha = {alpha}\n")

    txt_file.write(f"\n\n{distiller.metrics_names}\n")
    data = evaluation
    
    for index in evaluation.keys():
        txt_file.write("\nevaluation length = {}\n".format(len(data[index])))
        txt_file.write(f"{index} first = {data[index][0]}\n")
        txt_file.write(f"{index} last = {data[index][-1]}\n")

    txt_file.write("\nepochs = {}\n".format(history.epoch[-1]+1))
    txt_file.write("weights = {} {}\n".format(weights_student, weights_dist))
    txt_file.write("learning_rate_ini = {}\n".format(ini_learning_rate))
    txt_file.write("time = {}\n".format(datetime.datetime.now()))
    txt_file.close()

    student.save_weights(Result_Path+'model_weights/')
    # Save the entire model as a SavedModel.
    #student.save('saved_model/my_model')
    student.save(Result_Path+'model_save/model.h5')
    # tensorboard --logdir=E:\Bachelorarbeit\WP1\Model_avm2022\logs

    ###############################################################################
    """
    Output Segmentation Result
    """

    if Save_and_Show_Result:

        index = random.randint(0,len(X_train[:,0,0,0])-1)
        X_show = np.expand_dims(X_train[index], 0)
        Y_show = np.expand_dims(Y_train[index], 0)

        index_0 = random.randint(0,len(X_val_0[:,0,0,0])-1)
        X_show = np.append(X_show,np.expand_dims(X_val_0[index_0], 0),axis = 0)
        Y_show = np.append(Y_show,np.expand_dims(Y_val_0[index_0], 0),axis = 0)
        index_1 = random.randint(0,len(X_val_1[:,0,0,0])-1)
        X_show = np.append(X_show,np.expand_dims(X_val_1[index_1], 0),axis = 0)
        Y_show = np.append(Y_show,np.expand_dims(Y_val_1[index_1], 0),axis = 0)
        index_2 = random.randint(0,len(X_val_2[:,0,0,0])-1)
        X_show = np.append(X_show,np.expand_dims(X_val_2[index_2], 0),axis = 0)
        Y_show = np.append(Y_show,np.expand_dims(Y_val_2[index_2], 0),axis = 0)
        index_3 = random.randint(0,len(X_val_3[:,0,0,0])-1)
        X_show = np.append(X_show,np.expand_dims(X_val_3[index_3], 0),axis = 0)
        Y_show = np.append(Y_show,np.expand_dims(Y_val_3[index_3], 0),axis = 0)

        model = unet_vanilla(input_size=(image_width, image_height, 1), base=2, uncertainty=False, trainable=False)
        #model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.load_weights(Result_Path+'model_weights/')


        for idx in range (0,len(X_show[:,0,0,0])):
            Pred_Dim = Y_show.shape
            prediction = model.predict(np.expand_dims(X_show[idx], 0))
            prediction = prediction.reshape(Pred_Dim[1], Pred_Dim[2], Pred_Dim[3])
            X = X_show[idx].reshape(512, 512)
            Y = Y_show[idx]
            image_pred = predict(X, Y, prediction,overlay_with_label, overlay_with_info)
            #image_pred.show()
            folder = os.path.exists(image_pred_path)

            if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(image_pred_path)            #makedirs 创建文件时如果路径不存在会创建这个
            else:
                pass
            if idx == 0 :
                image_pred.save(image_pred_path + f"train{idx}.png")
            else :
                image_pred.save(image_pred_path + f"test{idx}.png")
            #wx_push.Push("important","image",name=image_pred_path + f"validation{idx}.png")
    
    wx_push.Push("important","file",name=txt_path + f"Info_{I}.txt")