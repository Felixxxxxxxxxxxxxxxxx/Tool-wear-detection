
from math import nan
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt



def generate_form(record):

    df = pd.DataFrame(columns=record.keys())

    for key in df.columns:
        print(key)
        df[key]=record[key]

    print(df)
    file_path = sys.path[0] 
    with pd.ExcelWriter(file_path + '/fine_tune') as writer:
        sheetname = "metrics"
        df.to_excel( writer,sheet_name='{}'.format(sheetname))


def metrics_arrange(record_1,record_2,record_3,record_4, eval_dataset, metric):

    data1 = record_1[eval_dataset][:,metric[0]]
    data2 = record_2[eval_dataset][:,metric[0]]
    data2 = data2[1:201]
    data3 = record_3[eval_dataset][:,metric[0]]
    data3 = data3[1:201]
    data4 = record_4[eval_dataset][:,metric[0]]
    data4 = data4[1:201]
    data_m1 = np.concatenate((data1,data2,data3,data4), axis = 0) 
    print (np.shape(data_m1))

    data1 = record_1[eval_dataset][:,metric[1]]
    data2 = record_2[eval_dataset][:,metric[1]]
    data2 = data2[1:201]
    data3 = record_3[eval_dataset][:,metric[1]]
    data3 = data3[1:201]
    data4 = record_4[eval_dataset][:,metric[1]]
    data4 = data4[1:201]
    data_m2 = np.concatenate((data1,data2,data3,data4), axis = 0) 
    print (np.shape(data_m2))

    return data_m1, data_m2
    



def generate_pics(record_1,record_2,record_3,record_4,name, title):
    
    print(record_1.keys())
    n_epochs= np.shape(record_1["eval_0"])[0]
    eval = ["eval_0","eval_1","eval_2","eval_3"]
    #keys = ['loss', 'iou_binarized', 'iou', 'recall', 'precision', 'F1_score']
    
    

    color_shown = [[245/255,130/255,32/255],[23/255,156/255,125/255],[0/255,91/255,127/255],[178/255,210/255,53/255],[57/255,193/255,205/255]]
    font1 = {'family' : 'Arial',
    'weight' : 'normal',
    'size'   : 18,
    }
    plt.figure(figsize=(10.5, 7))
    
    if  title == "IOU":
        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[0], metric = [1,5])
        l1, = plt.plot(np.arange(0,4*n_epochs-3), m1, label="IOU", color= color_shown[0], linewidth=1.0, linestyle='-')
        #l1, =plt.plot(np.arange(0,4*n_epochs-3), m2, label="F1 Score", color= color_shown[0], linewidth=1.0, linestyle='-')

        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[1], metric = [1,5])
        l2, = plt.plot(np.arange(201,4*n_epochs-3), m1[201:4*n_epochs-3], label="IOU", color= color_shown[1], linewidth=1.0, linestyle='-')
        #l2, = plt.plot(np.arange(201,4*n_epochs-3), m2[201:4*n_epochs-3], label="F1 Score", color= color_shown[1], linewidth=1.0, linestyle='-')

        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[2], metric = [1,5])
        l3, = plt.plot(np.arange(401,4*n_epochs-3), m1[401:4*n_epochs-3], label="IOU", color= color_shown[2], linewidth=1.0, linestyle='-')
        #l3, = plt.plot(np.arange(401,4*n_epochs-3), m2[401:4*n_epochs-3], label="F1 Score", color= color_shown[2], linewidth=1.0, linestyle='-')

        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[3], metric = [1,5])
        l4, = plt.plot(np.arange(601,4*n_epochs-3), m1[601:4*n_epochs-3], label="IOU", color= color_shown[3], linewidth=1.0, linestyle='-')
        #l4, = plt.plot(np.arange(601,4*n_epochs-3), m2[601:4*n_epochs-3], label="F1 Score", color= color_shown[3], linewidth=1.0, linestyle='-')

    if  title == "F1 Score":
        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[0], metric = [1,5])
        #l1, = plt.plot(np.arange(0,4*n_epochs-3), m1, label="IOU", color= color_shown[0], linewidth=1.0, linestyle='-')
        l1, =plt.plot(np.arange(0,4*n_epochs-3), m2, label="F1 Score", color= color_shown[0], linewidth=1.0, linestyle='-')

        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[1], metric = [1,5])
        #l2, = plt.plot(np.arange(201,4*n_epochs-3), m1[201:4*n_epochs-3], label="IOU", color= color_shown[1], linewidth=1.0, linestyle='-')
        l2, = plt.plot(np.arange(201,4*n_epochs-3), m2[201:4*n_epochs-3], label="F1 Score", color= color_shown[1], linewidth=1.0, linestyle='-')

        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[2], metric = [1,5])
        #l3, = plt.plot(np.arange(401,4*n_epochs-3), m1[401:4*n_epochs-3], label="IOU", color= color_shown[2], linewidth=1.0, linestyle='-')
        l3, = plt.plot(np.arange(401,4*n_epochs-3), m2[401:4*n_epochs-3], label="F1 Score", color= color_shown[2], linewidth=1.0, linestyle='-')

        m1, m2 = metrics_arrange(record_1,record_2,record_3,record_4, eval[3], metric = [1,5])
        #l4, = plt.plot(np.arange(601,4*n_epochs-3), m1[601:4*n_epochs-3], label="IOU", color= color_shown[3], linewidth=1.0, linestyle='-')
        l4, = plt.plot(np.arange(601,4*n_epochs-3), m2[601:4*n_epochs-3], label="F1 Score", color= color_shown[3], linewidth=1.0, linestyle='-')

    plt.xlabel("Number of epochs",font = font1)
    plt.xticks(np.arange(0,4*n_epochs,200),font = font1)
    plt.ylabel("Value",font = font1)
    plt.yticks(np.arange(0,1,0.2),font = font1)
    plt.title(title + f" of {name}",font = font1)   
    ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.legend(handles=[l1, l2, l3, l4], labels=['dataset_0', 'dataset_1', 'dataset_2', 'dataset_3'],loc="lower left",prop = font1)
    
    #plt.tight_layout()
    plt.savefig( f"./Methods/{name}/" + "{}.png".format(title))
    #plt.show()

if __name__ == "__main__":

    names = ["LWF","EF","DF","FT","EWC","LWF + EF","Replay (0.2)","Replay (0.1)","Replay (0.05)"]
    
    titles = ["F1 Score","IOU"]

    for name in names:
        for title in titles:

            history_path_1 = f"E:\Bachelorarbeit\WP3\Methods\{name}\Model_0\evaluation.npy"
            record_1 = np.load(history_path_1,allow_pickle=True).item()

            history_path_2 = f"E:\Bachelorarbeit\WP3\Methods\{name}\Model_1\evaluation.npy"
            record_2 = np.load(history_path_2,allow_pickle=True).item()

            history_path_3 = f"E:\Bachelorarbeit\WP3\Methods\{name}\Model_2\evaluation.npy"
            record_3 = np.load(history_path_3,allow_pickle=True).item()

            history_path_3 = f"E:\Bachelorarbeit\WP3\Methods\{name}\Model_3\evaluation.npy"
            record_4 = np.load(history_path_3,allow_pickle=True).item()

            generate_pics(record_1,record_2,record_3,record_4, name, title)