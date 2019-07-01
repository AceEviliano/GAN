import numpy as np

__PRINTED__ = 0


def make_image_grid(sample, row, col):

    img = sample[0]
    #label = sample[1]

    r = 0
    c = 0
    inx = 0

    shape = img[0].size()

    if shape[0]==1:
        dim = 1
    else :
        dim = 3

    cflag = 0
    rflag = 0

    for i in range(0,row):

        cflag = 0

        for j in  range(0,col):

            try :
                temp = img[inx].cpu().detach().numpy()
            except IndexError :
                temp = np.zeros(shape)

            if dim == 3:
                temp = np.transpose(temp, (1,2,0))
            elif dim == 1:
                temp = temp[0]

            if cflag == 0 :
                col_matrix = temp
                cflag = 1

            elif cflag == 1 :
                col_matrix = np.concatenate((col_matrix,temp), axis=1)

            inx+=1

        if rflag == 0 :
            row_matrix = col_matrix
            rflag = 1

        else :
            row_matrix = np.concatenate((row_matrix,col_matrix),axis=0)

    return row_matrix

def print_train_progress(epoch, item_reached, length_data):
      
        global __PRINTED__
        
        percent = (item_reached/length_data)*100
        
        #print("EPOCH : ",epoch)
        #print("\t completed ({}/{}) :\t".format(item_reached,length_data),int(percent),"%")
        
        if int(percent)%10 == 0 and __PRINTED__ != int(percent):
            __PRINTED__ = int(percent)
            print("EPOCH : ",epoch,"\t completed ({}/{}) :\t".format(item_reached,length_data),int(percent),"%")
            
         
        return
    