from torch.utils.data import Dataset
import pandas as pd



class FraudDataset(Dataset):
    #Constructor for initially loading
    def __init__(self,train,stacked_matrix_df_train,stacked_matrix_df_test,y_train,y_test):

        if train:
            self.X=stacked_matrix_df_train.to_numpy()
            print("Shape of Dataset.X:"+str(self.X.shape))
            self.y=pd.DataFrame(y_train).to_numpy()
            print("Shape of Dataset.y"+str(self.y.shape))
        else:
            self.X=stacked_matrix_df_test.to_numpy()
            print("Shape of Dataset.X:"+str(self.X.shape))
            self.y=pd.DataFrame(y_test).to_numpy()
            print("Shape of Dataset.y:"+str(self.y.shape))

        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        print("After reshape: Dataset.y.shape= "+str(self.y.shape))

     # Get the number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # Get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]

