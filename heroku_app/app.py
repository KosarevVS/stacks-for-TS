#!/usr/local/bin/python3.7
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.pylab import plt
from sklearn.preprocessing import StandardScaler



from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.optimizers import SGD

@st.cache
def load_data(url='https://raw.githubusercontent.com/KosarevVS/stacks-for-TS/master/heroku_app/my_data.csv'):
    df_init=pd.read_csv(url)
    df_init=df_init[['CA','DF','DG']].dropna()
    my_dates=pd.date_range(start='2001-01-31',periods=len(df_init),freq='M')
    df_init.index=my_dates
    return df_init

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def prep_data(ncol):
    scl = StandardScaler()
    df_init_scal = scl.fit_transform(load_data().values)#сохраняем мат и дисп для скал тестовой выборки
    a=df_init_scal[:,ncol].reshape(230,1)
    b=np.roll(df_init_scal[:,ncol],-1).reshape(230,1)
    return np.hstack([a,b])

# def train_test(data_sep=200,n_steps=6):
#     X_cnn,y_cnn=split_sequences(prep_data(0,df_init_scal), n_steps)
#     return X_cnn[:data_sep-n_steps+2],X_cnn[data_sep-n_steps+1:],y_cnn[:data_sep-n_steps+2],y_cnn[data_sep-n_steps+1:]
# X_cnn_train1, X_cnn_test1, y_cnn_train1, y_cnn_test1=train_test(data_sep=200,n_steps=6)

def prep_data_2(stacked,n_steps=6,data_sep=200):
        X,y=split_sequences(stacked, n_steps)
        X_train=X[:data_sep-n_steps+2]
        X_test=X[data_sep-n_steps+1:]
        y_train=y[:data_sep-n_steps+2]
        y_test=y[data_sep-n_steps+1:]
        return X_train,X_test,y_train,y_test

# X_train1,X_test1,y_train1,y_test1=prep_data_2(prep_data(0))
# X_train2,X_test2,y_train2,y_test2=prep_data_2(prep_data(1))
# X_train3,X_test3,y_train3,y_test3=prep_data_2(prep_data(2))


class select_model():
    """
    Выбор модели и получение прогноза, отрисовка процесса обучения и результатов прогноза
    """
    def __init__(self,x_train,y_train,x_test,y_test,n_steps=6):
        self.n_steps  = n_steps
        self.x_train  = x_train
        self.y_train  = y_train
        self.x_test   = x_test
        self.y_test   = y_test
        # assert(len(x_train)==len(y_train)and len(x_test)==len(y_test))

    def simple_lstm(self):#,ytrain,xtest,ytest):
        ipp_1 = Input(shape=(self.n_steps, self.x_train.shape[2]),name='fact_ipp_1')
        lstm1=LSTM(6, activation='relu', input_shape=(self.n_steps, self.x_train.shape[2]))(ipp_1)
        ipp_1_pred=Dense(1,activation='linear', name='out_1')(lstm1)
        model = Model([ipp_1],[ipp_1_pred])
        optim=SGD(momentum=0.01, nesterov=True)
        model.compile(optimizer=optim,
                      loss={'out_1': 'mse'},
                    metrics=['mse', 'mae', 'mape'])
        history=model.fit({'fact_ipp_1': self.x_train},{'out_1':self.y_train},validation_data=({'fact_ipp_1': self.x_test},
              {'out_1':self.y_test}),epochs=200, batch_size=len(self.x_test), verbose=0)
        my_predicts=model.predict(self.x_test)# здесь нужно восстанавливать значения
        return my_predicts

    def simple_gru(self):
        print('x')

    def simple_cnn(self):
        print('x')

    def plot_forec_val(self):
        fig,ax = plt.subplots(figsize=(18, 5))
        ax.plot(self.y_test[:-1], color='black', label = 'Факт')
        ax.plot(self.simple_lstm().flatten(),'-.', color='blue', label = 'Прогноз')
        ax.legend(loc='best',fontsize=16)
        ax.grid()
        ax.set_title('Прогноз',fontsize=16)
        # plt.show()



def main():
    st.write("""
    # Макроэкономические прогнозы
    """)
    #
    st.sidebar.header('User Input Parameters')
    #
    # # line or bar plots
    plot_types = st.sidebar.radio("NN type",
        ['LSTM','GRU','CNN'])
    #
    # call the above function
    lstm = (plot_types=='LSTM')
    gru = (plot_types=='GRU')
    cnn = (plot_types=='CNN')
    #
    # # скрипт запускается сверхувниз, значит в каталоге файла может лежать csv, с которого скрипт будет тянуть данные
    # # деплой на heroku

    if lstm:
        x_train,x_test,y_train,y_test=prep_data_2(prep_data(0))
        my_select_model=select_model(x_train,y_train,x_test,y_test,6)
        forec_plot=my_select_model.plot_forec_val() #экземпляр класса
        #как сделать, чтоб за методом вызывался еще один метод? plot наверное тоже класс должен быть, только как он прикручивается к другому классу
        st.pyplot(fig=forec_plot, clear_figure=True)
    # if gru:
    #     mainlastm()
    # if cnn:
    #     maingru()







if __name__ == '__main__':
    main()
