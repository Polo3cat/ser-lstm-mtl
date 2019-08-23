from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed
from keras import regularizers
from keras.optimizers import Adam

from categorical_focal_loss import categorical_focal_loss


def generate_lstm_model(learning_rate: float, gender_weight: float, acted_weight: float, private_layers: int):
    input_ =            Input(shape=(None,27), name='input') 
    output =            LSTM(128, kernel_regularizer=regularizers.l2(0.01), recurrent_dropout=0.5, dropout=0.5, name='lstm-0', return_sequences=True,)(input_)
    output =            LSTM(128, kernel_regularizer=regularizers.l2(0.01), recurrent_dropout=0.5, dropout=0.5, name='lstm-1', return_sequences=True,)(output)
    
    i = 0
    if private_layers:
      gender_output =     TimeDistributed(Dense(128, activation='sigmoid'), name=f'private_gender-{i}')(output)
      acted_output =      TimeDistributed(Dense(128, activation='sigmoid'), name=f'private_acted-{i}')(output)
      emotion_output =    TimeDistributed(Dense(128, activation='sigmoid'), name=f'private_emotion-{i}')(output)
      i += 1
    while i < (private_layers):
      gender_output =     TimeDistributed(Dense(128, activation='sigmoid'), name=f'private_gender-{i}')(gender_output)
      acted_output =      TimeDistributed(Dense(128, activation='sigmoid'), name=f'private_acted-{i}')(acted_output)
      emotion_output =    TimeDistributed(Dense(128, activation='sigmoid'), name=f'private_emotion-{i}')(emotion_output)
      i += 1

    gender_output =     TimeDistributed(Dense(2, activation='softmax'), name='gender')(gender_output if private_layers else output)
    acted_output =      TimeDistributed(Dense(2, activation='softmax'), name='acted')(acted_output if private_layers else output)
    emotion_output =    TimeDistributed(Dense(4, activation='softmax'), name='emotion')(emotion_output if private_layers else output)

    model = Model(inputs=input_, outputs=[gender_output, acted_output, emotion_output])
    model.summary()

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=0.5)

    model.compile(optimizer=adam, 
                  loss=categorical_focal_loss(), 
                  metrics=['acc'], 
                  loss_weights=[gender_weight, acted_weight, 1])
    return model
