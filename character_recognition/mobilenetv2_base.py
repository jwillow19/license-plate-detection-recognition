from keras.applications import MobileNetV2
# Layers
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
# Compile
from keras.optimizers import Adam
# from keras.optimizers.schedules import ExponentialDecay


# Load pretrained MobileNetv2 and build layers on top

def mobilev2_base(lr=1e-4, output_shape=36):
    '''
    Input: learning rate, lr_decay, output_shape=36 (36 classes)
    Output: compiled model built on top of pretrained mobilenetv2
    '''
    base_model = MobileNetV2(
        # Load weights pretrained on ImageNet
        weights='imagenet',
        input_shape=None,
        input_tensor=Input(shape=(80, 80, 3)),
        # Exclude ImageNet classifier at the top
        include_top=False
    )

    # Freeze base_model
    base_model.trainable = False

    # Create new model on top
    # Instantiate variable for model input
    # inputs = Input(shape=(80, 80, 3))

    # Store base_model output to x
    x = base_model.output
    # Convert features of shape 'base_model.output_shape[1:] to vectors
    x = AveragePooling2D(pool_size=(3, 3))(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Optimizer
    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate=lr,
    #     decay_steps=10000,
    #     decay_rate=0.9)

    opt = Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07)
    # Compile model - Adam optimizer, categorical loss, evaluate on accuracy
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# print('Creating model...')
# model = mobilev2_base()
# print('Saving model...')
# # model.save('../model/mobilenetv2_base')
# print('Model: ', model.summary())
