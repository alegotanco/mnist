import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

(xtrain,ytrain),(xtest,ytest) = tf.keras.datasets.mnist.load_data()

# xtrain = xtrain.reshape(60000,784).astype('float32')/255.0
# xtest = xtest.reshape(10000, 784).astype('float32') / 255.0

# model = models.Sequential([
#     Input(shape=(784,)),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(128, activation='relu',),
#     layers.Dense(10, activation='softmax')
# ])
model = models.Sequential([
    Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.5),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=20,batch_size=128)

test_loss, test_acc = model.evaluate(xtest, ytest)
print('Test accuracy:', test_acc)







