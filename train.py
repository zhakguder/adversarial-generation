import tensorflow as tf
from tensorflow import keras
from ipdb import set_trace
from models import build_net, VAE

def loss(inputs, reconstruction, mean, logvar):

    BCE = tf.reduce_sum(tf.keras.metrics.binary_crossentropy(inputs, reconstruction))
    KL = tf.reduce_sum(-0.5 * (1 + logvar - mean**2 - tf.exp(logvar)))
    return BCE + KL


def grad(inputs, targets):
    relu_dims = [40,20]
    softmax_dim = 10
    latent_dim = 10
    batch_size = 64
    with tf.GradientTape() as tape:
        reconstruction = VAE(inputs, relu_dims, latent_dim, batch_size)
        loss_value = loss(inputs, reconstruction, mean, logvar)
    return tape.gradient(loss_value, model.trainable_variables)

if __name__ == '__main__':

    from data import x_train, y_train, x_test, y_test

    relu_dims = [40,20]
    softmax_dim = 10
    latent_dim = 10
    batch_size = 64


    inputs = keras.Input(shape=(784,), name = 'mnist_img')
    #mean, logvar, reconstruction = VAE(inputs, relu_dims, latent_dim, batch_size)
    reconstruction = VAE(inputs, relu_dims, latent_dim, batch_size)
    model = keras.Model(inputs=inputs, outputs=reconstruction, name='crazy_vae')
    #optimizer = tf.keras.optimizers.RMSprop()
    #model.summary()
    #keras.utils.plot_model(model, 'my_first_model.png')

    for i in range(4):
        grads = grad(inputs, inputs)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))
        print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
        print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))



    #history = model.fit(x_train, [x_train], batch_size=batch_size, epochs=5, validation_split=0.2)

    #test_scores = model.evaluate(x_test, y_test, verbose=0)

    #print('Test loss:', test_scores[0])
    #print('Test accuracy:', test_scores[1])
