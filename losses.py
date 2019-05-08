import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def VAE_loss(features, p_theta_x, q_z_x, p_z):

    # `distortion` is just the negative log likelihood.
    distortion = -p_theta_x.log_prob(features)
    avg_distortion = tf.reduce_mean(input_tensor=distortion)
    tf.compat.v1.summary.scalar("distortion", avg_distortion)

    rate = tfd.kl_divergence(q_z_x, p_z)
    avg_rate = tf.reduce_mean(input_tensor=rate)
    tf.compat.v1.summary.scalar("rate", avg_rate)

    elbo_local = -(rate + distortion)

    elbo = tf.reduce_mean(input_tensor=elbo_local)
    loss = -elbo
    tf.compat.v1.summary.scalar("elbo", elbo)
    return loss, elbo, avg_rate, avg_distortion
