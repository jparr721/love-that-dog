import numpy.linalg as LA
import tensorflow as tf
from numpy import dot


def squash_capsule_input(s_j):
    """
    The squash function determines if the data represented
    by a particular capsule is in the given input. If so,
    then this function will pass it to that capsule to be
    analyzed. The nonlinearity will be used by the discriminative
    learning to determine this connection during the routing algorithms
    processing.

    Parameters
    ----------
    s_j : np.array
        The input sample to be squished.

    Returns
    -------
    np.array vector_output (v_j)
        The vector output of a given capsule in the network
        of capsules.
    """
    frac_one = (LA.norm(s_j)**2 / 1 + LA.norm(s_j)**2)
    frac_two = s_j / LA.norm(s_j)

    return frac_one * frac_two


def compute_prediction_vectors(W, u_i):
    """
    The prediction vectors are what we will be squishing to determine
    usefuleness in the routing function. We take the weight matrix W
    and we multiply it by the output of capsule i (u) and use this to
    make a prediction as we see ordinarily in feed forward neural networks.

    Parameters
    ----------
    W : np.array
        The weight matrix for the network.
    u_i : np.array
        The output of a given capsule i which is being fed forward.

    Returns
    -------
    np.array prediction_vectors (u_hat_ij)
        The prediction vectors which will supply the output from a given
        capsule within in a given layer in the neural network.
    """
    return W * u_i


def compute_capsule_input(c_ij, u_ji):
    """
    This computes the input to a given capsule before being squashed. This
    is the rough output of a vector which is used in routing to determine
    if there is a relationship between two layers.

    Parameters
    ----------
    c_ij : np.array
        Our coupling coefficient which are determined during the dynamic
        routing process. These are used to weight each of the prediction
        vectors to help it assign to the proper capsule better after squashing.
    u_ij : np.array
        Our prediction vectors which we aquired from our standard feed forward
        process in the neural network.

    Returns
    -------
    np.array : s_j
        The weighted total capsule input. This is the weighed sum of each of
        the prediction vectors weighted by the coupling coefficient.
    """
    return dot(c_ij, u_ji)


def calculate_coupling_coefficients(b_ij):
    """
    The coupling coefficients are the coefficients between a capsule i and
    all the capsules in the next layer which takes the logit input b which
    is the probability that a capsule i will be coupled to a capsule j in
    the next layer. These coefficients weight the prediction vectors to allow
    routing to work more effectively.

    Parameters
    ----------
    b_ij : np.array
        The logits from capsule i --> capsule j.

    Returns
    -------
    np.array : c_ij
        Returns the coupling coefficients between layer i and layer j
    """
    return tf.nn.softmax(b_ij, axis=None)
