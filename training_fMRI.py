"""
===============================================================================
Self-Supervised Deep Unrolled SMS/SENSE fMRI Reconstruction - Training Script
===============================================================================

Author:       Burak Demirel, PhD
Institution:  University of Minnesota

Citation:
    doi: XXXXX

Description
-----------
Training code for a physics-guided, self-supervised functional MRI (fMRI)
reconstruction model.
The reconstruction combines a residual CNN regularizer with iterative SENSE
encoding/data-consistency updates. For simultaneous multi-slice (SMS) data,
multiple slice images are encoded with their corresponding coil sensitivity
maps and combined in k-space. A held-out subset of acquired k-space samples is
used to form the self-supervised training loss.

Important implementation note
-----------------------------
This version is a readability/maintenance cleanup of the original research
code. Variable names, comments, and organization have been improved while the
underlying reconstruction mathematics, tensor operations, model architecture,
training behavior, and numerical constants are intentionally preserved.
===============================================================================
"""

import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import hdf5storage

# Disable HDF5 file locking for the shared research filesystem used by this code.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# =============================================================================
# Global fMRI reconstruction and training configuration
# =============================================================================
nb_blocks = 10
epochs = 100
batchSize = 1
num_res_blocks = 15
acc_rate = 4

# Optimizer parameters
learning_rate = 3e-4
LR = 'LR_3e4_'

num_gpus = [2, 3]

rho = 0.4
num_reps = 5
data_opt = 'DL_perf'
mask_type = 'Uniform'
directory_suffix = mask_type + '_Random' + str(np.int(rho * 100)) + 'Perc_' + str(num_reps) + 'Reps'
print('data opt: ', data_opt, ', acc rate : ', acc_rate, ', mask type :', mask_type, ', num reps: ', num_reps)
transfer_learning_option = False

if data_opt == 'DL_perf':

    slice_size, nrow_GLOB, ncol_GLOB, ncoil_GLOB = 5, 128, 110, 32

    if transfer_learning_option:
        data_tag = 'SSDU_fMRI_R4_TL_'
        TL_path = '/home/daedalus2-data2/icarus/Burak_Files/perfusion_DL/savedModels/SSDU_perf_ae_pfres_goldnew4_firstR_kkkk_20PCG_R5_4R_10K_15RB_100E_LR_3e4_Uniform_Random40Perc_1Reps'
    else:
        data_tag = 'SSDU_fMRI_R4_'

    # =========================================================================
    # fMRI training data loading
    # =========================================================================
    # Each MAT file provides collapsed SMS fMRI k-space, slice-specific SENSE maps,
    # and acquisition masks. Arrays are transposed below into the layout used by
    # the reconstruction graph.

    file_dir = "/home/daedalus1-raid1/omer-data/fMRI_FOV3/training/subject_"
    maps_list = list()
    kspace_list = list()
    padded_mask_list = list()
    unpadded_mask_list = list()
    subjects = [5, 6, 7, 8]

    for subject_num in subjects:
        slices = [1, 2, 3, 4, 5, 6, 7]

        for slice_num in slices:
            print ('Subject Num: ', subject_num)
            kspace = hdf5storage.loadmat(file_dir + str(subject_num) + "_run_" + str(slice_num) + ".mat")['kspace_all_r4']
            maps = hdf5storage.loadmat(file_dir + str(subject_num) + "_run_" + str(slice_num) + ".mat")['sense_maps_all_small']

            padded_mask = hdf5storage.loadmat(file_dir + str(subject_num) + "_run_" + str(slice_num) + ".mat")['padded_all_r4']
            unpadded_mask = hdf5storage.loadmat(file_dir + str(subject_num) + "_run_" + str(slice_num) + ".mat")['not_padded_all_r4']

            kspace = np.transpose(np.copy(kspace[:, :, :, :]), axes=(3, 0, 1, 2))

            maps = np.transpose(np.copy(maps[:, :, :, :, :]), axes=(4, 3, 0, 1, 2))

            padded_mask = np.transpose(np.copy(padded_mask[:, :, :, :]), axes=(3, 0, 1, 2))
            unpadded_mask = np.transpose(np.copy(unpadded_mask[:, :, :, :]), axes=(3, 0, 1, 2))

            print('kspace: ', kspace.shape, ', maps: ', maps.shape)
            print('padded mask: ', padded_mask.shape, ', unpadded mask: ', unpadded_mask.shape)

            maps_list.append(maps)
            kspace_list.append(kspace)
            padded_mask_list.append(padded_mask[..., 0])
            unpadded_mask_list.append(unpadded_mask[..., 0])
            print('kspace: ', np.concatenate(np.asarray(kspace_list), axis=0).shape, ', maps: ',np.concatenate(np.asarray(maps_list), axis=0).shape, ',  padded mask : ',np.concatenate(np.asarray(padded_mask_list), axis=0).shape)

            kspace_train = np.concatenate(np.asarray(kspace_list), axis=0)
            trnCsm = np.concatenate(np.asarray(maps_list), axis=0)
            padded_mask = np.concatenate(np.asarray(padded_mask_list), axis=0)
            unpadded_mask = np.concatenate(np.asarray(unpadded_mask_list), axis=0)

# =============================================================================
# Complex/real representation helpers
# =============================================================================
# TensorFlow 1.x stores the CNN input as two real-valued channels: [real, imag].
c2r_tf = lambda x: tf.stack([tf.real(x), tf.imag(x)], axis=-1)
r2c_tf = lambda x: tf.complex(x[..., 0], x[..., 1])

def create_mask(input_data, padded_mask, unpadded_mask, mask_option='Uniform', rho=0.1, num_iter=0):
    """Split acquired k-space samples into training and held-out SSDU masks.

    The central 4x4 k-space region is excluded from the held-out set so that
    low-frequency calibration information remains available to reconstruction.
    """
    [nx, ny] = padded_mask.shape

    if mask_option == 'Uniform':
        mask_val = np.zeros_like(unpadded_mask)
        temp_mask = np.copy(unpadded_mask)
        mx = int(find_center_ind(input_data, axes=(1, 2)))
        my = int(find_center_ind(input_data, axes=(0, 2)))
        if num_iter == 0:
            print('center of kspace, mx: ', mx, ', my: ', my)
        temp_mask[mx - 2: mx + 2, my - 2: my + 2] = 0
        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nx * ny),
                               size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = index_flatten2nd(ind, (nx, ny))
        mask_val[ind_x, ind_y] = 1
        mask_trn = padded_mask - mask_val
        return padded_mask, mask_trn, mask_val

    if mask_option == 'Gaussian':
        count = 0
        test_pts = np.int(np.ceil(np.sum(unpadded_mask[:]) * rho))
        mask_val = np.zeros_like(unpadded_mask)
        temp_mask = np.copy(unpadded_mask)
        mx = int(find_center_ind(input_data, axes=(1, 2)))
        my = int(find_center_ind(input_data, axes=(0, 2)))
        if num_iter == 0:
            print('center of kspace, mx: ', mx, ', my: ', my)
        temp_mask[mx - 2: mx + 2, my - 2: my + 2] = 0
        while count <= test_pts:
            indx = np.int(np.round(np.random.normal(loc=mx, scale=(nx - 1) / 2)))
            indy = np.int(np.round(np.random.normal(loc=my, scale=(ny - 1) / 2)))
            if (0 <= indx < nx and 0 <= indy < ny and temp_mask[indx, indy] == 1 and mask_val[indx, indy] != 1):
                mask_val[indx, indy] = 1
                count = count + 1
        mask_trn = padded_mask - mask_val
        return padded_mask, mask_trn, mask_val

def index_flatten2nd(ind, shape):
    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))
    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

def c2r(inp):
    """Convert a NumPy complex array to a final two-channel [real, imag] array."""
    if inp.dtype == 'complex64':
        dtype = np.float32
    else:
        dtype = np.float64
    out = np.zeros(inp.shape + (2,), dtype=dtype)
    out[..., 0] = inp.real
    out[..., 1] = inp.imag
    return out

# =============================================================================
# Fourier and k-space helpers
# =============================================================================

def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    ispace = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:
        fact = 1
        for axis in axes:
            fact = fact * ispace.shape[axis]
        ispace = ispace * np.sqrt(fact)

    return ispace

def tf_fftshift(input_x):
    """2-D FFT shift for tensors using the global full-resolution dimensions."""
    half_rows = int(nrow_GLOB / 2)
    half_cols = int(ncol_GLOB / 2)

    row_first_half = tf.identity(input_x[..., 0:half_rows, :])
    row_second_half = tf.identity(input_x[..., half_rows:, :])
    row_shifted = tf.concat([row_second_half, row_first_half], 1)

    col_first_half = tf.identity(row_shifted[..., :, 0:half_cols])
    col_second_half = tf.identity(row_shifted[..., :, half_cols:])
    fully_shifted = tf.concat([col_second_half, col_first_half], 2)
    return fully_shifted

def norm(tensor, axes=(0, 1, 2), keepdims=True):
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
    if not keepdims: return tensor.squeeze()
    return tensor

def find_center_ind(kspace, axes=(1, 2, 3)):
    pow = norm(kspace, axes=axes).squeeze()
    return np.argsort(pow)[-1:]

# =============================================================================
# Residual CNN regularizer
# =============================================================================
def conv_layer(x, szW, is_training, is_relu, is_scaling):
    """Create one convolutional layer with optional ReLU and 0.1 scaling."""
    W = tf.get_variable('W', shape=szW,
                        initializer=tf.random_normal_initializer(0, 0.05))
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    if (is_relu):
        x = tf.nn.relu(x)
    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = tf.multiply(scalar, x)
    return x

def ResNet(inp, is_training, num_res_block):
    """Residual CNN regularizer used at every unrolled reconstruction block.

    The network uses 3x3 convolutions, 64 feature channels, residual blocks,
    and a final two-channel output corresponding to real and imaginary parts.
    """
    nw = {}
    szW = {}
    szW[1] = (3, 3, 2, 64)  # Convolution at the input
    szW[2] = (3, 3, 64, 64)  # convolution during the residual blocks
    szW[3] = (3, 3, 64, 2)  # convolutions at the output
    with tf.variable_scope('FirstLayer'):
        nw['c0'] = conv_layer(inp, szW[1], is_training, is_relu=False, is_scaling=False)

    for i in np.arange(1, num_res_block + 1):
        with tf.variable_scope('ResBlock' + str(i)):
            conv_layer1 = conv_layer(nw['c' + str(i - 1)], szW[2], is_training, is_relu=True, is_scaling=False)
            conv_layer2 = conv_layer(conv_layer1, szW[2], is_training, is_relu=False, is_scaling=True)
            nw['c' + str(i)] = conv_layer2 + nw['c' + str(i - 1)]

    with tf.variable_scope('LastLayer'):
        rb_output = conv_layer(nw['c' + str(i)], szW[2], is_training, is_relu=False, is_scaling=False)

    with tf.variable_scope('LastLayer2'):
        temp_output = rb_output + nw['c0']
        nw_output = conv_layer(temp_output, szW[3], is_training, is_relu=False, is_scaling=False)

    with tf.name_scope('Residual'):
        dw = tf.identity(nw_output)  # tf.identity(inp)
    return dw

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

# =============================================================================
# SENSE data-consistency operators
# =============================================================================
def getLambda():
    """Return the shared trainable data-consistency regularization parameter."""
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=.005)
    return lam

class Aclass:
    """Linear operator used by conjugate gradient in the DC update.

    The input image contains five SMS slices stacked along the row dimension.
    For each slice, the operator applies its slice-specific coil sensitivities,
    Fourier encoding, and the acquired-sample mask. The five encoded slices are
    summed to form the collapsed SMS measurement. The adjoint then maps that
    shared k-space residual back into each slice, followed by the lambda*I term.
    """

    def __init__(self, csm, mask, lam):
        with tf.name_scope('Ainit'):
            mask_shape = tf.shape(mask)
            self.nrow, self.ncol = mask_shape[0], mask_shape[1]
            print('nrow:', mask_shape[0], 'ncol:', mask_shape[1])
            self.pixels = self.nrow * self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.)
            self.lam = lam

    def myAtA(self, img):
        """Apply (E^H E + lambda I) to the five-slice stacked image."""
        with tf.name_scope('AtA'):
            slice1_coil_images = self.csm[0, :, :, :] * img[:nrow_GLOB, :]
            slice2_coil_images = self.csm[1, :, :, :] * img[nrow_GLOB:nrow_GLOB * 2, :]
            slice3_coil_images = self.csm[2, :, :, :] * img[nrow_GLOB * 2:nrow_GLOB * 3, :]
            slice4_coil_images = self.csm[3, :, :, :] * img[nrow_GLOB * 3:nrow_GLOB * 4, :]
            slice5_coil_images = self.csm[4, :, :, :] * img[nrow_GLOB * 4:nrow_GLOB * 5, :]

            slice1_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice1_coil_images))) / self.SF
            masked_slice1_kspace = slice1_kspace * self.mask
            slice2_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice2_coil_images))) / self.SF
            masked_slice2_kspace = slice2_kspace * self.mask
            slice3_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice3_coil_images))) / self.SF
            masked_slice3_kspace = slice3_kspace * self.mask
            slice4_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice4_coil_images))) / self.SF
            masked_slice4_kspace = slice4_kspace * self.mask
            slice5_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice5_coil_images))) / self.SF
            masked_slice5_kspace = slice5_kspace * self.mask

            collapsed_kspace = (masked_slice1_kspace + masked_slice2_kspace +
                                masked_slice3_kspace + masked_slice4_kspace +
                                masked_slice5_kspace)

            slice1_coil_images_adj = tf_fftshift(tf.ifft2d(tf_fftshift(collapsed_kspace))) * self.SF
            slice1_combined = tf.reduce_sum(slice1_coil_images_adj * tf.conj(self.csm[0, :, :, :]), axis=0)
            slice1_combined = slice1_combined + self.lam * img[:nrow_GLOB, :]

            slice2_coil_images_adj = tf_fftshift(tf.ifft2d(tf_fftshift(collapsed_kspace))) * self.SF
            slice2_combined = tf.reduce_sum(slice2_coil_images_adj * tf.conj(self.csm[1, :, :, :]), axis=0)
            slice2_combined = slice2_combined + self.lam * img[nrow_GLOB:nrow_GLOB * 2, :]

            slice3_coil_images_adj = tf_fftshift(tf.ifft2d(tf_fftshift(collapsed_kspace))) * self.SF
            slice3_combined = tf.reduce_sum(slice3_coil_images_adj * tf.conj(self.csm[2, :, :, :]), axis=0)
            slice3_combined = slice3_combined + self.lam * img[nrow_GLOB * 2:nrow_GLOB * 3, :]

            slice4_coil_images_adj = tf_fftshift(tf.ifft2d(tf_fftshift(collapsed_kspace))) * self.SF
            slice4_combined = tf.reduce_sum(slice4_coil_images_adj * tf.conj(self.csm[3, :, :, :]), axis=0)
            slice4_combined = slice4_combined + self.lam * img[nrow_GLOB * 3:nrow_GLOB * 4, :]

            slice5_coil_images_adj = tf_fftshift(tf.ifft2d(tf_fftshift(collapsed_kspace))) * self.SF
            slice5_combined = tf.reduce_sum(slice5_coil_images_adj * tf.conj(self.csm[4, :, :, :]), axis=0)
            slice5_combined = slice5_combined + self.lam * img[nrow_GLOB * 4:nrow_GLOB * 5, :]

            combined_image = tf.concat(
                [slice1_combined, slice2_combined, slice3_combined,
                 slice4_combined, slice5_combined],
                axis=0
            )
        return combined_image

def myCG(A, rhs, wim, lam2):
    """Run 10 iterations of complex-valued conjugate gradient in TensorFlow.

    `wim` is used as the warm start. `lam2` is retained in the signature for
    compatibility with the original implementation; lambda is stored in `A`.
    """
    rhs = r2c_tf(rhs)
    wim = r2c_tf(wim)
    cond = lambda i, *_: tf.less(i, 10)

    def body(i, rTr, x, r, p):
        with tf.name_scope('cgBody'):
            Ap = A.myAtA(p)
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p) * Ap))
            alpha = tf.complex(alpha, 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float(tf.reduce_sum(tf.conj(r) * r))
            beta = rTrNew / rTr
            beta = tf.complex(beta, 0.)
            p = r + beta * p
        return i + 1, rTrNew, x, r, p

    x = wim
    i, r, p = 0, rhs - A.myAtA(x) * 1, rhs - A.myAtA(x) * 1
    rTr = tf.to_float(tf.reduce_sum(tf.conj(r) * r), )
    loopVar = i, rTr, x, r, p
    out = tf.while_loop(cond, body, loopVar, name='CGwhile', parallel_iterations=1)[2]
    cg_out = out
    return c2r_tf(cg_out)

class Uclass:
    """Forward SMS/SENSE encoding operator used for the SSDU held-out loss."""

    def __init__(self, csm, mask):
        with tf.name_scope('Uinit'):
            mask_shape = tf.shape(mask)
            self.nrow, self.ncol = mask_shape[0], mask_shape[1]
            print('shape of mask', tf.shape(mask))

            self.pixels = self.nrow * self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.)

    def myUlAtA(self, img):
        """Encode five stacked slices into the collapsed held-out SMS k-space."""
        with tf.name_scope('ULAtA'):
            slice1_coil_images = self.csm[0, :, :, :] * img[:nrow_GLOB, :]
            slice2_coil_images = self.csm[1, :, :, :] * img[nrow_GLOB:nrow_GLOB * 2, :]
            slice3_coil_images = self.csm[2, :, :, :] * img[nrow_GLOB * 2:nrow_GLOB * 3, :]
            slice4_coil_images = self.csm[3, :, :, :] * img[nrow_GLOB * 3:nrow_GLOB * 4, :]
            slice5_coil_images = self.csm[4, :, :, :] * img[nrow_GLOB * 4:nrow_GLOB * 5, :]

            slice1_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice1_coil_images))) / self.SF
            masked_slice1_kspace = slice1_kspace * self.mask
            slice2_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice2_coil_images))) / self.SF
            masked_slice2_kspace = slice2_kspace * self.mask
            slice3_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice3_coil_images))) / self.SF
            masked_slice3_kspace = slice3_kspace * self.mask
            slice4_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice4_coil_images))) / self.SF
            masked_slice4_kspace = slice4_kspace * self.mask
            slice5_kspace = tf_fftshift(tf.fft2d(tf_fftshift(slice5_coil_images))) / self.SF
            masked_slice5_kspace = slice5_kspace * self.mask

            collapsed_kspace = (masked_slice1_kspace + masked_slice2_kspace +
                                masked_slice3_kspace + masked_slice4_kspace +
                                masked_slice5_kspace)
        return collapsed_kspace

def dc_unsupervised(rhs, csm, mask):
    """Forward-encode reconstructed images at held-out k-space locations."""
    rhs = r2c_tf(rhs)

    def fn(tmp):
        c, m, r = tmp
        Aobj = Uclass(c, m)
        y = Aobj.myUlAtA(r)
        return y

    inp = (csm, mask, rhs)
    rec = tf.map_fn(fn, inp, dtype=tf.complex64, name='valmapFn')
    return c2r_tf(rec)

def dc(rhs, csm, mask, lam1, warm_image):
    """Apply the batched CG data-consistency solve for one unrolled block."""
    lam2 = tf.complex(lam1, 0.)

    def fn(tmp):
        c, m, r, wim = tmp
        Aobj = Aclass(c, m, lam2)
        y = myCG(Aobj, r, wim, lam2)
        return y

    inp = (csm, mask, rhs, warm_image)
    rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
    return rec

# =============================================================================
# SMS/CAIPI slice-shift helpers
# =============================================================================
def shifterb(input_data, shift_amo):
    """Undo the slice-dependent circular shifts applied before the CNN.

    The five slice images are stacked along axis 1. Slices 1 and 4 are left
    unchanged. The +2 offsets are retained exactly from the original inverse
    shift implementation.
    """
    slice1 = tf.identity(input_data[..., 0:nrow_GLOB, :, :])
    slice4 = tf.identity(input_data[..., nrow_GLOB * 3:4 * nrow_GLOB, :, :])

    slice2_inverse_shift = 2 * shift_amo + 2
    slice3_inverse_shift = shift_amo + 2
    slice5_inverse_shift = 2 * shift_amo + 2

    slice2_head = tf.identity(
        input_data[..., nrow_GLOB:2 * nrow_GLOB, :slice2_inverse_shift, :]
    )
    slice2_tail = tf.identity(
        input_data[..., nrow_GLOB:2 * nrow_GLOB, slice2_inverse_shift:, :]
    )
    slice3_head = tf.identity(
        input_data[..., nrow_GLOB * 2:3 * nrow_GLOB, :slice3_inverse_shift, :]
    )
    slice3_tail = tf.identity(
        input_data[..., nrow_GLOB * 2:3 * nrow_GLOB, slice3_inverse_shift:, :]
    )
    slice5_head = tf.identity(
        input_data[..., nrow_GLOB * 4:5 * nrow_GLOB, :slice5_inverse_shift, :]
    )
    slice5_tail = tf.identity(
        input_data[..., nrow_GLOB * 4:5 * nrow_GLOB, slice5_inverse_shift:, :]
    )

    slice2 = tf.concat([slice2_tail, slice2_head], axis=2)
    slice3 = tf.concat([slice3_tail, slice3_head], axis=2)
    slice5 = tf.concat([slice5_tail, slice5_head], axis=2)

    return tf.concat([slice1, slice2, slice3, slice4, slice5], axis=1)

def shifterf(input_data, shift_amo):
    """Apply slice-dependent circular CAIPI shifts before the CNN regularizer."""
    slice1 = tf.identity(input_data[..., 0:nrow_GLOB, :, :])
    slice4 = tf.identity(input_data[..., nrow_GLOB * 3:4 * nrow_GLOB, :, :])

    slice2_forward_shift = shift_amo
    slice3_forward_shift = 2 * shift_amo
    slice5_forward_shift = shift_amo

    slice2_head = tf.identity(
        input_data[..., nrow_GLOB:2 * nrow_GLOB, :slice2_forward_shift, :]
    )
    slice2_tail = tf.identity(
        input_data[..., nrow_GLOB:2 * nrow_GLOB, slice2_forward_shift:, :]
    )
    slice3_head = tf.identity(
        input_data[..., nrow_GLOB * 2:3 * nrow_GLOB, :slice3_forward_shift, :]
    )
    slice3_tail = tf.identity(
        input_data[..., nrow_GLOB * 2:3 * nrow_GLOB, slice3_forward_shift:, :]
    )
    slice5_head = tf.identity(
        input_data[..., nrow_GLOB * 4:5 * nrow_GLOB, :slice5_forward_shift, :]
    )
    slice5_tail = tf.identity(
        input_data[..., nrow_GLOB * 4:5 * nrow_GLOB, slice5_forward_shift:, :]
    )

    slice2 = tf.concat([slice2_tail, slice2_head], axis=2)
    slice3 = tf.concat([slice3_tail, slice3_head], axis=2)
    slice5 = tf.concat([slice5_tail, slice5_head], axis=2)

    return tf.concat([slice1, slice2, slice3, slice4, slice5], axis=1)

# =============================================================================
# Unrolled reconstruction network
# =============================================================================
class UnrolledNet():
    """Alternating residual-CNN regularization and physics data consistency."""

    def __init__(self, input_x, sens_maps, mask, mask_val, nb_blocks, num_res_blocks, wrm, training):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.mask = mask
        self.mask_val = mask_val
        self.nb_blocks = nb_blocks
        self.num_res_blocks = num_res_blocks
        self.wrm = wrm
        self.training = training
        self.model = self.Unrolled()

    def Unrolled(self):
        """Build the unrolled TensorFlow graph and the held-out k-space output."""
        x, denoiser_output, dc_output = self.input_x, self.input_x, self.wrm

        all_intermediate_results = [[0 for _ in range(2)] for _ in range(self.nb_blocks)]
        lam_init = tf.constant(0.05, dtype=tf.float32)

        x0 = dc(self.input_x, self.sens_maps, self.mask, lam_init, self.wrm * 0)
        x = x0
        dc_output = x0

        with tf.name_scope('myModel'):
            with tf.variable_scope('Wts', reuse=tf.AUTO_REUSE):
                for i in range(self.nb_blocks):
                    # Move the five slices to the CAIPI-aligned representation used by
                    # the CNN. The original shift amount (36) is intentionally unchanged.
                    x = shifterf(x, 36)
                    # Shared-weight residual CNN regularization.
                    x = ResNet(x, self.training, self.num_res_blocks)  # denoising#unet.Unet(x) #denoising
                    denoiser_output = x
                    # Return from the CNN/CAIPI representation to the original slice
                    # geometry before the SENSE data-consistency solve.
                    x = shifterb(x, 36)

                    lam1 = getLambda()

                    # Right-hand side of the regularized DC system:
                    #     E^H y + lambda * z_k
                    rhs = self.input_x + ((lam1 * 1) * x)

                    x = dc(rhs, self.sens_maps, self.mask, lam1, dc_output)

                    dc_output = x

                    all_intermediate_results[i][0] = r2c_tf(tf.squeeze(denoiser_output))
                    all_intermediate_results[i][1] = r2c_tf(tf.squeeze(dc_output))

            # Re-encode the final image only at the held-out SSDU samples. This
            # prediction is compared with measured held-out k-space in the loss.
            ul_output = dc_unsupervised(x, self.sens_maps, self.mask_val)
        return x, ul_output, x0, all_intermediate_results, lam1

# =============================================================================
# NumPy preprocessing: SENSE adjoint and SSDU example construction
# =============================================================================
def sense1(input_kspace, sens_maps):
    """Apply the NumPy SENSE adjoint E^H to one slice and its coil maps."""
    image_space = ifft(input_kspace, axes=(0, 1), norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    Eh_op = np.sum(Eh_op, axis=2)

    return Eh_op

# =============================================================================
# Optional transfer-learning weight loading
# =============================================================================
if transfer_learning_option:
    print('Getting weights from trained model:')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    loadChkPoint_tl = TL_path + '/model-96'
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(TL_path + '/modelTst.meta')
        new_saver.restore(sess, loadChkPoint_tl)
        trainable_collection_trained = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        nontrainable_variables_trained = [sess.run(v) for v in trainable_collection_trained]
        print(len(nontrainable_variables_trained))
        print('\n\n\n')
    print('Trained model is loaded')

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
print('*************************************************')
start_time = time.time()
saveDir = 'savedModels/'
directory = saveDir + \
            data_tag + str(acc_rate) + 'R_' + str(nb_blocks) + 'K_' + str(num_res_blocks) + 'RB_' + str(
    epochs) + 'E_' + LR \
            + directory_suffix  # 'Uniform_' + str(num_reps) + 'Reps_UniformRandom'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName = directory + '/model'

tf.reset_default_graph()
csmT = tf.placeholder(tf.complex64, shape=(None, slice_size, ncoil_GLOB, nrow_GLOB, ncol_GLOB), name='csm')
maskT = tf.placeholder(tf.complex64, shape=(None, nrow_GLOB, ncol_GLOB), name='mask')
maskV = tf.placeholder(tf.complex64, shape=(None, nrow_GLOB, ncol_GLOB), name='testmaskV')
atbT = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='atb')
WarmT = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='warmt')
nw_out, ul_output, x0, all_intermediate_outputs, lam = UnrolledNet(atbT, csmT, maskT, maskV, nb_blocks, num_res_blocks,
                                                                   WarmT, False).model

out = tf.identity(nw_out, name='out')
ul_output = tf.identity(ul_output, name='predTst')
all_intermediate_outputs = tf.identity(all_intermediate_outputs, name='all_intermediate_outputs')
x0 = tf.identity(x0, name='x0')
lam = tf.identity(lam, name='lam')
sessFileNameTst = directory + '/modelTst'

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile = saver.save(sess, sessFileNameTst, latest_filename='checkpointTst')
print('testing model saved:' + savedFile)

print('Loading training data...')

print('size of the training data', np.shape(kspace_train))

print('Normalize the kspace to 0-1 region')
for ii in range(np.shape(kspace_train)[0]):
    temp = np.copy(kspace_train[ii, ...])
    kspace_train[ii, ...] = temp / np.max(np.abs(temp[:]))

    for kk in range(slice_size):
       temp = np.copy(trnCsm[ii, kk, ...])
       trnCsm[ii, kk, ...] = temp #/ np.max(np.abs(temp[:]))
    if np.max(np.abs(temp[:])) == 0:
        print('Max is zero at Iter ', ii)
        raise ValueError('Max is zero')

print('size of the training data', kspace_train.shape, ', coil maps: ', trnCsm.shape)

nSlice, nrow, ncol, ncoil = kspace_train.shape

trnMask = np.empty((nSlice, num_reps, nrow, ncol), dtype=np.complex64)
valMask = np.empty((nSlice, num_reps, nrow, ncol), dtype=np.complex64)

trnAtb = np.empty((nSlice, num_reps, nrow * slice_size, ncol), dtype=np.complex64)
ref_kspace = np.empty((nSlice, num_reps, nrow, ncol, ncoil), dtype=np.complex64)
trnCsmAll = np.empty((nSlice, num_reps, slice_size, nrow, ncol, ncoil), dtype=np.complex64)
trnWarmAll = np.empty((nSlice, num_reps, nrow * slice_size, ncol), dtype=np.complex64)

# =============================================================================
# Build repeated fMRI SSDU mask partitions and SENSE adjoints
# =============================================================================
print('Multi Mask Version -- getting the refs and aliased sense1 images')
for ii in range(nSlice):
    if np.mod(ii, 15) == 0:
        print('Iteration: ', ii)
    for jj in range(num_reps):
        _, trnMask[ii, jj, ...], valMask[ii, jj, ...] = create_mask(kspace_train[ii],
                                                                                        padded_mask[ii, ...],
                                                                                        unpadded_mask[ii, ...],
                                                                                        mask_option=mask_type, rho=rho,
                                                                                        num_iter=ii)
        proc_mask = np.copy(trnMask[ii, jj, ...])
        proc_maskV = np.copy(valMask[ii, jj, ...])
        proc_mask = np.tile(proc_mask[:, :, np.newaxis], (1, 1, ncoil))
        proc_maskV = np.tile(proc_maskV[:, :, np.newaxis], (1, 1, ncoil))
        sub_kspace = kspace_train[ii] * proc_mask
        ref_kspace[ii, jj, ...] = kspace_train[ii] * proc_maskV
        for kk in range(slice_size):
            idx_start, idx_end = kk * nrow_GLOB, (kk + 1) * nrow_GLOB
            trnAtb[ii, jj, idx_start:idx_end, ...] = sense1(sub_kspace, trnCsm[ii, kk, ...])

        trnCsmAll[ii, jj, ...] = np.copy(trnCsm[ii, ...])
        trnWarmAll[ii, jj, ...] = np.copy(trnAtb[ii, jj,...])

trnWarm = np.concatenate(list(trnWarmAll), axis=0)
trnCsm = np.concatenate(list(trnCsmAll), axis=0)
ref_kspace = np.concatenate(list(ref_kspace), axis=0)
trnMask = np.concatenate(list(trnMask), axis=0)
valMask = np.concatenate(list(valMask), axis=0)
trnAtb = np.concatenate(list(trnAtb), axis=0)

sio.savemat(('unnormalized_Sense1_images.mat'), {'input': trnAtb})
sio.savemat(('warm_ins.mat'), {'warm': trnWarm})
sio.savemat(('valmasks.mat'), {'trnMask': trnMask, 'valMask': valMask})
print('size of ref kspace: ', np.shape(ref_kspace), ', coil maps: ', trnCsm.shape, ' size of input: ', np.shape(trnAtb), \
      'size of trn mask: ', trnMask.shape, ', size of val mask: ', valMask.shape)
trnAtb = c2r(trnAtb)
trnWarm = c2r(trnWarm)
trnCsm = np.transpose(trnCsm, (0, 1, 4, 2, 3))
ref_kspace = np.transpose(ref_kspace, (0, 3, 1, 2))
ref_kspace = c2r(ref_kspace)
print(
    'size of reference kspace after c2r: ', np.shape(ref_kspace), ', size of maps: ', trnCsm.shape, ', size of input: ',
    trnAtb.shape)
# =============================================================================
# TensorFlow input pipeline and multi-GPU training graph
# =============================================================================
nTrn = trnAtb.shape[0]
total_batch = int(np.floor(np.float32(nTrn) / (batchSize * len(num_gpus))))
assert not np.any(np.isnan(trnAtb))
tf.reset_default_graph()
with tf.device('/cpu:0'):
    tower_grads = []
    kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
    csmP = tf.placeholder(tf.complex64, shape=(None, slice_size, ncoil_GLOB, nrow_GLOB, ncol_GLOB), name='csm')
    maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='mask')
    maskVal = tf.placeholder(tf.complex64, shape=(None, None, None), name='maskVal')
    atbP = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='atb')
    WarmP = tf.placeholder(tf.float32, shape=(None, slice_size * nrow_GLOB, ncol_GLOB, 2), name='Warm')
    dataset = tf.data.Dataset.from_tensor_slices((kspaceP, atbP, csmP, maskP, maskVal, WarmP))
    dataset = dataset.shuffle(buffer_size=10 * len(num_gpus))
    dataset = dataset.batch(batchSize)
    dataset = dataset.prefetch(len(num_gpus))
    iterator = dataset.make_initializable_iterator()
    for i in range(len(num_gpus)):
        with tf.device(assign_to_device('/gpu:{}'.format(num_gpus[i]), ps_device='/cpu:0')):
            refT, atbT, csmT, maskT, maskV, WarmT = iterator.get_next('getNext')
            out, ul_output, _, _, _ = UnrolledNet(atbT, csmT, maskT, maskV, nb_blocks,
                                                                         num_res_blocks, WarmT, True).model
            out = tf.identity(out, name='nw_out')
            # SSDU objective: equal-weight normalized L2 and L1 errors between
            # measured held-out k-space and the forward-encoded reconstruction.
            scalar = tf.constant(0.5, dtype=tf.float32)
            loss = tf.multiply(scalar, tf.norm(refT - ul_output) / tf.norm(refT)) + tf.multiply(scalar, tf.norm(
                refT - ul_output, ord=1) / tf.norm(refT,
                                                   ord=1))
            all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss)
            tower_grads.append(grads)
    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)
    print ('parameters are: Epochs:', epochs, ' BS:', batchSize, 'nblocks:', nb_blocks)
    saver = tf.train.Saver(max_to_keep=100)
    totalLoss, totalTime, ep = [], [], 0
    avg_cost = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        feedDict = {kspaceP: ref_kspace, atbP: trnAtb, maskP: trnMask, maskVal: valMask,
                    csmP: trnCsm, WarmP: trnWarm}
        if transfer_learning_option:
            print('Assigning weights to new model:')
            trainable_collection_test = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
            nontrainable_variables_test = [v for v in trainable_collection_test]
            print(len(nontrainable_variables_test))
            print('\n\n\n')
            for ii in range(len(nontrainable_variables_test)):
                sess.run(nontrainable_variables_test[ii].assign(nontrainable_variables_trained[ii]))
        # Main optimization loop
        print('Training...')
        for ii in range(epochs):
            sess.run(iterator.initializer, feed_dict=feedDict)
            if ii == 0:
                savedFile = saver.save(sess, sessFileName)
                print("Model meta graph saved in::%s" % savedFile)
                print('Number of Parameters')
                print(sess.run(all_trainable_vars))
            ep = ep + 1
            avg_cost = 0
            tic = time.time()
            try:
                for jj in range(total_batch):
                    tmp, _, _ = sess.run([loss, update_ops, train_op])
                    avg_cost += tmp / total_batch
                    if (ii == 0 and jj == 0):
                        print('Iter: ', ii, 'Loss : ', tmp)

                toc = time.time() - tic
                totalLoss.append(avg_cost)
                totalTime.append(toc)
                print("Epoch:", ii, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))

            except tf.errors.OutOfRangeError:
                pass
            saver.save(sess, sessFileName, global_step=ep)
            sio.savemat((directory + '/TrainingLog.mat'), {'loss': totalLoss})
        saver.save(sess, sessFileName, global_step=ep)
    end_time = time.time()
    sio.savemat((directory + '/TrainingLog.mat'), {'loss': totalLoss})
    print ('Training completed in minutes ', ((end_time - start_time) / 60))
    print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
    print ('*************************************************')
