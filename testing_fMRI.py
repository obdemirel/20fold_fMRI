"""
===============================================================================
Self-Supervised Deep Unrolled SMS/SENSE fMRI Reconstruction - Testing Script
===============================================================================

Author:       Burak Demirel, PhD
Institution:  University of Minnesota

Citation:
    doi: XXXXX

Description
-----------
Inference/testing code for the trained physics-guided fMRI reconstruction
model. The script loads fully acquired undersampled R=4 SMS fMRI k-space,
constructs the five-slice SENSE-adjoint input using the acquired-sample mask,
restores a trained TensorFlow checkpoint, and reconstructs each fMRI frame.

The model architecture intentionally matches the corresponding training script:
    - SMS factor: 5
    - Image matrix: 128 x 110
    - Receiver coils: 32
    - Unrolled blocks: 10
    - Residual blocks per CNN: 15
    - CAIPI shift parameter: 36
    - Data consistency: 10-iteration CG-SENSE

Testing uses the complete acquired R=4 mask. No SSDU training/validation split is
created during inference.
===============================================================================
"""

import os
import numpy as np
import tensorflow as tf
import hdf5storage

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# =============================================================================
# User configuration
# =============================================================================

# Directory produced by training_fMRI_R4_clean.py.
MODEL_DIR = (
    "savedModels/"
    "SSDU_fMRI_R4_4R_10K_15RB_100E_LR_3e4_Uniform_Random40Perc_5Reps"
)

# Set to None to restore the latest training checkpoint in MODEL_DIR.
# To use a specific epoch instead, provide the complete checkpoint prefix,
# for example: "/path/to/model-100".
CHECKPOINT_PATH = None

# Add one or more fMRI MAT files here. Each file is expected to contain:
#   kspace_all_r4
#   sense_maps_all_small
#   padded_all_r4
TEST_FILES = [
    # "/home/.../subject_9_run_1.mat",
]

OUTPUT_DIR = "testResults_fMRI_R4"
TEST_BATCH_SIZE = 1
TEST_GPU = 2

# Save the initial CG-SENSE reconstruction together with the final network output.
SAVE_INITIAL_CG = True

# =============================================================================
# Reconstruction constants - must match training
# =============================================================================

NB_BLOCKS = 10
NUM_RES_BLOCKS = 15
SLICE_SIZE = 5
NROW = 128
NCOL = 110
NCOIL = 32
CAIPI_SHIFT = 36
CG_ITERATIONS = 10


# =============================================================================
# Complex/real representation helpers
# =============================================================================

c2r_tf = lambda x: tf.stack([tf.real(x), tf.imag(x)], axis=-1)
r2c_tf = lambda x: tf.complex(x[..., 0], x[..., 1])


def c2r(inp):
    """Convert a NumPy complex array to a two-channel [real, imag] array."""
    dtype = np.float32 if inp.dtype == np.complex64 else np.float64
    out = np.zeros(inp.shape + (2,), dtype=dtype)
    out[..., 0] = inp.real
    out[..., 1] = inp.imag
    return out


def r2c(inp):
    """Convert a final two-channel NumPy array back to a complex array."""
    return inp[..., 0] + 1j * inp[..., 1]


# =============================================================================
# Fourier helpers
# =============================================================================


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """Centered unitary inverse FFT used by the NumPy SENSE adjoint."""
    image_space = np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(kspace, axes=axes),
            axes=axes,
            norm=norm,
        ),
        axes=axes,
    )

    if unitary_opt:
        scale = 1
        for axis in axes:
            scale *= image_space.shape[axis]
        image_space *= np.sqrt(scale)

    return image_space


def tf_fftshift(input_x):
    """Centered 2-D FFT shift for tensors with the training matrix dimensions."""
    half_rows = NROW // 2
    half_cols = NCOL // 2

    row_shifted = tf.concat(
        [input_x[..., half_rows:, :], input_x[..., :half_rows, :]],
        axis=1,
    )
    fully_shifted = tf.concat(
        [row_shifted[..., :, half_cols:], row_shifted[..., :, :half_cols]],
        axis=2,
    )
    return fully_shifted


# =============================================================================
# Residual CNN regularizer - identical trainable-variable structure to training
# =============================================================================


def conv_layer(x, filter_shape, is_relu, is_scaling):
    weights = tf.get_variable(
        "W",
        shape=filter_shape,
        initializer=tf.random_normal_initializer(0, 0.05),
    )
    x = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding="SAME")

    if is_relu:
        x = tf.nn.relu(x)
    if is_scaling:
        x = 0.1 * x

    return x


def ResNet(inp, num_res_blocks):
    """Shared residual CNN used in each unrolled reconstruction block."""
    features = {}

    input_filter = (3, 3, 2, 64)
    residual_filter = (3, 3, 64, 64)
    output_filter = (3, 3, 64, 2)

    with tf.variable_scope("FirstLayer"):
        features["c0"] = conv_layer(
            inp,
            input_filter,
            is_relu=False,
            is_scaling=False,
        )

    for block_index in range(1, num_res_blocks + 1):
        with tf.variable_scope("ResBlock" + str(block_index)):
            residual = conv_layer(
                features["c" + str(block_index - 1)],
                residual_filter,
                is_relu=True,
                is_scaling=False,
            )
            residual = conv_layer(
                residual,
                residual_filter,
                is_relu=False,
                is_scaling=True,
            )
            features["c" + str(block_index)] = (
                residual + features["c" + str(block_index - 1)]
            )

    with tf.variable_scope("LastLayer"):
        residual_output = conv_layer(
            features["c" + str(num_res_blocks)],
            residual_filter,
            is_relu=False,
            is_scaling=False,
        )

    with tf.variable_scope("LastLayer2"):
        residual_output = residual_output + features["c0"]
        network_output = conv_layer(
            residual_output,
            output_filter,
            is_relu=False,
            is_scaling=False,
        )

    return network_output


# =============================================================================
# SMS/SENSE data-consistency operator
# =============================================================================


def getLambda():
    """Return the shared trainable DC regularization parameter."""
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        return tf.get_variable(name="lam1", dtype=tf.float32, initializer=0.005)


class Aclass:
    """Apply (E^H E + lambda I) for the five-slice SMS SENSE model."""

    def __init__(self, coil_sensitivities, sampling_mask, lam):
        with tf.name_scope("Ainit"):
            mask_shape = tf.shape(sampling_mask)
            self.pixels = mask_shape[0] * mask_shape[1]
            self.mask = sampling_mask
            self.csm = coil_sensitivities
            self.lam = lam
            self.scale = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.0)

    def myAtA(self, stacked_image):
        """Apply the normal SMS encoding operator and lambda regularization."""
        with tf.name_scope("AtA"):
            encoded_slices = []

            for slice_index in range(SLICE_SIZE):
                row_start = slice_index * NROW
                row_end = (slice_index + 1) * NROW

                slice_image = stacked_image[row_start:row_end, :]
                coil_images = self.csm[slice_index, :, :, :] * slice_image
                slice_kspace = (
                    tf_fftshift(tf.fft2d(tf_fftshift(coil_images))) / self.scale
                )
                encoded_slices.append(slice_kspace * self.mask)

            # SMS acquisition: all simultaneously excited slices contribute to
            # the same collapsed measured k-space.
            collapsed_kspace = tf.add_n(encoded_slices)
            collapsed_coil_images = (
                tf_fftshift(tf.ifft2d(tf_fftshift(collapsed_kspace))) * self.scale
            )

            adjoint_slices = []
            for slice_index in range(SLICE_SIZE):
                row_start = slice_index * NROW
                row_end = (slice_index + 1) * NROW

                slice_adjoint = tf.reduce_sum(
                    collapsed_coil_images * tf.conj(self.csm[slice_index, :, :, :]),
                    axis=0,
                )
                slice_adjoint += self.lam * stacked_image[row_start:row_end, :]
                adjoint_slices.append(slice_adjoint)

            return tf.concat(adjoint_slices, axis=0)


def myCG(operator, rhs, warm_image):
    """Run the same 10-iteration complex conjugate-gradient solve as training."""
    rhs = r2c_tf(rhs)
    warm_image = r2c_tf(warm_image)

    def condition(iteration, *_):
        return tf.less(iteration, CG_ITERATIONS)

    def body(iteration, rTr, x, residual, direction):
        with tf.name_scope("cgBody"):
            Adirection = operator.myAtA(direction)
            alpha = rTr / tf.to_float(
                tf.reduce_sum(tf.conj(direction) * Adirection)
            )
            alpha = tf.complex(alpha, 0.0)

            x = x + alpha * direction
            residual = residual - alpha * Adirection

            new_rTr = tf.to_float(
                tf.reduce_sum(tf.conj(residual) * residual)
            )
            beta = tf.complex(new_rTr / rTr, 0.0)
            direction = residual + beta * direction

        return iteration + 1, new_rTr, x, residual, direction

    x = warm_image
    residual = rhs - operator.myAtA(x)
    direction = residual
    rTr = tf.to_float(tf.reduce_sum(tf.conj(residual) * residual))

    loop_variables = (0, rTr, x, residual, direction)
    reconstructed = tf.while_loop(
        condition,
        body,
        loop_variables,
        name="CGwhile",
        parallel_iterations=1,
    )[2]

    return c2r_tf(reconstructed)


def dc(rhs, coil_sensitivities, sampling_mask, lam, warm_image):
    """Apply the CG data-consistency solve independently across the batch."""
    complex_lam = tf.complex(lam, 0.0)

    def reconstruct_one(inputs):
        csm, mask, rhs_one, warm_one = inputs
        operator = Aclass(csm, mask, complex_lam)
        return myCG(operator, rhs_one, warm_one)

    return tf.map_fn(
        reconstruct_one,
        (coil_sensitivities, sampling_mask, rhs, warm_image),
        dtype=tf.float32,
        name="mapFn",
    )


# =============================================================================
# SMS/CAIPI slice shifts - same geometry used during training
# =============================================================================


def shifterf(input_data, shift_amount):
    """Apply the slice-dependent circular shifts before the CNN."""
    slice1 = input_data[..., 0:NROW, :, :]
    slice2 = input_data[..., NROW:2 * NROW, :, :]
    slice3 = input_data[..., 2 * NROW:3 * NROW, :, :]
    slice4 = input_data[..., 3 * NROW:4 * NROW, :, :]
    slice5 = input_data[..., 4 * NROW:5 * NROW, :, :]

    def circular_left_shift(slice_image, amount):
        return tf.concat(
            [slice_image[..., amount:, :], slice_image[..., :amount, :]],
            axis=2,
        )

    slice2 = circular_left_shift(slice2, shift_amount)
    slice3 = circular_left_shift(slice3, 2 * shift_amount)
    slice5 = circular_left_shift(slice5, shift_amount)

    return tf.concat([slice1, slice2, slice3, slice4, slice5], axis=1)


def shifterb(input_data, shift_amount):
    """Undo the training CAIPI shifts before the SENSE DC update.

    The +2 offsets are retained exactly from the original implementation.
    """
    slice1 = input_data[..., 0:NROW, :, :]
    slice2 = input_data[..., NROW:2 * NROW, :, :]
    slice3 = input_data[..., 2 * NROW:3 * NROW, :, :]
    slice4 = input_data[..., 3 * NROW:4 * NROW, :, :]
    slice5 = input_data[..., 4 * NROW:5 * NROW, :, :]

    def circular_left_shift(slice_image, amount):
        return tf.concat(
            [slice_image[..., amount:, :], slice_image[..., :amount, :]],
            axis=2,
        )

    slice2 = circular_left_shift(slice2, 2 * shift_amount + 2)
    slice3 = circular_left_shift(slice3, shift_amount + 2)
    slice5 = circular_left_shift(slice5, 2 * shift_amount + 2)

    return tf.concat([slice1, slice2, slice3, slice4, slice5], axis=1)


# =============================================================================
# Inference network
# =============================================================================


class UnrolledInference:
    """Inference-only version of the trained unrolled reconstruction network."""

    def __init__(self, input_x, sens_maps, sampling_mask, nb_blocks, num_res_blocks):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.mask = sampling_mask
        self.nb_blocks = nb_blocks
        self.num_res_blocks = num_res_blocks
        self.model = self.build()

    def build(self):
        # Initial CG-SENSE reconstruction. The zero warm start reproduces training.
        initial_lambda = tf.constant(0.05, dtype=tf.float32)
        zero_warm_start = tf.zeros_like(self.input_x)
        x0 = dc(
            self.input_x,
            self.sens_maps,
            self.mask,
            initial_lambda,
            zero_warm_start,
        )

        x = x0
        dc_output = x0

        with tf.name_scope("myModel"):
            # Variable-scope name and reuse behavior must match the training graph
            # so the trained checkpoint can be restored directly.
            with tf.variable_scope("Wts", reuse=tf.AUTO_REUSE):
                for _ in range(self.nb_blocks):
                    x = shifterf(x, CAIPI_SHIFT)
                    x = ResNet(x, self.num_res_blocks)
                    x = shifterb(x, CAIPI_SHIFT)

                    lam = getLambda()
                    rhs = self.input_x + lam * x
                    x = dc(rhs, self.sens_maps, self.mask, lam, dc_output)
                    dc_output = x

        return x, x0, lam


# =============================================================================
# NumPy SENSE-adjoint preprocessing
# =============================================================================


def sense_adjoint(input_kspace, sensitivity_maps):
    """Apply E^H for one SMS slice using its coil sensitivity maps."""
    image_space = ifft(input_kspace, axes=(0, 1), norm=None, unitary_opt=True)
    return np.sum(np.conj(sensitivity_maps) * image_space, axis=2)


def load_fmri_file(file_path):
    """Load and transpose one R=4 fMRI MAT file into inference layout."""
    mat = hdf5storage.loadmat(file_path)

    required_keys = ["kspace_all_r4", "sense_maps_all_small", "padded_all_r4"]
    missing_keys = [key for key in required_keys if key not in mat]
    if missing_keys:
        raise KeyError(
            "Missing required MAT variable(s) in {}: {}".format(
                file_path,
                ", ".join(missing_keys),
            )
        )

    # Match the training layout:
    #   k-space -> [frame, row, col, coil]
    #   maps    -> [frame, slice, row, col, coil]
    #   mask    -> [frame, row, col]
    kspace = np.transpose(np.copy(mat["kspace_all_r4"]), axes=(3, 0, 1, 2))
    maps = np.transpose(
        np.copy(mat["sense_maps_all_small"]),
        axes=(4, 3, 0, 1, 2),
    )
    mask = np.transpose(np.copy(mat["padded_all_r4"]), axes=(3, 0, 1, 2))[..., 0]

    if kspace.shape[1:] != (NROW, NCOL, NCOIL):
        raise ValueError(
            "Unexpected k-space shape {} after transpose; expected [frames, {}, {}, {}].".format(
                kspace.shape,
                NROW,
                NCOL,
                NCOIL,
            )
        )

    if maps.shape[1:] != (SLICE_SIZE, NROW, NCOL, NCOIL):
        raise ValueError(
            "Unexpected sensitivity-map shape {} after transpose; expected "
            "[frames, {}, {}, {}, {}].".format(
                maps.shape,
                SLICE_SIZE,
                NROW,
                NCOL,
                NCOIL,
            )
        )

    if mask.shape[1:] != (NROW, NCOL):
        raise ValueError(
            "Unexpected mask shape {} after transpose; expected [frames, {}, {}].".format(
                mask.shape,
                NROW,
                NCOL,
            )
        )

    if not (kspace.shape[0] == maps.shape[0] == mask.shape[0]):
        raise ValueError(
            "Frame-count mismatch: k-space {}, maps {}, mask {}.".format(
                kspace.shape[0],
                maps.shape[0],
                mask.shape[0],
            )
        )

    return kspace.astype(np.complex64), maps.astype(np.complex64), mask.astype(np.complex64)


def prepare_inference_inputs(kspace, maps, sampling_mask):
    """Normalize k-space and construct the full-mask five-slice SENSE adjoint."""
    num_frames = kspace.shape[0]

    normalized_kspace = np.empty_like(kspace, dtype=np.complex64)
    normalization_scale = np.empty(num_frames, dtype=np.float32)
    atb = np.empty(
        (num_frames, NROW * SLICE_SIZE, NCOL),
        dtype=np.complex64,
    )

    for frame_index in range(num_frames):
        frame_scale = np.max(np.abs(kspace[frame_index]))
        if frame_scale == 0:
            raise ValueError("Zero-valued k-space frame at index {}.".format(frame_index))

        normalization_scale[frame_index] = frame_scale
        normalized_kspace[frame_index] = kspace[frame_index] / frame_scale

        coil_mask = np.tile(
            sampling_mask[frame_index, :, :, np.newaxis],
            (1, 1, NCOIL),
        )
        acquired_kspace = normalized_kspace[frame_index] * coil_mask

        for slice_index in range(SLICE_SIZE):
            row_start = slice_index * NROW
            row_end = (slice_index + 1) * NROW
            atb[frame_index, row_start:row_end, :] = sense_adjoint(
                acquired_kspace,
                maps[frame_index, slice_index],
            )

    # TensorFlow expects sensitivity maps as [batch, slice, coil, row, col].
    maps_tf = np.transpose(maps, (0, 1, 4, 2, 3))

    return (
        c2r(atb).astype(np.float32),
        maps_tf.astype(np.complex64),
        sampling_mask.astype(np.complex64),
        normalization_scale,
    )


def stacked_to_slices(stacked_complex):
    """Convert [frame, 5*row, col] into [frame, slice, row, col]."""
    num_frames = stacked_complex.shape[0]
    return stacked_complex.reshape(num_frames, SLICE_SIZE, NROW, NCOL)


# =============================================================================
# TensorFlow graph and checkpoint restoration
# =============================================================================


def build_inference_graph():
    tf.reset_default_graph()

    atb_placeholder = tf.placeholder(
        tf.float32,
        shape=(None, NROW * SLICE_SIZE, NCOL, 2),
        name="atb",
    )
    csm_placeholder = tf.placeholder(
        tf.complex64,
        shape=(None, SLICE_SIZE, NCOIL, NROW, NCOL),
        name="csm",
    )
    mask_placeholder = tf.placeholder(
        tf.complex64,
        shape=(None, NROW, NCOL),
        name="mask",
    )

    with tf.device("/gpu:{}".format(TEST_GPU)):
        reconstructed, initial_cg, lam = UnrolledInference(
            atb_placeholder,
            csm_placeholder,
            mask_placeholder,
            NB_BLOCKS,
            NUM_RES_BLOCKS,
        ).model

    reconstructed = tf.identity(reconstructed, name="out")
    initial_cg = tf.identity(initial_cg, name="x0")
    lam = tf.identity(lam, name="lam")

    saver = tf.train.Saver()

    return {
        "atb": atb_placeholder,
        "csm": csm_placeholder,
        "mask": mask_placeholder,
        "out": reconstructed,
        "x0": initial_cg,
        "lam": lam,
        "saver": saver,
    }


def resolve_checkpoint():
    if CHECKPOINT_PATH is not None:
        checkpoint = CHECKPOINT_PATH
    else:
        checkpoint = tf.train.latest_checkpoint(MODEL_DIR)

    if checkpoint is None:
        raise FileNotFoundError(
            "No training checkpoint found in MODEL_DIR: {}".format(MODEL_DIR)
        )

    return checkpoint


# =============================================================================
# Inference execution and output saving
# =============================================================================


def reconstruct_file(sess, graph, file_path):
    print("\nLoading test file:", file_path)
    kspace, maps, sampling_mask = load_fmri_file(file_path)

    print("  k-space:", kspace.shape)
    print("  maps:   ", maps.shape)
    print("  mask:   ", sampling_mask.shape)

    atb, maps_tf, mask_tf, normalization_scale = prepare_inference_inputs(
        kspace,
        maps,
        sampling_mask,
    )

    num_frames = atb.shape[0]
    reconstructed_batches = []
    initial_cg_batches = []
    learned_lambda = None

    for batch_start in range(0, num_frames, TEST_BATCH_SIZE):
        batch_end = min(batch_start + TEST_BATCH_SIZE, num_frames)

        feed_dict = {
            graph["atb"]: atb[batch_start:batch_end],
            graph["csm"]: maps_tf[batch_start:batch_end],
            graph["mask"]: mask_tf[batch_start:batch_end],
        }

        fetches = [graph["out"], graph["lam"]]
        if SAVE_INITIAL_CG:
            fetches.insert(1, graph["x0"])

        results = sess.run(fetches, feed_dict=feed_dict)

        reconstructed_batches.append(results[0])

        if SAVE_INITIAL_CG:
            initial_cg_batches.append(results[1])
            learned_lambda = results[2]
        else:
            learned_lambda = results[1]

        print(
            "  reconstructed frames {:4d}-{:4d} / {}".format(
                batch_start + 1,
                batch_end,
                num_frames,
            )
        )

    reconstructed_2ch = np.concatenate(reconstructed_batches, axis=0)
    reconstructed_stacked = r2c(reconstructed_2ch).astype(np.complex64)
    reconstructed = stacked_to_slices(reconstructed_stacked)

    # Restore each frame to its original k-space intensity scale. The model itself
    # always receives normalized inputs, exactly as during training.
    reconstructed_rescaled = (
        reconstructed * normalization_scale[:, np.newaxis, np.newaxis, np.newaxis]
    ).astype(np.complex64)

    output_data = {
        "reconstruction_normalized": reconstructed,
        "reconstruction_rescaled": reconstructed_rescaled,
        "normalization_scale": normalization_scale,
        "sampling_mask": sampling_mask,
        "lambda": np.asarray(learned_lambda),
    }

    if SAVE_INITIAL_CG:
        initial_cg_2ch = np.concatenate(initial_cg_batches, axis=0)
        initial_cg_stacked = r2c(initial_cg_2ch).astype(np.complex64)
        initial_cg = stacked_to_slices(initial_cg_stacked)
        output_data["initial_cg_normalized"] = initial_cg
        output_data["initial_cg_rescaled"] = (
            initial_cg
            * normalization_scale[:, np.newaxis, np.newaxis, np.newaxis]
        ).astype(np.complex64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(
        OUTPUT_DIR,
        base_name + "_SSDU_fMRI_R4_recon.mat",
    )

    # MATLAB v7.3/HDF5 output is used because fMRI time-series arrays can be large.
    hdf5storage.savemat(
        output_path,
        output_data,
        format="7.3",
    )

    print("  saved:", output_path)
    return output_path


def main():
    if len(TEST_FILES) == 0:
        raise ValueError(
            "TEST_FILES is empty. Add one or more R=4 fMRI MAT files in the "
            "configuration section at the top of this script."
        )

    missing_files = [path for path in TEST_FILES if not os.path.isfile(path)]
    if missing_files:
        raise FileNotFoundError(
            "Test file(s) not found:\n  " + "\n  ".join(missing_files)
        )

    checkpoint = resolve_checkpoint()
    print("Restoring checkpoint:", checkpoint)

    graph = build_inference_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        graph["saver"].restore(sess, checkpoint)
        print("Model restored successfully.")

        output_files = []
        for file_path in TEST_FILES:
            output_files.append(reconstruct_file(sess, graph, file_path))

    print("\nTesting complete.")
    for output_file in output_files:
        print(" ", output_file)


if __name__ == "__main__":
    main()
