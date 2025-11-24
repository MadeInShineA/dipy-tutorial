import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from dipy.data import default_sphere, get_fnames
    from dipy.io.image import load_nifti, load_nifti_data
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.direction import peaks_from_model
    from dipy.io.streamline import save_trk
    from dipy.reconst.shm import CsaOdfModel
    from dipy.tracking import utils
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.tracker import eudx_tracking
    from dipy.viz import actor, colormap, has_fury, window
    from PIL import Image
    return (
        CsaOdfModel,
        Image,
        Streamlines,
        ThresholdStoppingCriterion,
        actor,
        colormap,
        default_sphere,
        eudx_tracking,
        get_fnames,
        gradient_table,
        load_nifti,
        load_nifti_data,
        mo,
        peaks_from_model,
        plt,
        read_bvals_bvecs,
        utils,
        window,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Introduction to Basic Tracking
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load the data
    """)
    return


@app.cell
def _(
    get_fnames,
    gradient_table,
    load_nifti,
    load_nifti_data,
    read_bvals_bvecs,
):
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
    label_fname = get_fnames(name="stanford_labels")

    data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
    labels = load_nifti_data(label_fname)
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs=bvecs)
    return affine, data, gtab, labels


@app.cell
def _(mo):
    mo.md(r"""
    ## Create the white matter filter
    """)
    return


@app.cell
def _(labels):
    white_matter = (labels == 1) | (labels == 2)
    return (white_matter,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Create the ODF model (£Constant Solid Angle)
    """)
    return


@app.cell
def _(CsaOdfModel, data, default_sphere, gtab, peaks_from_model, white_matter):
    csa_model = CsaOdfModel(gtab, sh_order_max=6)
    csa_peaks = peaks_from_model(
        csa_model,
        data,
        default_sphere,
        relative_peak_threshold=0.8,
        min_separation_angle=45,
        mask=white_matter,
    )
    return (csa_peaks,)


@app.cell
def _(mo):
    mo.md(r"""
    # Render the ODF peak directions
    """)
    return


@app.cell
def _():
    interactive = False
    return (interactive,)


@app.cell
def _(Image, actor, csa_peaks, window):
    csa_scene = window.Scene()
    csa_scene.add(
        actor.peak_slicer(
            csa_peaks.peak_dirs, peaks_values=csa_peaks.peak_values, colors=None
        )
    )

    # Save image
    window.record(scene=csa_scene, out_path="res/csa_direction_field.png", size=(900, 900))


    # Render scene to a NumPy array (RGB)
    _img_array = window.snapshot(csa_scene, size=(900, 900))

    # Convert to PIL Image
    _img = Image.fromarray(_img_array)
    _img
    return (csa_scene,)


@app.cell
def _(csa_scene, interactive, window):
    if interactive:
        window.show(csa_scene, size=(900, 900))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Improve the threshold
    """)
    return


@app.cell
def _(csa_peaks, plt):
    sli = csa_peaks.gfa.shape[2] // 2
    plt.figure("GFA")
    plt.subplot(1, 2, 1).set_axis_off()
    plt.imshow(csa_peaks.gfa[:, :, sli].T, cmap="gray", origin="lower")

    plt.subplot(1, 2, 2).set_axis_off()
    plt.imshow((csa_peaks.gfa[:, :, sli] > 0.25).T, cmap="gray", origin="lower")

    plt.show()
    return


@app.cell
def _(ThresholdStoppingCriterion, csa_peaks):
    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, 0.25)
    return (stopping_criterion,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Define the seed (where to begin tracking)
    """)
    return


@app.cell
def _(affine, labels, utils):
    seed_mask = labels == 2
    seeds = utils.seeds_from_mask(seed_mask, affine, density=[2, 2, 2])
    return (seeds,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Use EuDX algorithm to generate streamlines
    """)
    return


@app.cell
def _(
    Streamlines,
    affine,
    csa_peaks,
    eudx_tracking,
    seeds,
    stopping_criterion,
):
    # Initialization of eudx_tracking. The computation happens in the next step.
    streamlines_generator = eudx_tracking(
        seeds, stopping_criterion, affine, step_size=0.5, pam=csa_peaks
    )
    # Generate streamlines object
    streamlines = Streamlines(streamlines_generator)
    return (streamlines,)


@app.cell
def _(Image, actor, colormap, streamlines, window):
    # Prepare the display objects.
    color = colormap.line_colors(streamlines)

    streamlines_actor = actor.line(
        streamlines, colors=colormap.line_colors(streamlines)
    )

    # Create the 3D display.
    streamlines_scene = window.Scene()
    streamlines_scene.add(streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    window.record(scene=streamlines_scene, out_path="res/tractogram_EuDX.png", size=(800, 800))

    # Render scene to a NumPy array (RGB)
    _img_array = window.snapshot(streamlines_scene, size=(800, 800))

    # Convert to PIL Image
    _img = Image.fromarray(_img_array)
    _img
    return


if __name__ == "__main__":
    app.run()
