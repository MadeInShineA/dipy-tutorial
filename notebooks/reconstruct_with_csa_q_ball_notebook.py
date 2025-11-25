import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np

    from dipy.core.gradients import gradient_table
    from dipy.data import default_sphere, get_fnames
    from dipy.direction import peaks_from_model
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti
    from dipy.reconst.shm import CsaOdfModel
    from dipy.segment.mask import median_otsu
    from dipy.viz import actor, window
    from PIL import Image
    return (
        CsaOdfModel,
        Image,
        actor,
        default_sphere,
        get_fnames,
        gradient_table,
        load_nifti,
        median_otsu,
        mo,
        np,
        peaks_from_model,
        read_bvals_bvecs,
        window,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Reconstruct with Constant Solid Angle (Q-Ball)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load the data
    """)
    return


@app.cell
def _(get_fnames, gradient_table, load_nifti, read_bvals_bvecs):
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

    data, affine = load_nifti(hardi_fname)

    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs=bvecs)
    return bvals, bvecs, data, gtab


@app.cell
def _(bvals, bvecs, data):
    print(f"data.shape {data.shape}")
    print(f"bvals.shape {bvals.shape}")
    print(f"bvecs.shape {bvecs.shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Display the first volume
    """)
    return


@app.cell
def _(Image, actor, bvals, data, np, window):
    # Pick b=0 volume
    b0_idx = np.where(bvals == 0)[0][0]
    vol = data[:, :, :, b0_idx].astype(np.float32)

    # Normalize for visualization (2–98 percentile)
    p2, p98 = np.percentile(vol, (2, 98))
    vol = np.clip(vol, p2, p98)
    vol = (vol - p2) / (p98 - p2 + 1e-8)

    # Create scene
    raw_scene = window.Scene()

    # Use slicer (shows 3 orthogonal planes)
    slicer_actor = actor.slicer(vol)
    raw_scene.add(slicer_actor)

    # Optional: position the slice at center
    slicer_actor.display(x=vol.shape[0] // 2, y=vol.shape[1] // 2, z=vol.shape[2] // 2)

    # Save as PNG
    print("Saving raw volume snapshot as res/raw_volume.png")
    window.record(scene=raw_scene, out_path="./res/raw_volume.png", size=(600, 600))

    # Render to NumPy array → PIL Image
    img_array = window.snapshot(raw_scene, size=(600, 600))
    img = Image.fromarray(img_array)
    img
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Use a mask to remove the background
    """)
    return


@app.cell
def _(data, median_otsu):
    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2
    )
    return mask, maskdata


@app.cell
def _(mo):
    mo.md(r"""
    ## Initialize the CSA with spherical harmonic order ($\ell$) of 4
    """)
    return


@app.cell
def _(CsaOdfModel, gtab):
    csamodel = CsaOdfModel(gtab, 4)
    return (csamodel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Get the CSA peaks
    """)
    return


@app.cell
def _(csamodel, default_sphere, mask, maskdata, peaks_from_model):
    csapeaks = peaks_from_model(
        model=csamodel,
        data=maskdata,
        sphere=default_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=mask,
        return_odf=False,
        normalize_peaks=True,
    )

    GFA = csapeaks.gfa

    print(f"GFA.shape {GFA.shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Slice the data
    """)
    return


@app.cell
def _(maskdata):
    data_small = maskdata[13:43, 44:74, 28:29]
    return (data_small,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Render the data slice
    """)
    return


@app.cell
def _(csamodel, data_small, default_sphere, np, window):
    scene = window.Scene()

    csaodfs = csamodel.fit(data_small).odf(default_sphere)
    csaodfs = np.clip(csaodfs, 0, np.max(csaodfs, -1)[..., None])
    return csaodfs, scene


@app.cell
def _(Image, actor, csaodfs, default_sphere, scene, window):
    csa_odfs_actor = actor.odf_slicer(
        csaodfs, sphere=default_sphere, colormap="plasma", scale=0.4
    )
    csa_odfs_actor.display(z=0)

    scene.add(csa_odfs_actor)
    print("Saving illustration as res/csa_odfs.png")
    window.record(scene=scene, n_frames=1, out_path="./res/csa_odfs.png", size=(600, 600))

    # Render scene to a NumPy array (RGB)
    _img_array = window.snapshot(scene, size=(600, 600))

    # Convert to PIL Image
    _img = Image.fromarray(_img_array)
    _img
    return


@app.cell
def _():
    # Enables/disables interactive visualization
    interactive = False
    return (interactive,)


@app.cell
def _(interactive, scene, window):
    if interactive:
        window.show(scene)
    return


if __name__ == "__main__":
    app.run()
