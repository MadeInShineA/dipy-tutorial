import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from dipy.core.gradients import gradient_table
    from dipy.data import get_fnames, get_sphere
    from dipy.direction import peaks_from_model
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti
    from dipy.reconst.gqi import GeneralizedQSamplingModel
    return (
        GeneralizedQSamplingModel,
        get_fnames,
        get_sphere,
        gradient_table,
        load_nifti,
        mo,
        np,
        peaks_from_model,
        read_bvals_bvecs,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Reconstruct with Generalized Q-Sampling Imaging
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load the data
    """)
    return


@app.cell
def _(get_fnames):
    # Load the raw diffusion MRI data (4D NIfTI), b-values, and b-vectors from the built-in DSI dataset
    q_fraw, q_fbval, q_fbvec = get_fnames(name="taiwan_ntu_dsi")
    return q_fbval, q_fbvec, q_fraw


@app.cell
def _(
    gradient_table,
    load_nifti,
    np,
    q_fbval,
    q_fbvec,
    q_fraw,
    read_bvals_bvecs,
):
    # Load the 4D image data, its affine transformation matrix, and voxel dimensions (in mm)
    q_data, q_affine, q_voxel_size = load_nifti(q_fraw, return_voxsize=True)

    # Load b-values (diffusion weighting strengths) and b-vectors (gradient directions)
    q_bvals, q_bvecs = read_bvals_bvecs(q_fbval, q_fbvec)

    # Normalize b-vectors to unit length (skip the first row, which corresponds to the b=0 volume and should remain [0,0,0])
    # This ensures accurate directional modeling in diffusion reconstruction
    q_bvecs[1:] = q_bvecs[1:] / np.sqrt(np.sum(q_bvecs[1:] * q_bvecs[1:], axis=1))[:, None]

    # Create a DIPY gradient table object that combines b-values and normalized b-vectors for use in reconstruction models
    q_gtab = gradient_table(q_bvals, bvecs=q_bvecs)

    # Print the shape of the diffusion data (x, y, z, number_of_volumes) for verification
    print(f"data.shape {q_data.shape}")
    return q_data, q_gtab


@app.cell
def _(mo):
    mo.md(r"""
    ## Initialize the Qball model
    """)
    return


@app.cell
def _(GeneralizedQSamplingModel, q_gtab):
    gqmodel = GeneralizedQSamplingModel(q_gtab, sampling_length=3)
    return (gqmodel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Only select 1 slice
    """)
    return


@app.cell
def _(q_data):
    q_dataslice = q_data[:, :, q_data.shape[2] // 2]

    q_mask = q_dataslice[..., 0] > 50
    return q_dataslice, q_mask


@app.cell
def _(mo):
    mo.md(r"""
    ## Fit the slice to the Qball model
    """)
    return


@app.cell
def _(gqmodel, q_dataslice, q_mask):
    q_gqfit = gqmodel.fit(q_dataslice, mask=q_mask)
    return (q_gqfit,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Get an ODF (orientation distribution function) representation sphere
    """)
    return


@app.cell
def _(get_sphere):
    sphere = get_sphere(name="repulsion724")
    return (sphere,)


@app.cell
def _(q_gqfit, sphere):
    q_ODF = q_gqfit.odf(sphere)

    print(f"ODF.shape {q_ODF.shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Get the main peaks from the ODF
    """)
    return


@app.cell
def _(gqmodel, peaks_from_model, q_dataslice, q_mask, sphere):
    gqpeaks = peaks_from_model(
        model=gqmodel,
        data=q_dataslice,
        sphere=sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=q_mask,
        return_odf=False,
        normalize_peaks=True,
    )
    return


if __name__ == "__main__":
    app.run()
