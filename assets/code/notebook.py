# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax==0.6.2",
#     "marimo",
#     "sparsediffax==0.1.0",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import sparsediffax as sd
    return jax, jnp, sd


@app.cell
def _(jnp):
    def f(x):
        return jnp.sum(jnp.cos(jnp.diff(x)))
    return (f,)


@app.cell
def _(jnp):
    x = jnp.arange(1, 1000, dtype=jnp.float32)
    return (x,)


@app.cell
def _(f, sd, x):
    S = sd.naive_hessian_sparsity(f)(x)
    return (S,)


@app.cell
def _(S):
    S.todense()
    return


@app.cell
def _(f, jax):
    H = jax.jit(jax.hessian(f))
    return (H,)


@app.cell
def _(S, f, jax, sd):
    Hs = jax.jit(sd.hessian_sparse(f, S))
    return (Hs,)


@app.cell
def _(H, x):
    H(x)
    return


@app.cell
def _(Hs, x):
    Hs(x)
    return


@app.cell
def _(Hs, x):
    Hs(x).todense()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
