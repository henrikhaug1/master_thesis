from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Domain:
    """Where collocation points may go; the strategies in src.config read this.

    lo, hi: lower and upper corners of a box that contains the domain.
    inside: which points of the box belong to the domain, called with one
        coordinate array per dimension (see grid_points); None keeps them all.
    center, theta_range, r_max_fn: the domain in polar coordinates around the
        point graded points pack toward (see graded_points). Only
        GradedCollocation needs them.
    """

    lo: tuple[float, ...]
    hi: tuple[float, ...]
    inside: Callable | None = None
    center: tuple[float, float] | None = None
    theta_range: tuple[float, float] | None = None
    r_max_fn: Callable | None = None


@dataclass(frozen=True)
class Resampler:
    """Collocation points that move during training (see src.train.train).

    initial: the points training starts on.
    draw: draw(model, key) returns new points, as many as `initial`.
    every: steps between draws. Only first-order stages draw: L-BFGS's
        curvature pairs assume one fixed loss, so it keeps the last points.
    seed: of the draws' keys.
    """

    initial: Array
    draw: Callable
    every: int
    seed: int = 0


def grid_points(lo, hi, n, inside=None):
    """An evenly spaced grid of n points per axis over the box from `lo` to `hi`.

    Points come in the order meshgrid(indexing="ij") flattens them, so values
    on the full grid reshape back to (n, ..., n).

    Args:
        lo, hi: the box's lower and upper corners, one entry per dimension.
        n: points per axis, endpoints included.
        inside: optional test of which points belong to the domain; called with
            one coordinate array per dimension, it returns a boolean array.

    Returns (m, d) points: all n^d of them, or those `inside` keeps.
    """
    axes = [jnp.linspace(a, b, n) for a, b in zip(lo, hi)]
    pts = jnp.stack([g.ravel() for g in jnp.meshgrid(*axes, indexing="ij")], axis=-1)
    if inside is not None:
        pts = pts[inside(*pts.T)]
    return pts


def random_points(lo, hi, n, key, inside=None):
    """n points drawn uniformly at random from the part of the box `inside` keeps.

    Draws n points from the whole box at a time and keeps those inside, until
    there are n; the same `key` always gives the same points.
    """
    lo, hi = jnp.asarray(lo, dtype=float), jnp.asarray(hi, dtype=float)
    pts = jnp.empty((0, lo.size))
    while pts.shape[0] < n:
        key, sub = jax.random.split(key)
        new = jax.random.uniform(sub, (n, lo.size), minval=lo, maxval=hi)
        if inside is not None:
            new = new[inside(*new.T)]
        pts = jnp.concatenate([pts, new])
    return pts[:n]


def segment_points(p0, p1, n=None, density=None, endpoint=True, beta=1.0):
    """Points on the straight line from p0 to p1, evenly spaced or packed toward p0.

    Give either `n`, the number of points, or `density`, points per unit
    length, so that edges of different lengths are sampled alike.
    endpoint=False leaves out p1, for chaining segments without repeating the
    points where they meet.
    beta: the point at fraction s of the way along an even spacing moves to
        s^beta, so beta = 1 is even and beta > 1 packs the points toward p0,
        as graded_points does along its rays.
    """
    if (n is None) == (density is None):
        raise ValueError("Give exactly one of n and density")
    p0, p1 = jnp.asarray(p0), jnp.asarray(p1)
    if n is None:
        n = max(2, round(float(jnp.linalg.norm(p1 - p0)) * density))
    s = jnp.linspace(0.0, 1.0, n, endpoint=endpoint)[:, None] ** beta
    return (1.0 - s) * p0 + s * p1


def graded_points(
    center, theta_min, theta_max, r_max_fn, n_r, n_theta, beta, s_min=0.0
):
    """Collocation points graded toward `center`, with the area each one stands for.

    A point at angle theta and fraction s of the way out to the boundary sits at
    radius r = r_max_fn(theta) * s^beta. s and theta are uniform midpoint grids,
    so beta = 1 is an ordinary polar grid and beta > 1 packs points toward the
    center; the midpoints keep every point off the center and off the edges
    theta = theta_min and theta = theta_max.

    In polar coordinates dA = r dr dtheta, and dr = R beta s^(beta - 1) ds, so
    each point stands for an area beta R^2 s^(2 beta - 1) ds dtheta. Passed as
    loss_fn's `weights`, these keep the residual loss an integral over the
    domain however the points are spread.

    Args:
        center: (x, y) of the point the grading is toward.
        theta_min, theta_max: the angles the domain covers, seen from `center`.
        r_max_fn: distance from `center` to the boundary along angle theta;
            vectorized over theta. A constant gives a disk sector.
        n_r, n_theta: points along s and along theta.
        beta: grading exponent, at least 1.
        s_min: the innermost s, to keep points further from the center.

    Returns (n_r * n_theta, 2) points and (n_r * n_theta,) areas.
    """
    ds = (1.0 - s_min) / n_r
    dtheta = (theta_max - theta_min) / n_theta
    s = s_min + (jnp.arange(n_r) + 0.5) * ds
    theta = theta_min + (jnp.arange(n_theta) + 0.5) * dtheta
    S, T = jnp.meshgrid(s, theta, indexing="ij")

    R = r_max_fn(T)
    r = R * S**beta
    area = beta * R**2 * S ** (2 * beta - 1) * ds * dtheta

    cx, cy = center
    pts = jnp.stack([cx + r * jnp.cos(T), cy + r * jnp.sin(T)], axis=-1)
    return pts.reshape(-1, 2), area.ravel()


def residual_density_points(model, residual, candidates, n, key, k=1.0, c=1.0):
    """n of the candidates, drawn without replacement with probability
    proportional to |r|^k / mean(|r|^k) + c, r the residual at each (RAD)."""
    r = jnp.abs(residual(model, candidates)) ** k
    p = r / jnp.mean(r) + c
    idx = jax.random.choice(
        key, candidates.shape[0], (n,), replace=False, p=p / jnp.sum(p)
    )
    return candidates[idx]
