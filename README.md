# jax-cosmo

[![Join the chat at https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo](https://badges.gitter.im/DifferentiableUniverseInitiative/jax_cosmo.svg)](https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Documentation Status](https://readthedocs.org/projects/jax-cosmo/badge/?version=latest)](https://jax-cosmo.readthedocs.io/en/latest/?badge=latest) [![CI Test](https://github.com/DifferentiableUniverseInitiative/jax_cosmo/workflows/Python%20package/badge.svg)]() [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![PyPI](https://img.shields.io/pypi/v/jax-cosmo)](https://pypi.org/project/jax-cosmo/) [![PyPI - License](https://img.shields.io/pypi/l/jax-cosmo)](https://github.com/google/jax-cosmo/blob/master/LICENSE)  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<h3 align="center">Finally a differentiable cosmology library, and it's in JAX!</h3>


**Note**: This package is still in the development phase, expect changes to the API. **We hope to make this project an open and collaborative community effort**, contributions of all kind are most welcome! A paper is being prepared [here](https://github.com/DifferentiableUniverseInitiative/jax-cosmo-paper), at this point, contributions to this project automatically grants you authorship to the paper :-)  
Have a look at the [GitHub issues](https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues) to see what is needed or if you have any thoughts on the design, and don't hesitate to join the [Gitter room](https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo) for discussions.

## TL;DR

This is what `jax-cosmo` aims to do:

```python
...
def likelihood(cosmo):
  # Compute mean and covariance of angular Cls, for specific probes
  mu, cov = jax_cosmo.angular_cl.gaussian_cl_covariance_and_mean(cosmo, ell, probes)
  # Return likelihood value
  return jax_cosmo.likelihood.gaussian_log_likelihood(data, mu, cov)

# Compute derivatives of the likelihood with respect to cosmological parameters
g = jax.grad(likelihood)(cosmo)

# Compute Fisher matrix of cosmological parameters
F = - jax.hessian(likelihood)(cosmo)
```
This is how you can compute **gradients and hessians of any functions in `jax-cosmo`**,
all of this without any finite differences.

Check out a full example here: [![colab link](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DifferentiableUniverseInitiative/jax_cosmo/blob/master/docs/notebooks/jax-cosmo-intro.ipynb)

Have a look at the [design document](design.md) to learn more about the structure of the code.

## What is JAX?

[JAX](https://github.com/google/jax) = NumPy + autodiff + GPU

JAX is a framework for automatic differentiation (like TensorFlow or PyTorch) but following the NumPy API, and using the GPU/TPU enable XLA backend.

What does that mean?
  - You write plain Python/NumPy code, no need to learn a different language
  - It runs on GPU, you don't need to do anything particular
  - You can take derivatives of any quantity with respect to any parameters by
  automatic differentiation.

Checkout the [JAX](https://github.com/google/jax) project page to learn more!

## Install

`jax-cosmo` is pure Python, so installing is a breeze:
```bash
$ pip install jax-cosmo
```

## Philosophy

Here are some of the design guidelines:
  - Implementation of equations should be human readable, and documentation should always live next to the implementation.
  - Should always be trivially installable: external dependencies should be kept
  to a minimum, especially the ones that require compilation or with restrictive licenses.
  - Keep API and implementation simple and intuitive, minimize user and developer
  surprise.
  - “Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it.” -Brian Kernighan, quote stolen from
  [here](https://flax.readthedocs.io/en/latest/philosophy.html).

## Contributing

`jax-cosmo` aims to be a community effort, contributions are most welcome and
can come in several forms
  - Bug reports
  - API design suggestions
  - (Pull) requests for more features
  - Examples and notebooks of cool things that can be done with the code

You can chime-in on any aspects of the design by proposing a PR to the [design document](design.md). The issue page is a good place to start, but don't hesitate to come chat in the Gitter room.

This project follows the [All Contributors](https://allcontributors.org/) guidelines aiming at recognizing and valorizing
contributions at any levels.  

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://flanusse.net"><img src="https://avatars0.githubusercontent.com/u/861591?v=4" width="100px;" alt=""/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=EiffL" title="Code">💻</a></td>
    <td align="center"><a href="http://www.cosmostat.org/people/santiago-casas"><img src="https://avatars0.githubusercontent.com/u/6987716?v=4" width="100px;" alt=""/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues?q=author%3Asantiagocasas" title="Bug reports">🐛</a> <a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=santiagocasas" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/austinpeel"><img src="https://avatars0.githubusercontent.com/u/17024310?v=4" width="100px;" alt=""/><br /><sub><b>Austin Peel</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=austinpeel" title="Code">💻</a></td>
    <td align="center"><a href="https://minaskaramanis.com"><img src="https://avatars2.githubusercontent.com/u/23280751?v=4" width="100px;" alt=""/><br /><sub><b>Minas Karamanis</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=minaskar" title="Code">💻</a></td>
    <td align="center"><a href="https://faculty.sites.uci.edu/dkirkby/"><img src="https://avatars1.githubusercontent.com/u/185007?v=4" width="100px;" alt=""/><br /><sub><b>David Kirkby</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=dkirkby" title="Code">💻</a> <a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues?q=author%3Adkirkby" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://aboucaud.github.io"><img src="https://avatars0.githubusercontent.com/u/3065310?v=4" width="100px;" alt=""/><br /><sub><b>Alexandre Boucaud</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=aboucaud" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
