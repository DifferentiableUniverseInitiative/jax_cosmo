# jax-cosmo

[![Join the chat at https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo](https://badges.gitter.im/DifferentiableUniverseInitiative/jax_cosmo.svg)](https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Documentation Status](https://readthedocs.org/projects/jax-cosmo/badge/?version=latest)](https://jax-cosmo.readthedocs.io/en/latest/?badge=latest) [![CI Test](https://github.com/DifferentiableUniverseInitiative/jax_cosmo/workflows/Python%20package/badge.svg)]() [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![PyPI](https://img.shields.io/pypi/v/jax-cosmo)](https://pypi.org/project/jax-cosmo/) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md) [![PyPI - License](https://img.shields.io/pypi/l/jax-cosmo)](https://github.com/google/jax-cosmo/blob/master/LICENSE) <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-11-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<h3 align="center">Finally a differentiable cosmology library, and it's in JAX!</h3>

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

Please take a look at the [Contributing Document](CONTRIBUTING.md) for more information.

This project follows the [All Contributors](https://allcontributors.org/) guidelines aiming at recognizing and valorizing
contributions at any levels.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://flanusse.net"><img src="https://avatars0.githubusercontent.com/u/861591?v=4?s=100" width="100px;" alt="Francois Lanusse"/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=EiffL" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cosmostat.org/people/santiago-casas"><img src="https://avatars0.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues?q=author%3Asantiagocasas" title="Bug reports">🐛</a> <a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=santiagocasas" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/austinpeel"><img src="https://avatars0.githubusercontent.com/u/17024310?v=4?s=100" width="100px;" alt="Austin Peel"/><br /><sub><b>Austin Peel</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=austinpeel" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://minaskaramanis.com"><img src="https://avatars2.githubusercontent.com/u/23280751?v=4?s=100" width="100px;" alt="Minas Karamanis"/><br /><sub><b>Minas Karamanis</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=minaskar" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://faculty.sites.uci.edu/dkirkby/"><img src="https://avatars1.githubusercontent.com/u/185007?v=4?s=100" width="100px;" alt="David Kirkby"/><br /><sub><b>David Kirkby</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=dkirkby" title="Code">💻</a> <a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues?q=author%3Adkirkby" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://aboucaud.github.io"><img src="https://avatars0.githubusercontent.com/u/3065310?v=4?s=100" width="100px;" alt="Alexandre Boucaud"/><br /><sub><b>Alexandre Boucaud</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=aboucaud" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.cosmostat.org/people/denise-lanzieri"><img src="https://avatars.githubusercontent.com/u/72620117?v=4?s=100" width="100px;" alt="Denise Lanzieri"/><br /><sub><b>Denise Lanzieri</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=dlanzieri" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jecampagne"><img src="https://avatars.githubusercontent.com/u/20539759?v=4?s=100" width="100px;" alt="jecampagne"/><br /><sub><b>jecampagne</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues?q=author%3Ajecampagne" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eelregit"><img src="https://avatars.githubusercontent.com/u/7311098?v=4?s=100" width="100px;" alt="Yin Li"/><br /><sub><b>Yin Li</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=eelregit" title="Code">💻</a> <a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues?q=author%3Aeelregit" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mihir-r-kondapalli"><img src="https://avatars.githubusercontent.com/u/90729208?v=4?s=100" width="100px;" alt="mihir-r-kondapalli"/><br /><sub><b>mihir-r-kondapalli</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues?q=author%3Amihir-r-kondapalli" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aymgal"><img src="https://avatars.githubusercontent.com/u/10706063?v=4?s=100" width="100px;" alt="Aymeric Galan"/><br /><sub><b>Aymeric Galan</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo/commits?author=aymgal" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
