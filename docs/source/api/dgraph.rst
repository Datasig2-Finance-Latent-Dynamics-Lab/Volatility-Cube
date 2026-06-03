DGraph
======

Core graph-based dynamic model for fitting and rolling implied-volatility surfaces
across time using parametric node states, directed edges, and joint/separable optimisation.

Source Layer
------------

.. automodule:: dgraph.source.node
   :members:

.. automodule:: dgraph.source.state
   :members:

.. automodule:: dgraph.source.edge
   :members:

.. automodule:: dgraph.source.observation
   :members:

.. automodule:: dgraph.source.graph
   :members:

Losses
------

.. automodule:: dgraph.losses.data
   :members:

.. automodule:: dgraph.losses.node
   :members:

.. automodule:: dgraph.losses.temporal
   :members:

.. automodule:: dgraph.losses.graph
   :members:

.. automodule:: dgraph.losses.combined
   :members:

Time-Stepping
-------------

.. automodule:: dgraph.time_stepping.roller
   :members:

.. automodule:: dgraph.time_stepping.updater
   :members:

Experiments
-----------

.. automodule:: dgraph.experiments.splitter
   :members:

.. automodule:: dgraph.experiments.experiment
   :members:

Vol-Smile Example
-----------------

.. automodule:: dgraph.examples.vol_smiles.source.curves.base
   :members:

.. automodule:: dgraph.examples.vol_smiles.source.curves.svi
   :members:

.. automodule:: dgraph.examples.vol_smiles.source.curves.bspline
   :members:

.. automodule:: dgraph.examples.vol_smiles.source.nodes
   :members:

.. automodule:: dgraph.examples.vol_smiles.source.edges
   :members:

.. automodule:: dgraph.examples.vol_smiles.source.factory
   :members:

.. automodule:: dgraph.examples.vol_smiles.time_stepping.rollers
   :members:

.. automodule:: dgraph.examples.vol_smiles.losses.data
   :members:

.. automodule:: dgraph.examples.vol_smiles.losses.node
   :members:

Vol-Surface Example
-------------------

.. automodule:: dgraph.examples.vol_surface.source.states.base
   :members:

.. automodule:: dgraph.examples.vol_surface.source.states.ssvi
   :members:

.. automodule:: dgraph.examples.vol_surface.source.states.pca
   :members:

.. automodule:: dgraph.examples.vol_surface.source.factory
   :members:

.. automodule:: dgraph.examples.vol_surface.source.predictor
   :members:

.. automodule:: dgraph.examples.vol_surface.time_stepping.rollers
   :members:

.. automodule:: dgraph.examples.vol_surface.losses.data
   :members:

.. automodule:: dgraph.examples.vol_surface.losses.node
   :members:
