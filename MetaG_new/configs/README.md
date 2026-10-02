# Configurations

This directory records named scientific-policy releases. During the parity
phase, environment variables remain the executable source of configuration in
`metag.pipeline.effective_config()`.

`frozen-2026-10-01c.json` identifies the reference policy and its immutable
regression fixture. A later refactor will load a typed configuration object
and provide an explicit compatibility adapter for the current environment
variables.

