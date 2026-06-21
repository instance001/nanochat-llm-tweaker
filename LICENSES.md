# License Guide

This package includes more than one licensing layer.

## 1. ChattyCog Module Wrapper

The outer module wrapper in this folder is intended to be distributed under AGPLv3.

This includes files such as:

- [manifest.json](manifest.json)
- [visual_load.json](visual_load.json)
- [HANDSHAKE.md](HANDSHAKE.md)
- [ui.json](ui.json)
- [state.json](state.json)
- [start-builder.cmd](start-builder.cmd)
- [README.md](README.md)

See:

- [LICENSE](LICENSE)

## 2. Nested Builder App

The nested builder application in [nanochat-master](nanochat-master) is upstream-derived and carries its own license file.

See:

- [nanochat-master/LICENSE](nanochat-master/LICENSE)

That license should continue to travel with the nested builder app and any upstream-derived files inside it.

## 3. Practical Rule For Redistribution

If you copy or redistribute this packaged module, keep:

- the outer wrapper [LICENSE](LICENSE)
- this [LICENSES.md](LICENSES.md) explainer
- the nested [nanochat-master/LICENSE](nanochat-master/LICENSE)

## 4. Attribution

The nested builder app is based on Andrej Karpathy's `nanochat` project and should continue to preserve its upstream attribution and license notices.

For context, see:

- [README.md](README.md)
- [nanochat-master/README.md](nanochat-master/README.md)
- [nanochat-master/LOCAL_BUILDER_USER_MANUAL.md](nanochat-master/LOCAL_BUILDER_USER_MANUAL.md)
