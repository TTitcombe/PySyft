<!---Documatic-section-fixed: top1-start--->
<h1 align="center">

  <br>
  <a href="http://duet.openmined.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/logo_big.png" alt="PySyft" width="200"></a>
  <br>
  A library for computing on data<br /> you do not own and cannot see
  <br>

</h1>

<div align="center"> <a href="https://pypi.org/project/syft/"><img src="https://pepy.tech/badge/syft" /></a> <a href="https://pypi.org/project/syft/"><img src="https://badge.fury.io/py/syft.svg" /></a> <a href="https://github.com/OpenMined/PySyft/actions/workflows/syft-version_tests.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/syft-version_tests.yml/badge.svg?branch=dev" /></a> <a href="https://openmined.github.io/PySyft/dev/bench/"><img src="https://github.com/OpenMined/PySyft/actions/workflows/syft-benchmark.yml/badge.svg?branch=dev" /></a><br /> <a href="#"><img src="https://github.com/OpenMined/PySyft/workflows/Tutorials/badge.svg" /></a> <a href="https://openmined.slack.com/messages/support"><img src="https://img.shields.io/badge/chat-on%20slack-7A5979.svg" /></a> <a href="https://mybinder.org/v2/gh/OpenMined/PySyft/main"><img src="https://mybinder.org/badge.svg" /></a> <a href="http://colab.research.google.com/github/OpenMined/PySyft/blob/main"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
<br /><br />

<div align="center"><a href="#"><img src="https://stars.medv.io/openmined/pysyft.svg" /></a></div>

</div>

# PySyft is a Python library for secure and private Deep Learning.

PySyft decouples private data from model training, using
[Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html),
[Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy),
and Encrypted Computation (like
[Multi-Party Computation (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
and [Homomorphic Encryption (HE)](https://en.wikipedia.org/wiki/Homomorphic_encryption))
within the main Deep Learning frameworks like PyTorch and TensorFlow. Join the movement on
[Slack](http://slack.openmined.org/).

---

Most software libraries let you compute over the information you own and see inside of machines you control. However, this means that you cannot compute on information without first obtaining (at least partial) ownership of that information. It also means that you cannot compute using machines without first obtaining control over those machines. This is very limiting to human collaboration and systematically drives the centralization of data, because you cannot work with a bunch of data without first putting it all in one (central) place.

The Syft ecosystem seeks to change this system, allowing you to write software which can compute over information you do not own on machines you do not have (total) control over. This not only includes servers in the cloud, but also personal desktops, laptops, mobile phones, websites, and edge devices. Wherever your data wants to live in your ownership, the Syft ecosystem exists to help keep it there while allowing it to be used privately for computation.

<!---Documatic-section-fixed: top1-end--->
<!---Documatic-section-fixed: top2-start--->
## Mono Repo 🚝

This repo contains multiple projects which work together, namely PySyft and PyGrid.

```
OpenMined/PySyft
├── README.md
└── packages
    ├── grid    <-- Code formerly @ OpenMined/PyGrid
    └── syft    <-- You are here 📌
```

_NOTE_ Changing the entire folder structure will likely result in some minor issues.
If you spot one please let us know or open a PR.

## PySyft

PySyft is the centerpiece of the Syft ecosystem. It has two primary purposes. You can either use PySyft to perform two types of computation:

1. _Dynamic:_ Directly compute over data you cannot see.
2. _Static:_ Create static graphs of computation which can be deployed/scaled at a later date on different compute.

The [PyGrid library](https://github.com/OpenMined/PyGrid) serves as an API for the management and deployment of PySyft at scale. It also allows for you to extend PySyft for the purposes of Federated Learning on web, mobile, and edge devices using the following Syft worker libraries:

- [KotlinSyft](https://github.com/OpenMined/KotlinSyft) (Android)
- [SwiftSyft](https://github.com/OpenMined/SwiftSyft) (iOS)
- [syft.js](https://github.com/OpenMined/syft.js) (Javascript)
- PySyft (Python, you can use PySyft itself as one of these "FL worker libraries")

However, the Syft ecosystem only focuses on consistent object serialization/deserialization, core abstractions, and algorithm design/execution across these languages. These libraries alone will not connect you with data in the real world. The Syft ecosystem is supported by the Grid ecosystem, which focuses on the deployment, scalability, and other additional concerns around running real-world systems to compute over and process data (such as data compliance web applications).

- PySyft is the library that defines objects, abstractions, and algorithms.
- [PyGrid](https://github.com/OpenMined/PyGrid) is the platform which lets you deploy them within a real institution.
- [PyGrid Admin](https://github.com/OpenMined/pygrid-admin) is a UI which allows a data owner to manage their PyGrid deployment.

A more detailed explanation of PySyft can be found in the
[white paper on Arxiv](https://arxiv.org/abs/1811.04017).

PySyft has also been explained in videos on YouTube:

- [PriCon Sep 2020 Duet Demo](https://www.youtube.com/watch?v=DppXfA6C8L8&ab_channel=OpenMined)
- [Introduction to Privacy-Preserving AI using PySyft by @iamtrask](https://www.youtube.com/watch?v=NJBBE_SN90A)

<!---Documatic-section-fixed: top2-end--->
<!---Documatic-section-fixed: top3-start--->
### Duet

<a href="https://github.com/OpenMined/PySyft/tree/main/packages/syft/examples/duet"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/logo_duet.png" alt="PySyft" width="350"></a>

Duet is a peer-to-peer tool within PySyft that provides a research-friendly API for a Data Owner to privately expose their data, while a Data Scientist can access or manipulate the data on the owner's side through a zero-knowledge access control mechanism. It's designed to lower the barrier between research and privacy-preserving mechanisms, so that scientific progress can be made on data that is currently inaccessible or tightly controlled. **The main benefit of using Duet is that allows you to get started using PySyft, without needing to manage a full PyGrid deployment. It is the simplest path to using Syft, without needing to install anything (except Syft 😉).**

You can find all [Duet examples](https://github.com/OpenMined/PySyft/tree/main/packages/syft/examples/duet) in the `examples/duet` folder.
<!---Documatic-section-fixed: top3-end--->

<!---Documatic-section-group: helloworld-start--->
<!---Documatic-section-helloworld: setup-start--->

## Getting started

### Requirements

* Syft is compatible with Python 3.7 and above
*  Run `pip install -e .` in the directory to install
package locally
* Alternatively, install globally from pypi with

```
pip install syft
```

<!---Documatic-section-fixed: inplace-start--->
We support **Linux**, **MacOS** and **Windows** and the following Python and Torch versions.
Older versions may work, however we have stopped testing and supporting them.

| Py / Torch | 1.6 | 1.7 | 1.8 |
| ---------- | --- | --- | --- |
| 3.7        | ✅  | ✅  | ✅  |
| 3.8        | ✅  | ✅  | ✅  |
| 3.9        | ➖  | ✅  | ✅  |
<!---Documatic-section-fixed: inplace-start--->

## Getting started

Syft has the following CLI commands:

<!---Documatic-note: Functions could not be resolved - TODO summaries to be generated--->
* `syft-device` runs `syft.grid.example_nodes.device.run`
* `syft-domain` runs `syft.grid.example_nodes.domain.run`
* `syft-network` runs `syft.grid.example_nodes.network.run`
* `syft-proto` runs `syft.protobuf_setuptools.run`

Run a CLI command in a terminal to execute.


<!---Documatic-section-helloworld: setup-end--->

<!---Documatic-section-group: overview-start--->

## Code Overview

The Syft codebase has a 5-deep folder structure, with 463 python files in total.

```
syft
├── ast
├── core
├── grid
├── img
├── lib
└── proto
```

### Entrypoints

There are 17 source code objects
in top-level `__main__`/`__init__`.
They are broken down into the following modules:

* `syft.core` has 12 entrypoints
* `syft.grid` has 3 entrypoints
* `syft.experimental_flags` has 1 entrypoints
* `syft.registry` has 1 entrypoints

<!---Documatic-section-overview: classes-start--->

### Classes

Some important Syft classes include:

#### Node

```
syft.core.io.address.Address
└──syft.core.node.abstract.node.AbstractNode
   └──syft.core.node.common.node.Node
      ├──syft.core.node.device.device.Device
      ├──syft.core.node.domain.domain.Domain
      └──syft.core.node.vm.vm.VirtualMachine
```

<!---Documatic-section-code: f181ca-packages/syft/src/syft/core/node/common/node.py-102-131--->
Basic class for a syft node behavior, explicit purpose node will
inherit this class (e.g., Device, Domain, Network, and VirtualMachine).
Each node is identified by an id of type ID and a name of type string.

```python
class Node(AbstractNode):
    client_type = ClientT
    child_type_client_type = ClientT
    ChildT = TypeVar("ChildT", bound="Node")
    child_type = ChildT
    signing_key: Optional[SigningKey]
    verify_key: Optional[VerifyKey]

    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        TableBase: Any = None,
        db_engine: Any = None,
    )
```

#### Client

```
syft.core.common.node.client.Client
├──syft.core.node.device.client.DeviceClient
├──syft.core.node.domain.client.DomainClient
├──syft.core.node.network.client.NetworkClient
└──syft.core.node.vm.client.VirtualMachineClient 
```

<!---Documatic-section-code: f181ca-packages/syft/src/syft/core/node/common/client.py-50-70--->
Client is an incredibly powerful abstraction in Syft. We assume that,
no matter where a client is, it can figure out how to communicate with
the Node it is supposed to point to. If I send you a client I have
with all of the metadata in it, you should have all the information
you need to know to interact with a node (although you might not
have permissions - clients should not store private keys).

```python
@serializable()
class Client(AbstractNodeClient):
    def __init__(
        self,
        name: Optional[str],
        routes: List[Route],
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        version: Optional[str] = None,
    )
```

<!---Documatic-section-overview: classes-end--->


<!---Documatic-section-group: overview-end--->

<!---Documatic-section-group: concept-start--->
## Concepts
<!---Documatic-section-group: concept-end--->

<!---Documatic-section-group: helloworld-end--->

<!---Documatic-section-group: dev-start--->


## Developers
<!---Documatic-section-dev: setup-start--->
* Tests are present in `tests/` (uses pytest)
* No CI/CD config files are present in this repo

<!---Documatic-section-dev: setup-end--->

<!---Documatic-section-group: dev-end--->

<!---Documatic-section-fixed: bottom1-start--->
## Contributing

The guide for contributors can be found [here](https://github.com/OpenMined/PySyft/blob/main/packages/syft/CONTRIBUTING.md).
It covers all that you need to know to start contributing code to PySyft today.

Also, join the rapidly growing community of 12,000+ on [Slack](http://slack.openmined.org).
The Slack community is very friendly and great about quickly answering questions about the use and development of PySyft!

## Disclaimer

This software is in beta. Use at your own risk.

## A quick note about 0.2.x

The PySyft 0.2.x codebase is now in its own branch [here](https://github.com/OpenMined/PySyft/tree/syft_0.2.x), but OpenMined will not offer official support for this version range. We have compiled a list of [FAQs](https://github.com/OpenMined/PySyft/blob/main/packages/syft/docs/FAQ_0.2.x.md) relating to this version.\_

## Support

For support in using this library, please join the **#support** Slack channel. [Click here to join our Slack community!](https://slack.openmined.org)

## Organizational Contributions

We are very grateful for contributions to PySyft from the following organizations!

- [<img src="https://github.com/udacity/private-ai/blob/master/udacity-logo-vert-white.png?raw=true" alt="Udacity" width="160"/>](https://udacity.com/)
- [<img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/organizations/comind.png" alt="coMind" width="160" />](https://github.com/coMindOrg/federated-averaging-tutorials)
- [<img src="https://i.ibb.co/vYwcG9N/arkhn-logo.png" alt="Arkhn" width="160" />](http://ark.hn)
- [<img src="https://raw.githubusercontent.com/dropoutlabs/files/master/dropout-labs-logo-white-2500.png" alt="Dropout Labs" width="160"/>](https://dropoutlabs.com/)
- [<img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/organizations/genbu.png" alt="GENBU AI" width="160"/>](https://genbu.ai/)
- [<img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/organizations/bitdefender.png" alt="Bitdefender" width="160"/>](https://www.bitdefender.com/) |

## License

[Apache License 2.0](https://github.com/OpenMined/PySyft/blob/main/packages/syft/LICENSE)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FOpenMined%2FPySyft.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FOpenMined%2FPySyft?ref=badge_large)
<!---Documatic-section-fixed: bottom1-end--->
