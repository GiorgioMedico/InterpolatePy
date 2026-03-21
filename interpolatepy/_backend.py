"""Backend detection for optional C++ acceleration.

When the compiled C++ extension ``interpolatecpp_py`` is available inside the
package directory, the library transparently delegates to it for faster
evaluation.  Set the environment variable ``INTERPOLATEPY_NO_CPP=1`` to force
the pure-Python fallback even when the extension is present.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_cpp: ModuleType | None = None
HAS_CPP: bool = False


def get_cpp_module() -> ModuleType:
    """Return the C++ extension module, raising if unavailable."""
    if _cpp is None:
        msg = "C++ backend not available"
        raise ImportError(msg)
    return _cpp


if not os.environ.get("INTERPOLATEPY_NO_CPP"):
    try:
        # The .so lives next to this file inside the interpolatepy package.
        _cpp = importlib.import_module(".interpolatecpp_py", package=__package__)
        HAS_CPP = True
    except ImportError:
        pass
