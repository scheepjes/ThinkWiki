#!/usr/bin/env bash
# Build the libzim C-wrapper shared library.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -fPIC -shared}"
LIBS="$(pkg-config --libs libzim 2>/dev/null || echo -lzim)"
g++ ${CXXFLAGS} -o "${HERE}/libzim_wrapper.so" "${HERE}/zim_wrapper.cpp" ${LIBS}
echo "built: ${HERE}/libzim_wrapper.so"
