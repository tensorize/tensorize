# ==============================================================================
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from setuptools import setup
from setuptools import find_packages

_VERSION = '0.11.0'

REQUIRED_PACKAGES = [
    'numpy >= 1.11.0',
    'six >= 1.10.0',
    'protobuf == 3.1.0',
]

setup(
      version=_VERSION,
      description='Deep Learning for Python',
      author='Fabrizio Milo',
      author_email='mistobaan@gmail.com',
      url='https://github.com/Mistobaan/',
      install_requires=REQUIRED_PACKAGES,
      extras_require={},
      packages=find_packages(),
      # PyPI package information.
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
      license='Apache 2.0',
      keywords='tensorize tensorflow tensor machine learning',
)
