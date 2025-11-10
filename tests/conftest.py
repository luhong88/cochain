import pytest

import cochain.datasets as datasets


@pytest.fixture
def two_tris_mesh():
    return datasets.load_two_tris_mesh()
