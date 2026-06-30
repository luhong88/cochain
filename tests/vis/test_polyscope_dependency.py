import pytest

from cochain.vis import polyscope


def test_polyscope_import_error(icosphere_mesh, monkeypatch):
    monkeypatch.setattr(polyscope, "_HAS_POLYSCOPE", False)

    with pytest.raises(ImportError):
        polyscope.PolyscopeViewer(name="test", mesh=icosphere_mesh)
