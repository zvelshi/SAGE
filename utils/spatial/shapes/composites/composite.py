# default
from __future__ import annotations

# ours
from utils.spatial.shapes._common import SCENE_SCALE


class Composite:
    """A named bundle of primitive shapes (and/or nested composites) that make up
    one physical component's 3-D representation -- a shape built from shapes.

    Subclasses implement :meth:`parts` (the moving pieces, rebuilt every frame
    from the component's current geometry) and optionally :meth:`static_parts`
    (built once, never re-placed -- e.g. reference guides). Both share the
    primitive contract (``to_3d`` / ``place``), so the generic build/update below
    works for any depth of nesting.
    """

    def parts(self) -> dict:
        raise NotImplementedError

    def static_parts(self) -> dict:
        return {}

    def to_3d(self, scene, scale: float = SCENE_SCALE) -> dict:
        out: dict = {}
        for name, spec in {**self.parts(), **self.static_parts()}.items():
            out[name] = self._build_spec(spec, scene, scale)
        return out

    def place(self, objs: dict, scale: float = SCENE_SCALE, restyle: bool = False) -> None:
        for name, spec in self.parts().items():
            if name in objs:
                self._place_spec(spec, objs[name], scale, restyle)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _build_spec(spec, scene, scale):
        if isinstance(spec, Composite):
            return spec.to_3d(scene, scale)
        if isinstance(spec, list):
            return [s.to_3d(scene, scale) for s in spec]
        return spec.to_3d(scene, scale)

    @staticmethod
    def _place_spec(spec, obj, scale, restyle):
        if isinstance(spec, Composite):
            spec.place(obj, scale, restyle)
        elif isinstance(spec, list):
            for s, o in zip(spec, obj):
                s.place(o, scale, restyle)
        else:
            spec.place(obj, scale, restyle)
