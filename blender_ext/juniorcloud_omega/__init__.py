"""JuniorCloud Omega — import ~/.juniorhome/omega/patch.obj"""
from __future__ import annotations

bl_info = {
    "name": "JuniorCloud Omega",
    "author": "JuniorCloud LLC",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > JuniorHome",
    "description": "Local OBJ from Home omega harness. UE5 off.",
    "category": "Import-Export",
}

from pathlib import Path

import bpy

DEFAULT_OBJ = Path.home() / ".juniorhome" / "omega" / "patch.obj"


def _import_obj(path: Path) -> str:
    if not path.is_file():
        return "missing"
    op = getattr(bpy.ops.wm, "obj_import", None) or getattr(bpy.ops, "import_scene", None)
    try:
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
        return "ok"
    except Exception as exc:
        return str(exc)[:120]


class JUNIOR_OT_omega_import(bpy.types.Operator):
    bl_idname = "junior.omega_import"
    bl_label = "Import Omega OBJ"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        st = _import_obj(DEFAULT_OBJ)
        self.report({"INFO"} if st == "ok" else {"WARNING"}, st)
        return {"FINISHED"} if st == "ok" else {"CANCELLED"}


class JUNIOR_PT_omega(bpy.types.Panel):
    bl_label = "JuniorHome"
    bl_idname = "JUNIOR_PT_omega"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "JuniorHome"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="terrain-obj  |  ue5_launch=false")
        col.label(text=str(DEFAULT_OBJ))
        col.operator("junior.omega_import")


CLASSES = (JUNIOR_OT_omega_import, JUNIOR_PT_omega)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
