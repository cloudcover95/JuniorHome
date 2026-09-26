"""JuniorCloud Omega v0.2 — import/export local OBJ, dash scale."""
from __future__ import annotations

bl_info = {
    "name": "JuniorCloud Omega",
    "author": "JuniorCloud LLC",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > JuniorHome",
    "description": "Local OBJ harness. UE5 off.",
    "category": "Import-Export",
}

from pathlib import Path

import bpy
from bpy.props import FloatProperty

DEFAULT_OBJ = Path.home() / ".juniorhome" / "omega" / "patch.obj"
EXPORT_OBJ = Path.home() / ".juniorhome" / "omega" / "from_blender.obj"


def _import_obj(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
        sc = float(getattr(bpy.context.scene, "junior_omega_scale", 1.0) or 1.0)
        sc = min(max(sc, 0.35), 1.0)
        for obj in bpy.context.selected_objects:
            obj.scale = (sc, sc, sc)
        return "ok"
    except Exception as exc:
        return str(exc)[:120]


def _export_obj(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(filepath=str(path))
        else:
            bpy.ops.export_scene.obj(filepath=str(path))
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


class JUNIOR_OT_omega_export(bpy.types.Operator):
    bl_idname = "junior.omega_export"
    bl_label = "Export to Home omega"
    bl_options = {"REGISTER"}

    def execute(self, context):
        st = _export_obj(EXPORT_OBJ)
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
        col.prop(context.scene, "junior_omega_scale", text="scale")
        col.operator("junior.omega_import")
        col.operator("junior.omega_export")


CLASSES = (JUNIOR_OT_omega_import, JUNIOR_OT_omega_export, JUNIOR_PT_omega)


def register():
    bpy.types.Scene.junior_omega_scale = FloatProperty(
        name="scale", default=1.0, min=0.35, max=1.0, step=5
    )
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
    if hasattr(bpy.types.Scene, "junior_omega_scale"):
        del bpy.types.Scene.junior_omega_scale
