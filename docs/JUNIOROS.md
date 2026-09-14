# JuniorOS / JuniorOSai

Home is the overlay. Not a new distro ISO this week.

## Surfaces (`web3node/os_route.py`)
| task hint | surface | watts | LLM port |
|-----------|---------|-------|----------|
| frameforge / ff2d / canvas | frameforge2d | T0 ~45W | JuniorAstra |
| ue5 / unreal / spark | ue5 | T1 ≥80W | JuniorAstra |
| flagstaff / fieldcore / stonefield | fieldcore-intent | T4 mobile | JuniorBitNetFieldCore |
| juniorosai / overlay | juniorosai | T0 | JuniorAstra |
| dxf / cad / omega | blender-cmd | T0 | JuniorBitNetDraft |
| else | home-clock | T0 | JuniorAstra |

`launch` is always false here. UE5 does not boot from this module.

## FrameForge2D
Hard-split canvas/itch kernel. 60 Hz. Local trit sidecar. No fetch in the tick.
https://github.com/cloudcover95/FrameForge2D
UE5 / web3d-grok lives in FrameForge Unreal tree — learning slice, not this kernel.

## FieldCore + Flagstaff
Crowd/field trit scorer. Flagstaff 6-vote (terraform_ok, covenant, area, junior_port, finite_y, no_bind) stays in JuniorLLM. Home only routes the surface.

## Git identity
Commits on this connector are `cloudcover95` (`nico juniorcloudllc`). There is no separate bot GitHub user. Automations use the same account.
