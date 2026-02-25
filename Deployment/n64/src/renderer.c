#include <libdragon.h>
#include <stdint.h>
#include <t3d/t3d.h>
#include <t3d/t3dmath.h>
#include <t3d/t3dmodel.h>

#include "engine.h"
#include "iron/system/input.h"
#include "types.h"
#include "utils.h"

#if ENGINE_ENABLE_PHYSICS
#include "oimo/physics.h"
#if ENGINE_ENABLE_PHYSICS_DEBUG
#include "oimo/debug/physics_debug.h"
#endif
#endif

static int frameIdx = 0;

void renderer_begin_frame(T3DViewport *viewport, ArmScene *scene) {
  	ArmCamera *cam = &scene->cameras[scene->active_camera_id];
  	t3d_viewport_set_projection(viewport, T3D_DEG_TO_RAD(cam->fov), cam->near, cam->far);
  	t3d_viewport_look_at(viewport, (T3DVec3 *)&cam->transform.loc, &cam->target, &(T3DVec3){{0.0f, 1.0f, 0.0f}});
}

void renderer_update_objects(ArmScene *scene) {
	for (uint16_t i = 0; i < scene->object_count; i++) {
		ArmObject *obj = &scene->objects[i];

		// Skip removed objects
		if (obj->is_removed) {
			continue;
		}

		if (obj->transform.dirty == 0) {
			continue;
		}

		// Safety check: skip objects with invalid transform values
		if (!transform_is_safe(obj->transform.loc.v, obj->transform.scale.v)) {
			obj->transform.loc = (T3DVec3){{0.0f, 0.0f, 0.0f}};
			obj->visible = false;
		}

		int mat_idx = obj->is_static ? 0 : frameIdx;
		t3d_mat4fp_from_srt(&obj->model_mat[mat_idx], obj->transform.scale.v, obj->transform.rot.v, obj->transform.loc.v);

		// Update cached world-space AABB for frustum culling.
		// bounds_min/max are pre-scaled to world coordinates (Blender units).
		// We must rotate the local AABB by the object's quaternion to get a
		// correct world-space AABB. Uses Arvo's method: iterate over the 3x3
		// rotation matrix elements and accumulate min/max contributions.
		{
			float qx = obj->transform.rot.v[0];
			float qy = obj->transform.rot.v[1];
			float qz = obj->transform.rot.v[2];
			float qw = obj->transform.rot.v[3];
			float xx = qx*qx, yy = qy*qy, zz = qz*qz;
			float xy = qx*qy, xz = qx*qz, yz = qy*qz;
			float wx = qw*qx, wy = qw*qy, wz = qw*qz;
			// Rotation matrix from quaternion
			float m[3][3] = {
				{ 1.0f - 2.0f*(yy+zz), 2.0f*(xy-wz),         2.0f*(xz+wy)         },
				{ 2.0f*(xy+wz),         1.0f - 2.0f*(xx+zz),  2.0f*(yz-wx)         },
				{ 2.0f*(xz-wy),         2.0f*(yz+wx),         1.0f - 2.0f*(xx+yy)  }
			};
			// Start from position, accumulate rotated AABB extents
			for (int i = 0; i < 3; i++) {
				obj->cached_world_aabb_min.v[i] = obj->transform.loc.v[i];
				obj->cached_world_aabb_max.v[i] = obj->transform.loc.v[i];
				for (int j = 0; j < 3; j++) {
					float a = m[i][j] * obj->bounds_min.v[j];
					float b = m[i][j] * obj->bounds_max.v[j];
					if (a < b) {
						obj->cached_world_aabb_min.v[i] += a;
						obj->cached_world_aabb_max.v[i] += b;
					} else {
						obj->cached_world_aabb_min.v[i] += b;
						obj->cached_world_aabb_max.v[i] += a;
					}
				}
			}
		}

		obj->transform.dirty--;
	}
}

void renderer_draw_scene(T3DViewport *viewport, ArmScene *scene) {
	frameIdx = (frameIdx + 1) % FB_COUNT;
	renderer_update_objects(scene);

	surface_t *fb = display_get();
	rdpq_attach(fb, display_get_zbuf());
	t3d_frame_start();
	t3d_viewport_attach(viewport);

	t3d_screen_clear_color(RGBA32(scene->world.clear_color[0], scene->world.clear_color[1], scene->world.clear_color[2], scene->world.clear_color[3]));
	t3d_screen_clear_depth();

	t3d_state_set_drawflags(T3D_FLAG_DEPTH | T3D_FLAG_CULL_BACK);

	t3d_light_set_ambient(scene->world.ambient_color);
	t3d_light_set_count(scene->light_count);

	for (uint8_t i = 0; i < scene->light_count; i++) {
		ArmLight *l = &scene->lights[i];
		t3d_light_set_directional(i, l->color, &l->dir);
	}

	const T3DFrustum *frustum = &viewport->viewFrustum;
	int visible_count = 0;

	// Render dynamic objects with per-frame frustum culling
	t3d_matrix_push_pos(1);
	for (uint16_t i = 0; i < scene->object_count; i++) {
		ArmObject *obj = &scene->objects[i];
		if (!obj->visible) {
			continue;
		}

		// Use cached world AABB - updated only when transform.dirty > 0
		if (!t3d_frustum_vs_aabb(frustum, &obj->cached_world_aabb_min, &obj->cached_world_aabb_max)) {
			continue;
		}

		// Skip if display list not loaded (model failed to load)
		if (!obj->dpl) {
			continue;
		}

		visible_count++;
		int mat_idx = obj->is_static ? 0 : frameIdx;
		t3d_matrix_set(&obj->model_mat[mat_idx], true);
		rspq_block_run(obj->dpl);
	}
	t3d_matrix_pop(1);

#ifdef ARM_DEBUG_HUD
	rdpq_sync_pipe();
	rdpq_text_printf(NULL, FONT_BUILTIN_DEBUG_MONO, 200, 220, "FPS: %.2f", display_get_fps());
	rdpq_text_printf(NULL, FONT_BUILTIN_DEBUG_MONO, 200, 230, "Obj: %d/%d", visible_count, scene->object_count);

	// Input debug
	rdpq_text_printf(NULL, FONT_BUILTIN_DEBUG_MONO, 10, 10, "Stick: %.2f, %.2f", input_stick_x(), input_stick_y());
	rdpq_text_printf(NULL, FONT_BUILTIN_DEBUG_MONO, 10, 20, "A:%d B:%d Z:%d Start:%d", input_down(N64_BTN_A), input_down(N64_BTN_B), input_down(N64_BTN_Z), input_down(N64_BTN_START));
	rdpq_text_printf(NULL, FONT_BUILTIN_DEBUG_MONO, 10, 30, "D: %d%d%d%d  C: %d%d%d%d", input_down(N64_BTN_DUP), input_down(N64_BTN_DDOWN), input_down(N64_BTN_DLEFT), input_down(N64_BTN_DRIGHT), input_down(N64_BTN_CUP), input_down(N64_BTN_CDOWN), input_down(N64_BTN_CLEFT), input_down(N64_BTN_CRIGHT));
	rdpq_text_printf(NULL, FONT_BUILTIN_DEBUG_MONO, 10, 40, "L:%d R:%d", input_down(N64_BTN_L), input_down(N64_BTN_R));
#endif
}

void renderer_end_frame(T3DViewport *viewport) {
#if ENGINE_ENABLE_PHYSICS && ENGINE_ENABLE_PHYSICS_DEBUG
	// Draw physics debug using RDP hardware (while still attached)
	physics_debug_draw(viewport, physics_get_world());
#endif
	rdpq_detach_show();
}
