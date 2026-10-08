import copy
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from comfyui_turing_utils.canvas.store import Project, contained, local_source
from comfyui_turing_utils.canvas.graph import validate_canvas, material_inputs, signature, plan
from comfyui_turing_utils.canvas.compiler import compile_task
from comfyui_turing_utils.canvas.routes import guard_prompt
from comfyui_turing_utils.canvas.media import read_material
from comfyui_turing_utils.canvas.execution import PrepareH3, Publish, SigmaRefiner
from comfyui_turing_utils.canvas.nodes import PUBLIC_NODES


def graph():
    return {"nodes": [
        {"id": "1", "type": "TuringCanvasSettings", "values": {"work_directory": "project", "cache_directory": "canvas/project",
            "dit": "dit.safetensors", "clip": "clip.safetensors", "video_vae": "vae.safetensors", "audio_vae": "audio.safetensors", "loras": "[]"}},
        {"id": "2", "type": "TuringCanvasImage", "values": {"asset_id": ""}},
        {"id": "3", "type": "TuringCanvasH3", "values": {"mode": "reference", "width": 64, "height": 64,
            "frames": 5, "user_prompt": "user", "model_prompt": "", "seed": 0}, "inputs": {"images.image_0": {"node": "2", "slot": 0}}},
        {"id": "4", "type": "TuringCanvasH3", "values": {"mode": "edit", "width": 64, "height": 64,
            "frames": 5, "user_prompt": "edit", "seed": 0}, "inputs": {"target": {"node": "3", "slot": 0}}},
    ]}


class CanvasTest(unittest.TestCase):
    def test_sigma_refiner_defaults_and_no_tail(self):
        original = torch.tensor([1., .9, .6, .3, 0.], dtype=torch.float64)
        refined, = SigmaRefiner().refine(original)
        torch.testing.assert_close(refined, torch.tensor([1., .9, .6, .45, .15, 0.], dtype=torch.float64))
        self.assertEqual(refined.dtype, original.dtype)
        self.assertEqual(refined.device, original.device)
        for values in ([1., .8, 0.], [0.], []):
            sigmas = torch.tensor(values)
            self.assertIs(SigmaRefiner().refine(sigmas)[0], sigmas)
        low = SigmaRefiner().refine(torch.tensor([.2, .1, 0.]))[0]
        self.assertEqual(len(low), 4)
        self.assertTrue(torch.all(low[:-1] >= low[1:]))

    def test_sampler_refiner_and_disabled_loras(self):
        settings = self.graph["nodes"][0]["values"]
        settings.update(sampler_name="heun", scheduler="karras", steps=4,
                        loras=json.dumps([{"name": "disabled", "on": False}, {"name": "zero", "strength": 0}, {"name": "enabled", "strength": .5}]))
        prompt = compile_task(self.graph, "3", self.project)
        self.assertEqual(prompt["sampler"]["inputs"]["sampler_name"], "heun")
        self.assertEqual(prompt["sigmas"]["inputs"]["scheduler"], "karras")
        self.assertEqual(prompt["sigmas"]["inputs"]["steps"], 4)
        self.assertEqual(prompt["sample"]["inputs"]["sigmas"], ["refiner", 0])
        self.assertNotIn("lora_0", prompt)
        self.assertNotIn("lora_1", prompt)
        self.assertIn("lora_2", prompt)
        settings["refiner"] = False
        prompt = compile_task(self.graph, "3", self.project)
        self.assertEqual(prompt["sample"]["inputs"]["sigmas"], ["sigmas", 0])
        self.graph["nodes"][2]["values"]["denoise"] = 0
        settings["refiner"] = True
        self.assertNotIn("refiner", compile_task(self.graph, "3", self.project))

    def test_generated_selection_is_consumed_without_sampling(self):
        asset = self.asset("video")
        self.project.publish("3", asset, "old", {})
        self.graph["nodes"][2]["values"].update(start_seconds=1.0, end_seconds=2.5)
        nodes, _ = validate_canvas(self.graph)
        refs = material_inputs(nodes["4"], nodes, self.project.state(), self.project)
        self.assertEqual(refs["target"]["start"], 1.)
        self.assertEqual(refs["target"]["duration"], 1.5)
        self.assertEqual(refs["target"]["modality"], "av")
        nodes["4"]["inputs"] = {"audios.audio_0": {"node": "3", "slot": 0}}
        refs = material_inputs(nodes["4"], nodes, self.project.state(), self.project)
        self.assertEqual(refs["audios.audio_0"]["modality"], "audio")

    def test_single_container_ports_and_sampling_controls(self):
        for name in ("TuringCanvasH3", "TuringCanvasVideo"):
            self.assertEqual(len(PUBLIC_NODES[name].define_schema().outputs), 1)
        inputs = {i.id for i in PUBLIC_NODES["TuringCanvasH3Settings"].define_schema().inputs}
        self.assertTrue({"sampler_name", "scheduler", "steps", "refiner"} <= inputs)
        self.assertNotIn("sigmas", inputs)
        self.assertNotIn("import_mode", {i.id for i in PUBLIC_NODES["TuringCanvasSettings"].define_schema().inputs})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = mock.patch("folder_paths.get_output_directory", return_value=str(self.root))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.project = Project("project")
        self.graph = graph()
        self.image = self.asset("image")
        self.graph["nodes"][1]["values"]["asset_id"] = self.image

    def asset(self, kind):
        aid, path = self.project.reserve(".png" if kind == "image" else ".mp4")
        if kind == "image":
            Image.new("RGB", (16, 16), (128, 64, 32)).save(path)
        else:
            path.touch()
        self.project.register(aid, path, kind, "test")
        return aid

    def publish(self, key, asset):
        nodes, _ = validate_canvas(self.graph)
        refs = material_inputs(nodes[key], nodes, self.project.state(), self.project)
        self.project.publish(key, asset, signature(refs), {"materials": refs})

    def test_global_settings_and_prompts_do_not_invalidate_material(self):
        self.publish("3", self.asset("video"))
        self.publish("4", self.asset("video"))
        self.graph["nodes"][0]["values"].update(dit="different", loras="changed", steps=20)
        self.graph["nodes"][2]["values"].update(model_prompt="new", user_prompt="new", width=720)
        self.assertEqual([p["status"] for p in plan(self.graph, self.project)], ["current", "current"])

    def test_new_material_propagates_only_after_publication(self):
        self.publish("3", self.asset("video"))
        self.publish("4", self.asset("video"))
        self.graph["nodes"][1]["values"]["asset_id"] = self.asset("image")
        self.assertEqual([p["status"] for p in plan(self.graph, self.project)], ["changed", "upstream"])
        self.publish("3", self.asset("video"))
        self.assertEqual([p["status"] for p in plan(self.graph, self.project)], ["current", "changed"])

    def test_pinned_result_does_not_follow_republication(self):
        first = self.asset("video")
        self.publish("3", first)
        self.graph["nodes"][3]["inputs"]["target"]["asset"] = first
        self.publish("4", self.asset("video"))
        self.publish("3", self.asset("video"))
        self.assertEqual(plan(self.graph, self.project)[1]["status"], "current")

    def test_missing_inputs_cycles_and_mixed_nodes(self):
        self.graph["nodes"][1]["values"]["asset_id"] = ""
        self.assertEqual(plan(self.graph, self.project)[0]["status"], "blocked")
        self.graph["nodes"].append({"id": "5", "type": "LoadImage"})
        with self.assertRaisesRegex(ValueError, "ordinary"):
            validate_canvas(self.graph)
        self.graph["nodes"].pop()
        self.graph["nodes"][2]["inputs"]["target"] = {"node": "4"}
        with self.assertRaisesRegex(ValueError, "acyclic"):
            plan(self.graph, self.project)

    def test_queue_guard_is_fail_closed_even_with_empty_prompt(self):
        ordinary = {"prompt": {"1": {"class_type": "LoadImage"}}}
        self.assertIs(guard_prompt(ordinary), ordinary)
        mixed = {"prompt": {}, "extra_data": {"extra_pnginfo": {"workflow": self.graph}}}
        self.assertIn("canvas_error", guard_prompt(mixed)["prompt"])

    def test_private_ports_and_public_cards_cannot_execute(self):
        self.assertNotIn("TuringCanvasMask", PUBLIC_NODES)
        self.assertNotIn("sec_model", [i.id for i in PUBLIC_NODES["TuringCanvasSettings"].GET_SCHEMA().inputs])
        for cls in PUBLIC_NODES.values():
            schema = cls.GET_SCHEMA()
            self.assertEqual(schema.category, "Turing Utils/Canvas")
            with self.assertRaisesRegex(ValueError, "card buttons"):
                cls.execute()

    def test_path_traversal_and_symlink_escape(self):
        with self.assertRaises(ValueError):
            Project("../escape")
        with self.assertRaises(ValueError):
            self.project.asset("../../etc/passwd")
        (self.root / "outside").mkdir()
        (self.project.root / "escape").symlink_to(self.root / "outside", target_is_directory=True)
        with self.assertRaises(ValueError):
            contained(self.project.root, "escape/file")

    def test_image_decode_and_persistent_history(self):
        images, audio, mask = read_material(self.project, {"asset": self.image}, 32, 32)
        self.assertEqual(tuple(images.shape), (1, 32, 32, 3))
        self.assertIsNone(audio)
        self.assertEqual(tuple(mask.shape), (1, 32, 32))
        first, second = self.asset("video"), self.asset("video")
        self.publish("3", first)
        self.publish("3", second)
        reopened = Project("project")
        self.assertEqual(len(reopened.state()["tasks"]["3"]["history"]), 2)
        reopened.select("3", first)
        self.assertEqual(reopened.state()["tasks"]["3"]["asset"], first)

    def test_compile_manual_regeneration_keeps_encoder_inputs_stable(self):
        one = compile_task(self.graph, "3", self.project)
        two = compile_task(self.graph, "3", self.project)
        self.assertEqual(one["semantic"], two["semantic"])
        self.assertEqual(one["read_images.image_0"], two["read_images.image_0"])
        self.assertNotEqual(one["run_noise"], two["run_noise"])
        self.assertEqual(one["semantic"]["inputs"]["prompt"], "user")
        self.graph["nodes"][2]["values"]["model_prompt"] = "model"
        changed = compile_task(self.graph, "3", self.project)
        self.assertEqual(changed["semantic"]["inputs"]["prompt"], "model")
        self.assertEqual(one["prepare"], changed["prepare"])
        self.assertFalse(any(n["class_type"] == "TuringUtilsMultimodalPromptChat" for n in one.values()))

    def test_reference_prepare_uses_native_h3_av_shapes(self):
        values = self.graph["nodes"][2]["values"]
        latent, count, prefix, audio = PrepareH3().prepare(json.dumps(values), None, None)
        video, sound = latent["samples"].unbind()
        self.assertEqual(tuple(video.shape[:3]), (1, 24, 2))
        self.assertEqual(tuple(sound.shape[:3]), (1, 32, 2))
        self.assertEqual((count, prefix, audio), (5, 0, None))
        latent, count, _, _ = PrepareH3().prepare(json.dumps({**values, "frames": 7}), None, None)
        self.assertEqual(count, 7)
        self.assertGreater(latent["samples"].unbind()[0].shape[2], 2)

    def test_denoise_uses_basic_scheduler_and_zero_prunes_dit(self):
        self.graph["nodes"][2]["values"]["denoise"] = 0.35
        prompt = compile_task(self.graph, "3", self.project)
        self.assertEqual(prompt["sigmas"]["inputs"]["denoise"], 0.35)
        self.graph["nodes"][2]["values"]["denoise"] = 0
        prompt = compile_task(self.graph, "3", self.project)
        self.assertNotIn("sample", prompt)
        self.assertNotIn("dit", prompt)
        self.assertNotIn("clip", prompt)
        self.assertEqual(prompt["separate"]["inputs"]["av_latent"], ["prepare", 0])

    def test_sol_expanded_settings_reach_strategy(self):
        self.graph["nodes"][0]["values"].update(sol="enabled", **{"sol.routing_threshold": 0.7, "sol.dense_prefix_steps": 2})
        prompt = compile_task(self.graph, "3", self.project)
        settings = {k.removeprefix("strategy."): v for k, v in prompt["strategy"]["inputs"].items()}
        self.assertEqual(settings["routing_threshold"], 0.7)
        self.assertEqual(settings["dense_prefix_steps"], 2)
        self.graph["nodes"][0]["values"]["sol"] = "disabled"
        self.assertNotIn("sol", compile_task(self.graph, "3", self.project))

    def test_filename_history_and_collision_are_project_local(self):
        source = self.root / "portrait.png"
        Image.new("RGB", (8, 8)).save(source)
        first = self.project.copy(source, "image")
        second = self.project.copy(source, "image")
        self.assertEqual(first["id"], "materials/images/portrait.png")
        self.assertEqual(second["id"], "materials/images/portrait_000001.png")
        self.assertEqual(len(self.project.history("image")), 3)
        aid, path = self.project.reserve(".mp4", prefix="shot_a")
        self.project.register(aid, path, "video", "shot")
        self.assertEqual(aid, "generations/shot_a/shot_a_000001.mp4")
        self.assertEqual(len(self.project.history("video", "shot_a")), 1)
        self.assertEqual(self.project.history("video", "shot_b"), [])
        with self.assertRaises(ValueError):
            self.project.reserve(".mp4", prefix="../escape")

    def test_dynamic_reference_order_and_public_controls(self):
        node = self.graph["nodes"][2]
        node["inputs"] = {"images.image_2": {"node": "2"}, "images.image_0": {"node": "2"}}
        prompt = compile_task(self.graph, "3", self.project)
        self.assertEqual(prompt["image_ref"]["inputs"]["images.image_0"], ["read_images.image_0", 0])
        fields = [i.id for i in PUBLIC_NODES["TuringCanvasH3"].GET_SCHEMA().inputs]
        self.assertEqual(fields[:7], ["first_frame", "last_frame", "images", "videos", "audios", "prefix", "target"])
        for field in ("mode", "seed", "random_seed", "mask"):
            self.assertNotIn(field, fields)

    def test_image_maximum_megapixels(self):
        images, _, _ = read_material(self.project, {"asset": self.image, "max_megapixels": 0.001}, 100, 100)
        self.assertLessEqual(images.shape[1] * images.shape[2], 0.001 * 1024 * 1024)
        images, _, _ = read_material(self.project,
            {"asset": self.image, "max_megapixels": 0.01, "spatial_multiple": 32}, 832, 480)
        self.assertEqual(images.shape[1] % 32, 0)
        self.assertEqual(images.shape[2] % 32, 0)
        self.assertLessEqual(images.shape[1] * images.shape[2], 0.01 * 1024 * 1024)

    def test_empty_directory_cleanup_never_removes_files_or_links(self):
        from comfyui_turing_utils.canvas.store import remove_empty_directory
        (self.root / "empty/a/b").mkdir(parents=True)
        remove_empty_directory("empty")
        self.assertFalse((self.root / "empty").exists())
        (self.root / "occupied/a").mkdir(parents=True)
        (self.root / "occupied/file").write_text("keep")
        with self.assertRaises(ValueError):
            remove_empty_directory("occupied")
        self.assertTrue((self.root / "occupied/a").exists())
        (self.root / "alias").symlink_to(self.root / "occupied", target_is_directory=True)
        with self.assertRaises(ValueError):
            remove_empty_directory("alias")
        with self.assertRaises(ValueError):
            remove_empty_directory(".")

    def test_read_only_project_does_not_create_directory(self):
        project = Project("not-created", create=False)
        self.assertEqual(project.state(), {"tasks": {}})
        self.assertFalse(project.root.exists())

    def test_duration_alignment_and_future_schema(self):
        self.graph["nodes"][2]["values"].update(duration=2.5, width=835, height=479)
        prompt = compile_task(self.graph, "3", self.project)
        values = json.loads(prompt["prepare"]["inputs"]["settings"])
        self.assertEqual((values["frames"], values["width"], values["height"]), (60, 832, 480))
        with self.assertRaises(ValueError):
            validate_canvas({**self.graph, "schema_version": 999})

    def test_canvas_settings_reuse_attention_schema(self):
        from comfyui_turing_utils.canvas.nodes import CanvasH3Settings, CanvasSettings, CanvasVideo
        strategy = next(i for i in CanvasH3Settings.define_schema().inputs if i.id == "strategy")
        self.assertEqual(len(strategy.options), 4)
        inputs = {i.id: i for i in CanvasSettings.define_schema().inputs}
        self.assertEqual(inputs["max_megapixels"].default, 4)
        names = [i.id for i in CanvasVideo.define_schema().inputs]
        self.assertIn("include_audio", names)
        self.assertIn("end_seconds", names)
        for removed in ("select_every_nth", "skip_first_frames", "frame_load_cap", "custom_width"):
            self.assertNotIn(removed, names)

    def test_video_reference_does_not_implicitly_connect_audio(self):
        self.graph["nodes"][1].update(type="TuringCanvasVideo", values={"asset_id": self.asset("video"),
            "include_audio": True, "start_seconds": 1, "end_seconds": 3, "max_megapixels": .5})
        self.graph["nodes"][2]["inputs"] = {"videos.video_0": {"node": "2", "slot": 0}, "audios.audio_0": {"node": "2", "slot": 1}}
        prompt = compile_task(self.graph, "3", self.project)
        self.assertFalse(any(k.startswith("video_audios") for k in prompt["video_ref"]["inputs"]))
        ref = json.loads(prompt["read_videos.video_0"]["inputs"]["reference"])
        self.assertEqual((ref["modality"], ref["duration"], ref["max_megapixels"]), ("video", 2, .5))
        audio = json.loads(prompt["read_audios.audio_0"]["inputs"]["reference"])
        self.assertEqual(audio["modality"], "audio")
        self.graph["nodes"][1]["values"]["include_audio"] = False
        with self.assertRaisesRegex(ValueError, "disabled"):
            compile_task(self.graph, "3", self.project)

    def test_video_modality_decode_and_disabled_audio(self):
        sound = {"waveform": torch.zeros(1, 2, 44100), "sample_rate": 44100}
        result = Publish().publish("project", "3", "sig", json.dumps({"materials": {}}), "run",
            images=torch.rand(24, 32, 32, 3), audio=sound, length=24)
        ref = {"asset": result["result"][0], "start": .25, "duration": .5}
        pictures, audio, _ = read_material(self.project, {**ref, "modality": "video"})
        self.assertEqual(len(pictures), 12)
        self.assertIsNone(audio)
        pictures, audio, _ = read_material(self.project, {**ref, "modality": "audio"})
        self.assertIsNone(pictures)
        self.assertEqual(audio["waveform"].shape[-1], 22050)
        pictures, audio, _ = read_material(self.project, {**ref, "modality": "av", "include_audio": False})
        self.assertEqual(len(pictures), 12)
        self.assertIsNone(audio)
        with self.assertRaisesRegex(ValueError, "disabled"):
            read_material(self.project, {**ref, "modality": "audio", "include_audio": False})

    def test_selecting_history_does_not_replace_generation_checkpoint(self):
        first, second = self.asset("video"), self.asset("video")
        self.project.publish("3", first, "a", {})
        self.project.publish("3", second, "b", {})
        self.project.publish("3", first, "a", {}, generated=False)
        state = self.project.state()["tasks"]["3"]
        self.assertEqual(state["asset"], first)
        self.assertEqual(state["last_generated"], second)

    def test_target_and_prefix_audio_masks_and_padding(self):
        from comfy_api.latest import io
        from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
        def encode(pixels, vae):
            latent = EmptyMiniMaxH3LatentAV.execute(64, 64, len(pixels)).result[0]
            return ({"samples": latent["samples"].unbind()[0]},)
        def encode_audio(vae, audio):
            return io.NodeOutput({"samples": torch.zeros(1, 32, 2, 200)})
        settings = json.dumps({"width": 64, "height": 64, "frames": 7})
        sound = {"waveform": torch.ones(1, 2, 44100), "sample_rate": 44100}
        with mock.patch("comfyui_turing_utils.canvas.execution.MiniMaxH3VideoVAEEncode.encode", side_effect=encode), mock.patch(
                "comfyui_turing_utils.canvas.execution.VAEEncodeAudio.execute", side_effect=encode_audio):
            latent, count, prefix, original = PrepareH3().prepare(settings, None, None,
                prefix_images=torch.rand(5, 64, 64, 3), prefix_audio=sound, images=torch.rand(7, 64, 64, 3))
            self.assertEqual((count, prefix), (12, 5))
            self.assertIsNone(original)
            vm, am = latent["noise_mask"].unbind()
            self.assertTrue(torch.all(am[..., :45] == 0))
            self.assertTrue(torch.all(am[..., 46:] == 1))
            latent, _, _, original = PrepareH3().prepare(settings, None, None,
                prefix_images=torch.rand(5, 64, 64, 3), prefix_audio=sound,
                images=torch.rand(7, 64, 64, 3), audio=sound)
            self.assertTrue(torch.all(latent["noise_mask"].unbind()[1] == 0))
            self.assertEqual(original["waveform"].shape[-1], 22051)

    def test_video_frame_selection_keeps_audio_timeline(self):
        rate = 44100
        result = Publish().publish("project", "3", "sig", json.dumps({"materials": {}}), "selection",
            images=torch.rand(24, 32, 32, 3), audio={"waveform": torch.ones(1, 2, rate) * 0.1, "sample_rate": rate}, length=24)
        images, audio, _ = read_material(self.project, {"asset": result["result"][0],
            "skip_first_frames": 6, "frame_load_cap": 6, "select_every_nth": 2, "force_rate": 24})
        self.assertEqual(len(images), 12)
        self.assertEqual(audio["waveform"].shape[-1], rate // 2)

    def test_compiled_graph_passes_real_comfy_validation_without_loading_models(self):
        import execution
        import nodes
        from comfy_api.latest import io
        from comfy_extras import nodes_custom_sampler, nodes_minimax_h3, nodes_lt, nodes_audio
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS
        mapping = dict(NODE_CLASS_MAPPINGS)
        for module in (nodes_custom_sampler, nodes_minimax_h3, nodes_lt, nodes_audio):
            for value in vars(module).values():
                if isinstance(value, type) and issubclass(value, io.ComfyNode) and value is not io.ComfyNode:
                    try:
                        mapping[value.GET_SCHEMA().node_id] = value
                    except NotImplementedError:
                        pass
        for label in ("MODEL", "CLIP", "VAE"):
            mapping["TestCanvas" + label] = type("TestLoader", (), {
                "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
                "RETURN_TYPES": (label,), "FUNCTION": "load"})
        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, mapping):
            for sol in (False, True):
                self.graph["nodes"][0]["values"]["sol"] = sol
                prompt = compile_task(self.graph, "3", self.project)
                for key, label in (("dit", "MODEL"), ("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE")):
                    prompt[key] = {"class_type": "TestCanvas" + label, "inputs": {}}
                valid = asyncio.run(execution.validate_prompt("canvas-test", prompt, ["publish"]))
                self.assertTrue(valid[0], valid)

    def test_publication_saves_trimmed_video_and_metadata(self):
        result = Publish().publish("project", "3", "sig", json.dumps({"materials": {}}), "run",
            images=torch.rand(5, 32, 32, 3), length=5, prefix=1)
        asset = result["result"][0]
        images, _, _ = read_material(self.project, {"asset": asset})
        self.assertEqual(len(images), 4)
        self.assertEqual(self.project.state()["tasks"]["3"]["asset"], asset)

    def test_video_audio_interval_preserves_duration(self):
        rate = 44100
        result = Publish().publish("project", "3", "sig", json.dumps({"materials": {}}), "av",
            images=torch.rand(24, 32, 32, 3), audio={"waveform": torch.ones(1, 2, rate) * 0.1, "sample_rate": rate}, length=24)
        images, audio, _ = read_material(self.project, {"asset": result["result"][0], "start": 0.25, "duration": 0.5})
        self.assertEqual(len(images), 12)
        self.assertEqual(audio["waveform"].shape[-1], rate // 2)
        self.assertGreater(float(audio["waveform"].abs().mean()), 0.05)
        self.graph["nodes"][2]["values"].update(start_seconds=.25, end_seconds=.75)
        nodes, _ = validate_canvas(self.graph)
        selection = material_inputs(nodes["4"], nodes, self.project.state(), self.project)["target"]
        selected_images, selected_audio, _ = read_material(self.project, selection)
        self.assertEqual(len(selected_images), 12)
        self.assertEqual(selected_audio["waveform"].shape[-1], rate // 2)
        self.assertEqual(self.project.asset(result["result"][0])["metadata"]["duration"], 1.)

    def test_local_copy_and_failed_import_preserve_original(self):
        source = self.root / "source.png"
        Image.new("RGB", (8, 8)).save(source)
        with mock.patch("folder_paths.get_input_directory", return_value=str(self.root)):
            item = self.project.copy(local_source("source.png"), "image")
            self.assertTrue(source.exists())
            self.assertEqual(item["metadata"]["width"], 8)
            with self.assertRaises(ValueError):
                local_source("../escape")
        before = set(self.project.assets.iterdir())
        with self.assertRaises(ValueError):
            self.project.copy(source, "video")
        self.assertEqual(before, set(self.project.assets.iterdir()))

    def test_edit_and_extension_prepare_masks_without_model_execution(self):
        from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
        def encode(pixels, vae):
            latent = EmptyMiniMaxH3LatentAV.execute(64, 64, len(pixels)).result[0]
            return ({"samples": latent["samples"].unbind()[0]},)
        with mock.patch("comfyui_turing_utils.canvas.execution.MiniMaxH3VideoVAEEncode.encode", side_effect=encode):
            for mode in ("edit", "extend"):
                values = {**self.graph["nodes"][2]["values"], "mode": mode, "mode.prefix_frames": 5}
                latent, count, prefix, _ = PrepareH3().prepare(json.dumps(values), None, None,
                    images=torch.rand(5, 64, 64, 3),
                    prefix_images=torch.rand(5, 64, 64, 3) if mode == "extend" else None)
                masks = latent["noise_mask"].unbind()
                self.assertEqual(count, 10 if mode == "extend" else 5)
                self.assertEqual(prefix, 5 if mode == "extend" else 0)
                if mode == "edit":
                    self.assertTrue(torch.all(masks[0] == 1))
