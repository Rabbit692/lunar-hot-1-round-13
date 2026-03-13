from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import numpy as np
from libs.trellis2.representations.mesh.base import MeshWithVoxel
import torch
from PIL import Image

from .settings import TrellisConfig
from config.settings import ModelVersionsConfig
from logger_config import logger
from libs.trellis2.pipelines import Trellis2ImageTo3DPipeline
from .schemas import TrellisRequest, TrellisParams

class TrellisService:
    def __init__(self, trellis_config: TrellisConfig, model_versions: ModelVersionsConfig):
        self.settings = trellis_config
        self.model_versions = model_versions
        self.pipeline: Optional[Trellis2ImageTo3DPipeline] = None
        self.gpu = trellis_config.gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    def _get_model_revisions(self) -> tuple[Optional[str], dict[str, str]]:
        """
        Get pinned revisions for Trellis pipeline and related models.
        
        Returns:
            Tuple of (trellis_revision, model_revisions_dict)
        """
        trellis_revision = self.model_versions.get_revision(self.settings.model_id)
        
        model_revisions = {
            model_id: revision 
            for model_id, revision in self.model_versions.models.items()
            if model_id != self.settings.model_id
        }
        
        return trellis_revision, model_revisions

    async def startup(self) -> None:
        logger.info("Loading TRELLIS.2 pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        trellis_revision, model_revisions = self._get_model_revisions()

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            self.settings.model_id,
            config_file=self.settings.pipeline_config_path,
            revision=trellis_revision,
            model_revisions=model_revisions,
        )
        
        self.pipeline.cuda()
        logger.success(f"{self.settings.model_id} pipeline ready.")

    async def shutdown(self) -> None:
        self.pipeline = None
        torch.cuda.empty_cache()
        logger.info(f"{self.settings.model_id} pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def generate(
        self,
        request: TrellisRequest,
    ) -> MeshWithVoxel:
        if not self.pipeline:
            raise RuntimeError(f"{self.settings.model_id} pipeline not loaded.")

        image_rgb = request.image.convert("RGB")
        
        params = self.default_params.overrided(request.params)
        
        logger.info(f"Generating Trellis {request.seed=} and image size {image_rgb.size} (Using 1 image) | Pipeline: {params.pipeline_type.value} | Max Tokens: {params.max_num_tokens} | {'Mode: ' + params.mode.value if params.mode.value else ''}")
        
        logger.debug(f"Trellis generation parameters: {params}")

        start = time.time()
        try:
            # Run TRELLIS.2 pipeline - returns list of MeshWithVoxel
            meshes = self.pipeline.run(
                image=image_rgb,
                seed=request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "guidance_strength": params.sparse_structure_cfg_strength,
                },
                shape_slat_sampler_params={
                    "steps": params.shape_slat_steps,
                    "guidance_strength": params.shape_slat_cfg_strength,
                },
                tex_slat_sampler_params={
                    "steps": params.tex_slat_steps,
                    "guidance_strength": params.tex_slat_cfg_strength,
                },
                pipeline_type=params.pipeline_type,
                max_num_tokens=params.max_num_tokens,
            )

            mesh = meshes[0]
            mesh.simplify()

            generation_time = time.time() - start
            logger.info(f"{self.settings.model_id} mesh generated in {generation_time:.2f}s")

            total_time = time.time() - start
            logger.success(
                f"{self.settings.model_id} finished in {total_time:.2f}s. "
            )
            return mesh

        finally:
            torch.cuda.empty_cache()
