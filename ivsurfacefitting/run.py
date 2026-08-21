from ivsurfacefitting.pipeline.base import IVSurfacePipeline

from ivsurfacefitting.models.cross_attn_set_encoder_mlp_decoder import CrossAttnEncodeMLPDecoder
from ivsurfacefitting.metrics.mse import RMSE

pipeline = IVSurfacePipeline(
        ["heston_train"],
        ["heston_test"],
        [CrossAttnEncodeMLPDecoder()],
        [RMSE()],
        )

pipeline.run(forcetrain=True)
