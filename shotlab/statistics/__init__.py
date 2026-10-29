"""Headless shot statistics; color decoding is imported only when requested."""

from .model import AnalysisScope, StatisticsParameters, build_context
from .shots import compute_shot_statistics, shot_statistics

def analysis_bundle(shot_result, color_result=None):
    """Compose matching results; reject a stale async result after boundary edits."""
    bundle = dict(metadata=shot_result['metadata'], shot_statistics=shot_result, color_state='not_requested')
    if color_result is not None:
        metadata = color_result['shot_color_statistics']['metadata']
        expected = shot_result['metadata']
        if (metadata['source_identity']['key'], metadata['shot_revision'], metadata['scope']) != (
                expected['source_identity']['key'], expected['shot_revision'], expected['scope']):
            raise ValueError("Color result belongs to a different source, partition or scope")
        if color_result['video_color_timeline']['metadata']['source_identity']['key'] != expected['source_identity']['key']:
            raise ValueError("Color timeline belongs to a different source")
        states = {color_result['video_color_timeline']['status'], color_result['shot_color_statistics']['status']}
        state = 'unsupported' if states == {'unsupported'} else 'partial' if 'unsupported' in states else 'ready'
        bundle.update(color_result, color_state=state)
    return bundle


__all__ = ["AnalysisScope", "StatisticsParameters", "build_context", "shot_statistics", "compute_shot_statistics", "analysis_bundle"]
