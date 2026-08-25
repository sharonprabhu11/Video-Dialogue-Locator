"""Exception hierarchy for the video dialogue locator.

The pipeline (pipeline.py) is the only place these are caught at the
boundary and converted into a PipelineResult(status="error", ...). Modules
below the pipeline should let these propagate rather than swallowing them.
"""


class VDLError(Exception):
    """Base class for all expected failure modes in this project."""


class AcquisitionError(VDLError):
    """The video URL could not be resolved or downloaded."""


class AudioExtractionError(VDLError):
    """The audio track could not be extracted from the acquired video."""


class TranscriptionError(VDLError):
    """The ASR backend failed to produce a transcript."""


class FrameExtractionError(VDLError):
    """A specific frame could not be decoded/saved from the video."""


class VideoProbeError(VDLError):
    """ffprobe metadata (fps, duration, frame count) could not be determined."""
