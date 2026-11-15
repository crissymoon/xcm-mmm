//! The three shipped models, opened once and shared by every reading.
//!
//! Loading is the slow part of a report: the detector compiles its cue
//! patterns, the emotion scanner builds its lookup tables and the analogy
//! index, and the segment artifact carries its centroids. Opening them here
//! keeps the report itself arithmetic.
//!
//! The detectors and the emotion model are required: a report that silently
//! omitted the emotional reading would look complete and not be. The segment
//! artifact is optional and says so, because a text report is still useful
//! before anyone has run `segment fit`.

use std::io;
use std::path::Path;

use crate::detect::Detector;
use crate::emotion::Scanner;
use crate::segment::Artifact as SegmentArtifact;

use super::textmap::{TextMap, TEXTMAP};

/// The models a report reads, with the paths they came from.
pub struct Loaded {
    pub detectors: Detector,
    pub emotion: Scanner,
    pub segments: Option<SegmentArtifact>,
    /// The learned copy map, when one was written beside the segment model.
    pub textmap: Option<TextMap>,
    pub detectors_path: String,
    pub emotion_path: String,
    pub segments_path: String,
    pub textmap_path: String,
}

impl Loaded {
    /// Open the shipped models, or the ones named.
    pub fn open(detectors: &str, emotion: &str, segments: &str) -> io::Result<Self> {
        let detector = Detector::load(detectors)?;
        let scanner = crate::emotion::EmotionModel::load(emotion)?.scanner();
        let segments_artifact = if Path::new(segments).exists() {
            Some(SegmentArtifact::load(segments)?)
        } else {
            None
        };
        // The map lives beside the model it scores: a folder that was fitted
        // together is read together, and a model copied without its map simply
        // reports without one.
        let textmap_path = Path::new(segments)
            .with_file_name(TEXTMAP)
            .to_string_lossy()
            .to_string();
        let textmap = if Path::new(&textmap_path).exists() {
            Some(TextMap::load(&textmap_path)?)
        } else {
            None
        };
        Ok(Self {
            detectors: detector,
            emotion: scanner,
            segments: segments_artifact,
            textmap,
            detectors_path: detectors.to_string(),
            emotion_path: emotion.to_string(),
            segments_path: segments.to_string(),
            textmap_path,
        })
    }
}
