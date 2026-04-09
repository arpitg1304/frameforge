"""Temporal clip sampling strategies."""

from frameforge import VideoReader
from frameforge.sampling import DenseSampler, EpisodeSampler, RandomSampler, UniformSampler

reader = VideoReader("sample.mp4", backend="pyav")

# Uniform: 16 evenly spaced frames across the whole video
uniform = UniformSampler(num_frames=16)
clip = uniform.sample(reader)
print(f"Uniform clip: {clip.shape}")

# Random: 8 frames from a random 2-second window
random_s = RandomSampler(num_frames=8, clip_duration_sec=2.0)
clip = random_s.sample(reader)
print(f"Random clip: {clip.shape}")

# Dense: every 4th frame
dense = DenseSampler(stride=4)
clip = dense.sample(reader)
print(f"Dense clip: {clip.shape}")

# Episode-aware: sample within robotics episode boundaries
episode = EpisodeSampler(episode_boundaries=[0, 100, 250, 400], num_frames=16)
clip = episode.sample(reader)
print(f"Episode clip: {clip.shape}")

reader.close()
