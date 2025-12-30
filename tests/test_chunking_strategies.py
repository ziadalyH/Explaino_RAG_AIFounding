"""Test script to compare chunking strategies."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import VideoTranscript, TranscriptToken
from src.processing.chunking import ChunkingModule
from config.config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load a sample transcript
with open("data/transcripts/machine_learning_intro.json", "r") as f:
    data = json.load(f)

transcript = VideoTranscript(
    video_id=data["video_id"],
    pdf_reference=data["pdf_reference"],
    tokens=[TranscriptToken(**token) for token in data["video_transcripts"]]
)

print(f"\n{'='*80}")
print(f"Testing Chunking Strategies on: {transcript.video_id}")
print(f"Total tokens: {len(transcript.tokens)}")
print(f"Duration: {transcript.tokens[-1].timestamp:.1f} seconds")
print(f"{'='*80}\n")

# Test 1: Sliding Window Strategy
print("=" * 80)
print("STRATEGY 1: Sliding Window (chunk_size=100, overlap=20)")
print("=" * 80)

class MockConfig:
    chunk_size = 100
    chunk_overlap = 20
    chunking_strategy = "sliding_window"
    max_chunk_window = 30

chunker = ChunkingModule(MockConfig(), logger)
chunks_sliding = chunker.chunk_transcript(transcript)

print(f"\nChunks created: {len(chunks_sliding)}")
print(f"\nFirst 3 chunks:")
for i, chunk in enumerate(chunks_sliding[:3]):
    print(f"\nChunk {i+1}:")
    print(f"  Tokens: {chunk.start_token_id} - {chunk.end_token_id}")
    print(f"  Timestamp: {chunk.start_timestamp:.1f}s - {chunk.end_timestamp:.1f}s")
    print(f"  Text: {chunk.text[:80]}...")

# Test 2: All Combinations Strategy (window=10 for demo)
print("\n" + "=" * 80)
print("STRATEGY 2: All Combinations (max_window=10)")
print("=" * 80)

class MockConfig2:
    chunk_size = 100
    chunk_overlap = 20
    chunking_strategy = "all_combinations"
    max_chunk_window = 10

chunker2 = ChunkingModule(MockConfig2(), logger)
chunks_all = chunker2.chunk_transcript(transcript)

print(f"\nChunks created: {len(chunks_all)}")
print(f"\nSample chunks showing different starting positions:")

# Show chunks starting from token 1
print(f"\nChunks starting from token 1:")
for chunk in chunks_all[:5]:
    if chunk.start_token_id == 1:
        print(f"  Tokens {chunk.start_token_id}-{chunk.end_token_id}: {chunk.text}")

# Show chunks starting from token 10
print(f"\nChunks starting from token 10:")
for chunk in chunks_all:
    if chunk.start_token_id == 10 and chunk.end_token_id <= 15:
        print(f"  Tokens {chunk.start_token_id}-{chunk.end_token_id}: {chunk.text}")

# Demonstrate precise retrieval
print("\n" + "=" * 80)
print("DEMONSTRATION: Finding 'machine learning is a subset'")
print("=" * 80)

target_text = "machine learning is a subset"
target_tokens = target_text.split()

print(f"\nTarget phrase: '{target_text}'")
print(f"Looking for chunks that match this phrase...")

# Find matching chunks in all_combinations
matching_chunks = []
for chunk in chunks_all:
    chunk_words = chunk.text.lower().split()
    if all(word in chunk_words for word in target_tokens):
        matching_chunks.append(chunk)

if matching_chunks:
    # Find the shortest matching chunk (most precise)
    best_chunk = min(matching_chunks, key=lambda c: c.end_token_id - c.start_token_id)
    print(f"\n✅ FOUND with all_combinations strategy:")
    print(f"  Tokens: {best_chunk.start_token_id} - {best_chunk.end_token_id}")
    print(f"  Timestamp: {best_chunk.start_timestamp:.1f}s - {best_chunk.end_timestamp:.1f}s")
    print(f"  Duration: {best_chunk.end_timestamp - best_chunk.start_timestamp:.1f}s")
    print(f"  Text: '{best_chunk.text}'")
else:
    print("\n❌ NOT FOUND with all_combinations strategy (window too small)")

# Check sliding window
print(f"\n🔍 With sliding_window strategy:")
for chunk in chunks_sliding:
    if target_text in chunk.text.lower():
        print(f"  Tokens: {chunk.start_token_id} - {chunk.end_token_id}")
        print(f"  Timestamp: {chunk.start_timestamp:.1f}s - {chunk.end_timestamp:.1f}s")
        print(f"  Duration: {chunk.end_timestamp - chunk.start_timestamp:.1f}s")
        print(f"  Text: '{chunk.text[:100]}...'")
        print(f"  ⚠️  Returns entire chunk, not just the relevant segment!")
        break

print("\n" + "=" * 80)
print("COST COMPARISON")
print("=" * 80)

cost_per_embedding = 0.0001  # $0.0001 per embedding

print(f"\nFor this {len(transcript.tokens)}-token transcript:")
print(f"  Sliding window: {len(chunks_sliding)} chunks × ${cost_per_embedding} = ${len(chunks_sliding) * cost_per_embedding:.4f}")
print(f"  All combinations (window=10): {len(chunks_all)} chunks × ${cost_per_embedding} = ${len(chunks_all) * cost_per_embedding:.4f}")

# Extrapolate to longer videos
print(f"\nExtrapolated to a 15-minute video (2,250 tokens):")
ratio_sliding = 2250 / len(transcript.tokens)
ratio_all = 2250 / len(transcript.tokens)
print(f"  Sliding window: ~{int(len(chunks_sliding) * ratio_sliding)} chunks = ${len(chunks_sliding) * ratio_sliding * cost_per_embedding:.2f}")
print(f"  All combinations (window=10): ~{int(len(chunks_all) * ratio_all)} chunks = ${len(chunks_all) * ratio_all * cost_per_embedding:.2f}")
print(f"  All combinations (window=30): ~{int(2250 * 30)} chunks = ${2250 * 30 * cost_per_embedding:.2f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Sliding Window:
  ✅ Very cost-effective
  ✅ Fast indexing
  ❌ Imprecise timestamps (returns entire chunk)
  ❌ Cannot retrieve arbitrary segments

All Combinations:
  ✅ Exact timestamp precision
  ✅ Can retrieve any segment
  ✅ Perfect for video highlighting
  ❌ Higher cost (but manageable)
  ❌ Slower indexing

Recommendation: Use all_combinations with window=30 for accurate timestamp extraction.
""")
