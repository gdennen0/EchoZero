---
name: echozero-demucs
description: Demucs audio source separation in EchoZero. Use when working on SeparatorBlock, Demucs models, audio source separation, or when the user asks about stem separation, Demucs integration, or SeparatorBlock.
---

# Demucs Domain

## What Demucs Does

Separates mixed audio into stems (vocals, drums, bass, other) using deep learning.

## Integration

- **Version:** demucs 4.0.0
- **Approach:** CLI subprocess wrapper in SeparatorBlockProcessor (thin adapter)
- **Models:** htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q
- **Device:** CPU or CUDA GPU (MPS not supported)

## Key Characteristics

- External dependency: users install via pip
- Model downloads: automatic on first use (~1GB)
- Processing: offline, batch-based
- 30-50x faster on GPU; significant RAM required

## Philosophy

- Defer to experts: use Demucs for separation, don't reinvent
- Simple wrapper: thin adapter, not deep integration
- User control: expose key options, keep defaults simple
- Clear errors: explain why and how to fix when Demucs fails
- No magic: subprocess with visible output

## Common Decision Points

| Proposal | Guidance |
|----------|----------|
| Add new model | Yes if officially supported, different use case |
| Expose more options | Yes if clear user value, stable in Demucs API |
| Embed as library | No - CLI approach is simpler |
| Custom separation | No - complex ML, Demucs is state-of-the-art |

## Reference

- Overview: `AgentAssets/modules/domains/demucs/DEMUCS_OVERVIEW.md`
- Implementation: `AgentAssets/modules/domains/demucs/CURRENT_IMPLEMENTATION.md`
- Block: `src/application/blocks/separator_block.py`
