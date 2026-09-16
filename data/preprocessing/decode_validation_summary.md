# FS-Jump3D Decode Validation Summary

## Decode Results

- Validation mode: open_and_first_frame_decode
- Total videos: 2880
- Successfully opened: 2880
- Successfully decoded: 2880
- Unreadable/corrupt videos: 0
- Zero decodable frame videos: 0
- Failed videos: 0

## Frame Count Statistics

- Minimum frame count: 300
- Maximum frame count: 300
- Mean frame count: 300.00

## Duration Statistics

- Minimum duration seconds: 5.0000
- Maximum duration seconds: 5.0000
- Mean duration seconds: 5.0000

## Preprocessing Configuration

- frames_per_clip: `32`
- image_size: `224`
- tensor_shape: `[T, C, H, W]`
- normalization: `[0, 1]`
- short_video_handling: `use all decoded frames in order, then repeat the final valid frame`
- augmentation: `none in deterministic Milestone 3 preprocessing`

## Representative Preprocessing Inspections

| Filepath | Shape | Dtype | Min | Max | Decoded frames | First index | Last index |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| skater_A/cam_1/Axel_1.mp4 | [32, 3, 224, 224] | float32 | 0.0000 | 1.0000 | 300 | 0 | 299 |
| skater_A/cam_1/Axel_10.mp4 | [32, 3, 224, 224] | float32 | 0.0000 | 0.9961 | 300 | 0 | 299 |
| skater_A/cam_1/Axel_2.mp4 | [32, 3, 224, 224] | float32 | 0.0000 | 1.0000 | 300 | 0 | 299 |

## Unusual Files

None
